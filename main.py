from utils.para_config import init_args
from utils.util import set_seed, get_data
import os
import pandas as pd
from utils.NIRDataset import NIROOD
import json
import torch
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.base_models import OneDCNN, LSTM, GRU, MambaNIR, HashMambaNIR
from model.MyModel import Ours
from model.InvarTSModel import InvarTSModel
import torch.optim as optim
import torch.nn as nn
from utils.util import validate_model, Logger, args_print
from utils.generate_pyg import get_pyg_dataset
from torch_geometric.loader import DataLoader as pygDataLoader
from utils.losses import get_irm_loss
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = init_args()
    args.seed = eval(args.seed)
    if args.current_time == None:
        args.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'{args.dataset}-{args.model}-{args.taskMode}-{args.current_time}'
    exp_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    all_seed_info = {
        "train": {
            "r2": [],
            "rmse": [],
            "rpd": [],
            "rer": [],
            "mse": [],
            "mae": [],
            "mape": []
        },
        "test": {
            "r2": [],
            "rmse": [],
            "rpd": [],
            "rer": [],
            "mse": [],
            "mae": [],
            "mape": []
        },
        "ood": {
            "r2": [],
            "rmse": [],
            "rpd": [],
            "rer": [],
            "mse": [],
            "mae": [],
            "mape": []
        }
    }
    for seed in args.seed:
        set_seed(seed)
        root = os.path.join(args.root, args.dataset+".json")
        print(root)
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset {args.dataset} not found in {args.root}")
        with open(root, 'r', encoding='utf-8') as f:
            data = json.load(f)
        iid_df = pd.json_normalize(data['IID'])
        ood_df = pd.json_normalize(data['OOD'])
        iid_data = data["IID"]
        ood_data = data["OOD"]
        if args.dataset.lower() in ["chunjian_size", "chunjian_area", "chunjian_area_rw", "manggo_set", "manggo_region", "manggo_harvesttime", "manggo_maturity", "fina_chunjian_area_rw"]:
            if args.model.lower() not in ["ours"]:
                iid_transmittance_spec, iid_label, wavelength, main_scaler = get_data(iid_data, args.label, args.dataset)
                iid_train, iid_test, iid_train_label, iid_test_label = train_test_split(
                    iid_transmittance_spec, iid_label, test_size=0.2, random_state=seed
                )
                ood_val, ood_val_label, _, _ = get_data(ood_data, args.label, args.dataset, main_scaler)
                train_dataset = NIROOD(iid_train, iid_train_label)
                ood_dataset = NIROOD(ood_val, ood_val_label)
                test_dataset = NIROOD(iid_test, iid_test_label)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, drop_last = False)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            else:
                train_path = os.path.join(args.root, f'{args.dataset}_{args.window_size}_{args.threshold}_train_graphs.pt')
                ood_path = os.path.join(args.root, f'{args.dataset}_{args.window_size}_{args.threshold}_ood_graphs.pt')
                test_path = os.path.join(args.root, f'{args.dataset}_{args.window_size}_{args.threshold}_test_graphs.pt')
                scaler_path = os.path.join(args.root, f'{args.dataset}_{args.window_size}_{args.threshold}_label_scaler.pkl') 
                if os.path.exists(train_path) and os.path.exists(ood_path) and os.path.exists(test_path) and os.path.exists(scaler_path):
                    train_dataset = torch.load(train_path)
                    ood_dataset = torch.load(ood_path)
                    test_dataset = torch.load(test_path)
                    main_scaler = joblib.load(scaler_path)
                else:
                    iid_transmittance_spec, iid_label, wavelength, main_scaler = get_data(iid_data, args.label, args.dataset)
                    iid_train, iid_test, iid_train_label, iid_test_label = train_test_split(
                        iid_transmittance_spec, iid_label, test_size=0.2, random_state=seed
                    )
                    ood_val, ood_val_label, _, _ = get_data(ood_data, args.label, args.dataset, main_scaler)
                    train_dataset, adj_matrix = get_pyg_dataset(iid_train, iid_train_label, window_size = args.window_size, threshold = args.threshold)
                    ood_dataset, _ = get_pyg_dataset(ood_val, ood_val_label,adj_matrix, window_size = args.window_size)
                    test_dataset, _ = get_pyg_dataset(iid_test, iid_test_label,adj_matrix, window_size = args.window_size)
                    torch.save(train_dataset, train_path)
                    torch.save(ood_dataset, ood_path)
                    torch.save(test_dataset, test_path)
                    joblib.dump(main_scaler, scaler_path)
                train_loader = pygDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                ood_loader = pygDataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, drop_last = False)
                test_loader = pygDataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            raise ValueError(f"Dataset {args.dataset} not supported")
        if args.num_envs > 1:
            env_idx = (torch.sigmoid(torch.randn(len(train_loader.dataset))) > 0.5).long()
            print(f"num env 0: {sum(env_idx == 0)} num env 1: {sum(env_idx == 1)}")
        if args.model.lower() == "1dcnn":
            model = OneDCNN(input_dim=iid_transmittance_spec.shape[1])
        elif args.model.lower() == "lstm":
            model = LSTM(bidirection=args.bidirection, input_dim=iid_transmittance_spec.shape[1])
        elif args.model.lower() == "gru":
            model = GRU(bidirection=args.bidirection, input_dim=iid_transmittance_spec.shape[1])
        elif args.model.lower() == 'mambanir':
            model = MambaNIR(bidirection=args.bidirection, input_dim=iid_transmittance_spec.shape[1])
        elif args.model.lower() == 'hashmambanir':
            model = HashMambaNIR(bidirection=args.bidirection, input_dim=iid_transmittance_spec.shape[1], hidden_size=iid_transmittance_spec.shape[1],hash_bit=32)
        elif args.model.lower() == 'invartsmodel':
            model = InvarTSModel(input_dim=iid_transmittance_spec.shape[1])
        elif args.model.lower() == 'ours':
            model = Ours(input_dim=args.window_size, out_dim=1, spec_dim = len(train_dataset[0].spec), args = args, drop_ratio = args.drop_rate)
        else:
            raise ValueError(f"Model {args.model} not supported")
        model.to(device)
        if args.taskMode == "classification":
            criterion = nn.CrossEntropyLoss()
        elif args.taskMode == "regression":
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Task mode {args.taskMode} not supported")
        model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        train_rc, train_rmsec, train_rp, train_rmsep, train_rpd, train_rer = [], [], [], [], [], []
        test_rc, test_rmsec, test_rp, test_rmsep, test_rpd, test_rer = [], [], [], [], [], []
        ood_rc, ood_rmsec, ood_rp, ood_rmsep, ood_rpd, ood_rer = [], [], [], [], [], []
        best_model_weights = []
        Y_trues, Y_preds = [], []
        train_curv, test_curv, ood_curv ={}, {}, {}
        num_batch = (len(train_dataset) // args.batch_size) + int((len(train_dataset) % args.batch_size) > 0)
        for epoch in range(args.epochs):
            train_loss = 0
            model.train()
            for step, batch in tqdm(enumerate(train_loader), total=num_batch, desc=f"Training Epoch {epoch}", ncols=100):
                if args.model.lower() != 'ours':
                    data, labels = batch
                    data, labels = data.to(device), labels.float().to(device)
                else:
                    graph = batch.to(device)
                if args.model.lower() in ["1dcnn", "lstm", "gru", "mambanir", "hashmambanir"]:
                    if args.num_envs > 1:
                        y_pred = model(data)
                        batch_env_idx = env_idx[step * args.batch_size:step * args.batch_size + labels.size(0)]
                        if args.irm_opt.lower() == 'irm':
                            loss = get_irm_loss(y_pred, labels, batch_env_idx, criterion=criterion)
                        elif args.irm_opt.lower() == 'vrex':
                            loss_0 = criterion(y_pred[batch_env_idx == 0], labels[batch_env_idx == 0])
                            loss_1 = criterion(y_pred[batch_env_idx == 1], labels[batch_env_idx == 1])
                            loss = torch.var(torch.stack([loss_0, loss_1]))
                        elif args.irm_opt.lower() == 'ib-irm':
                            ib_penalty = y_pred.var(dim=0).mean()
                            loss = get_irm_loss(y_pred, labels, batch_env_idx,
                                                criterion=criterion) + ib_penalty
                        elif args.irm_opt.lower() == 'eiil':
                            dummy_w = torch.tensor(1.).to(device).requires_grad_()
                            loss = F.mse_loss(y_pred * dummy_w, labels, reduction='none')
                            env_w = torch.randn(batch_env_idx.size(0)).to(device).requires_grad_()
                            optimizer = torch.optim.Adam([env_w], lr=1e-3)
                            for i in range(20):
                                lossa = (loss.squeeze() * env_w.sigmoid()).mean()
                                grada = torch.autograd.grad(lossa, [dummy_w], create_graph=True)[0]
                                penaltya = torch.sum(grada ** 2)
                                lossb = (loss.squeeze() * (1 - env_w.sigmoid())).mean()
                                gradb = torch.autograd.grad(lossb, [dummy_w], create_graph=True)[0]
                                penaltyb = torch.sum(gradb ** 2)
                                npenalty = -torch.stack([penaltya, penaltyb]).mean()
                                optimizer.zero_grad()
                                npenalty.backward(retain_graph=True)
                                optimizer.step()
                            new_batch_env_idx = (env_w.sigmoid() > 0.5).long()
                            env_idx[step * args.batch_size:step * args.batch_size + labels.size(0)] = new_batch_env_idx.to(env_idx.device)
                            loss = get_irm_loss(y_pred, labels, new_batch_env_idx, criterion=criterion)
                    else:
                        y_pred = model(data)
                        loss = criterion(y_pred, labels)
                elif args.model.lower() == 'invartsmodel':
                    y_pred = model(data)
                    loss_1 = criterion(y_pred, labels)
                    reg_term = torch.norm(model.W_ino - 1.0, p=2) ** 2
                    loss = loss_1 + model.lambda_reg * reg_term
                elif args.model.lower() == 'ours':
                    graph.spec = torch.tensor(np.array(graph.spec), dtype=torch.float32).to(device)
                    graph_pred, spec_pred, Ocnn_pred, invMode_loss, constrast_loss = model(batch = graph, return_data = 'rep', args = args)
                    if args.dataset.lower() in ["chunjian_area", "chunjian_area_rw"]:
                        loss = 4 * F.mse_loss(spec_pred, graph.y.float()) + args.spec_coe * F.mse_loss(graph_pred, graph.y.float()) +  args.Odcnn_coe * F.mse_loss(Ocnn_pred, graph.y.float()) + args.invMode_coe * invMode_loss + args.invCont_coe * constrast_loss
                    elif args.dataset.lower() == "manggo_harvesttime":
                        loss = 2 * F.mse_loss(spec_pred, graph.y.float()) + args.spec_coe * F.mse_loss(graph_pred, graph.y.float()) +  args.Odcnn_coe * F.mse_loss(Ocnn_pred, graph.y.float()) + args.invMode_coe * invMode_loss + args.invCont_coe * constrast_loss
                    else:
                        loss = F.mse_loss(spec_pred, graph.y.float()) + args.spec_coe * F.mse_loss(graph_pred, graph.y.float()) +  args.Odcnn_coe * F.mse_loss(Ocnn_pred, graph.y.float()) + args.invMode_coe * invMode_loss + args.invCont_coe * constrast_loss
                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            train_loss, train_perf = validate_model(model, train_loader, criterion, device, args, main_scaler)
            test_loss, test_perf = validate_model(model, test_loader, criterion, device, args, main_scaler)
            ood_loss, ood_perf, Y_true, Y_pred = validate_model(model, ood_loader, criterion, device, args, main_scaler, return_pred = "yes")
            best_model_weights.append(deepcopy(model.state_dict()))
            Y_trues.append(Y_true)
            Y_preds.append(Y_pred)
            for k, v in test_perf.items():
                if k not in test_curv:
                    train_curv[k], test_curv[k], ood_curv[k] = [], [], []
                train_curv[k].append(train_perf[k])
                test_curv[k].append(test_perf[k])
                ood_curv[k].append(ood_perf[k])
            logger.info(f"Epoch {epoch} train loss: {train_loss}, train perf: {train_perf}, test loss: {test_loss}, test perf: {test_perf}, ood loss: {ood_loss}, ood perf: {ood_perf}")
        best_result = {}
        pos= 0
        for k, v in test_curv.items():
            if k == args.metric:
                v = v[args.drop_early:]
                train_curv[k] = train_curv[k][args.drop_early:]
                test_curv[k] = test_curv[k][args.drop_early:]
                ood_curv[k] = ood_curv[k][args.drop_early:]
                if args.taskMode == "regression":
                    pos = int(np.argmin(v))
                else:
                    pos = int(np.argmax(v))
                best_result[k] = [pos, v[pos], train_curv[k][pos], test_curv[k][pos], ood_curv[k][pos]]
                best_model_weights = best_model_weights[pos]
                Y_pred_best = Y_preds[pos]
                Y_true_best = Y_trues[pos]
        torch.save({'Y_true':Y_true_best, 'Y_pred':Y_pred_best}, os.path.join(exp_dir, 'results.pt'))
        for metric in ['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae','mape']:
            all_seed_info['train'][metric].append(train_curv[metric][pos])
            all_seed_info['test'][metric].append(test_curv[metric][pos])
            all_seed_info['ood'][metric].append(ood_curv[metric][pos])
        logger.info(f"Best result: {best_result}")
        if args.save_model:
            print("Saving best weights..")
            torch.save(best_model_weights, exp_dir + f"/{args.dataset}_{args.model}_best_model_weights_{seed}.pth")
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(1, len(test_curv['mse']) + 1)), test_curv['mse'], '-', label='Train', color='blue')
        plt.plot(list(range(1, len(test_curv['mse']) + 1)), ood_curv['mse'], '-', label='Valid', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"{seed}_vis.png"), dpi=900)
    results = {
        "train": {
            "r2":0,
            "rmse": 0,
            "rpd": 0,
            "rer": 0,
            "mse": 0,
            "mae": 0,
            "mape": 0
        },
        "test": {
            "r2": 0,
            "rmse": 0,
            "rpd": 0,
            "rer": 0,
            "mse": 0,
            "mae": 0,
            "mape": 0
        },
        "ood": {
            "r2": 0,
            "rmse": 0,
            "rpd": 0,
            "rer": 0,
            "mse": 0,
            "mae": 0,
            "mape": 0
        }
    }
    for key in ['train', 'test', 'ood']:
        for metric in ['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae', 'mape']:
            results[key][metric] = {
                'mean': np.mean(all_seed_info[key][metric]),
                'std': np.std(all_seed_info[key][metric])
            }
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(exp_dir, "done.flag"), "w") as f:
        f.write("done")
    logger.info(f"Results: {results}")
    print('[INFO]: END')


