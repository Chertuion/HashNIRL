# This script was first copied from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py under the license
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Then we included our new regularization loss Fishr. To do so:
# 1. we first compute gradients covariance on each domain (see compute_grads_variance method) using BackPACK package
# 2. then, we compute l2 distance between these gradient covariances (see l2_between_grads_variance method)
import time
import random
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from models import OneDCNN
from util import get_data
from NIRDataset import NIROOD
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import datasets
from torch import nn, optim, autograd
import json
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import os
from infonce import compute_div_penalty
from metric import calculate_erank, calculate_patch_sim, cal_variance

parser = argparse.ArgumentParser(description='Colored MNIST')

# select your algorithm
parser.add_argument(
    '--algorithm',
    type=str,
    default="erm",
    choices=[
        ## Four main methods, for Table 2 in Section 6.A
        'erm',  # Empirical Risk Minimization
        'irm',  # Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
        'rex',  # Out-of-Distribution Generalization via Risk Extrapolation (https://icml.cc/virtual/2021/oral/9186)
        'fishr',  # Our proposed Fishr
        ## two Fishr variants, for Table 6 in Appendix C.2.4
        'fishr_offdiagonal'  # Fishr but on the full covariance rather than only the diagonal
        'fishr_notcentered',  # Fishr but without centering the gradient variances
    ]
)
# select whether you want to apply label flipping or not
# Set to 0 in Table 5 in Appendix C.2.3 and in the right half of Table 6 in Appendix C.2.4
parser.add_argument('--label_flipping_prob', type=float, default=0.25)

# Following hyperparameters are directly taken from from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/reproduce_paper_results.sh
# They should not be modified except in case of a new proper hyperparameter search with an external validation dataset.
# Overall, we compare all approaches using the hyperparameters optimized for IRM.
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--penalty_anneal_iters', type=int, default=190)
parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
parser.add_argument('--steps', type=int, default=10)
# experimental setup
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--n_restarts', type=int, default=4)
parser.add_argument('--seed', type=int, default=0, help='Seed for everything')
parser.add_argument('--ssl_weight', type=float, default=0.01, help='SSL weight')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='SSL Temp')
parser.add_argument('--proj_mask', type=float, default=0.5, help='SSL proj_mask')
parser.add_argument('--output_path', type=str, default='experiment_results.csv')
# NIROOD setup
parser.add_argument('--root', default='/data/home/wxl22/NIROOD/datasets',type=str, help='root for datasets')
parser.add_argument('--dataset', choices=['Chunjian_size', 'Chunjian_area', 'manggo_set', 'manggo_region', 'manggo_HarvestTime', 'manggo_maturity', 'Chunjian_area_rw'], default='Chunjian_area_rw', type=str, help='dataset name')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--device', default=0, type=int, help='cuda device')
flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

# random.seed(flags.seed)
# np.random.seed(flags.seed)
# torch.manual_seed(flags.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

final_train_accs = []
final_test_accs = []
final_graytest_accs = []

metric_train_similarity = []
metric_train_erank = []
metric_test_similarity = []
metric_test_erank = []
metric_graytest_similarity = []
metric_graytest_erank = []
device = torch.device(f"cuda:{flags.device}") if torch.cuda.is_available() else torch.device("cpu")
all_best_test_mse = []
all_best_test_mae = []
all_best_ood_mse = []
all_best_ood_mae = []
all_best_ood_mape = []
seeds = [2022,2024,2025,2027]
for restart in range(flags.n_restarts):
    print("Restart", restart)
    set_seed(seeds[restart])
    # Load MNIST, make train/val splits, and shuffle train set examples
    root = os.path.join(flags.root, flags.dataset+".json")
    print(root)
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset {flags.dataset} not found in {flags.root}")

    # 读取 JSON 文件并转换为字典
    with open(root, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 分别处理 'IID' 和 'OOD' 列表
    iid_df = pd.json_normalize(data['IID'])
    ood_df = pd.json_normalize(data['RWD'])
    iid_data = data["IID"]
    ood_data = data["RWD"]
    if flags.dataset.lower() in ["chunjian_size", "chunjian_area", "manggo_set", "manggo_region", "manggo_harvesttime", "manggo_maturity", "chunjian_area_rw"]:
        iid_transmittance_spec, iid_label, wavelength, main_scaler = get_data(iid_data, flags.dataset)
        # 数据集划分
        iid_train, iid_test, iid_train_label, iid_test_label = train_test_split(
            iid_transmittance_spec, iid_label, test_size=0.2, random_state=seeds[restart]
        )
        ood_val, ood_val_label, _, _ = get_data(ood_data, flags.dataset, main_scaler)
        train_dataset = NIROOD(iid_train, iid_train_label)
        ood_dataset = NIROOD(ood_val, ood_val_label)
        test_dataset = NIROOD(iid_test, iid_test_label)

        train_loader = DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
        ood_loader = DataLoader(ood_dataset, batch_size=flags.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=True)

    mlp = OneDCNN(input_dim = iid_transmittance_spec.shape[1]).to(device)
    mlp.classifier = extend(mlp.classifier)
    env_idx = (torch.sigmoid(torch.randn(len(train_loader.dataset))) > 0.5).long()
    print(f"num env 0: {sum(env_idx == 0)} num env 1: {sum(env_idx == 1)}")


    def mean_nll(logits, y):
        # return nn.functional.binary_cross_entropy_with_logits(logits, y)
        return nn.functional.mse_loss(logits.squeeze(), y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def compute_irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    # bce_extended = extend(nn.BCEWithLogitsLoss())
    mse_extended = extend(nn.MSELoss())

    def compute_grads_variance(features, labels, classifier):
        # bce_extended = extend(nn.BCEWithLogitsLoss())  # 也要包装 Loss
        mse_extended = extend(nn.MSELoss())
        logits = classifier(features)
        loss = mse_extended(logits.squeeze(), labels)

        with backpack(BatchGrad()):
            loss.backward(retain_graph=True, create_graph=True
            )

        # dict_grads = OrderedDict(
        #     [
        #         (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
        #         for name, weights in classifier.named_parameters()
        #     ]
        # )

        dict_grads = OrderedDict()
        for name, weights in classifier.named_parameters():
            # 检查是否有 grad_batch 属性（确保被 extend 过并处于 BackPACK 上下文）
            # if not hasattr(weights, "grad_batch"):
            #     raise AttributeError(f"Parameter '{name}' does not have 'grad_batch'. Make sure you used 'extend()' and called backward inside 'with backpack(...)'.")

            grad_batch = weights.grad_batch  # 每个样本的梯度，形状：[batch_size, num_params]
            grad_batch = grad_batch.clone()  # 克隆防止修改原数据
            grad_batch = grad_batch.view(grad_batch.size(0), -1)  # 保证展平为 [B, D]
            dict_grads[name] = grad_batch


        dict_grads_variance = {}
        for name, grads in dict_grads.items():
            grads = grads * labels.size(0)  # 放大
            env_mean = grads.mean(dim=0, keepdim=True)
            if flags.algorithm != "fishr_notcentered":
                grads = grads - env_mean
            if flags.algorithm == "fishr_offdiagonal":
                dict_grads_variance[name] = torch.einsum("na,nb->ab", grads, grads) / (grads.size(0) * grads.size(1))
            else:
                dict_grads_variance[name] = grads.pow(2).mean(dim=0)

        return dict_grads_variance


    def l2_between_grads_variance(cov_1, cov_2):
        assert len(cov_1) == len(cov_2)
        cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
        cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
        ).pow(2).sum()

    # Train loop
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)



    # 训练环境划分（每个 batch 中动态构造）
    def get_env_splits(data, labels, env_idx):
        envs = []
        for env_id in [0, 1]:
            mask = (env_idx == env_id)
            if mask.sum() == 0:
                continue
            env_data = data[mask]
            env_labels = labels[mask]
            features, logits = mlp(env_data, return_data="feat")
            nll = mean_nll(logits, env_labels)
            acc = mean_accuracy(logits, env_labels)
            irm = compute_irm_penalty(logits, env_labels)
            ssl_loss = compute_div_penalty(mlp.proj, mlp._main, features, env_data, flags.ssl_temp, flags.proj_mask) if flags.ssl_weight != 0 else torch.tensor(0.)
            grads_var = compute_grads_variance(features, env_labels, mlp.classifier)
            erank = calculate_erank(features)
            sim = calculate_patch_sim(features)
            return {
                'features': features,
                'logits': logits,
                'labels': env_labels,
                'nll': nll,
                'acc': acc,
                'irm': irm,
                'ssl_loss': ssl_loss,
                'grads_variance': grads_var,
                'erank': erank,
                'similarity': sim,
            }

    # 获取测试集指标
    # def eval_env(loader, label_scaler):
    #     mlp.eval()
    #     all_logits, all_labels, all_features = [], [], []
    #     with torch.no_grad():
    #         for x, y in loader:
    #             x, y = x.to(device), y.float().to(device)
    #             f, logits = mlp(x, return_data="feat")
    #             # 反归一化
    #             f = f.detach().cpu().numpy().reshape(-1, 1)
    #             y = y.detach().cpu().numpy().reshape(-1, 1)
                
    #             all_logits.append(logits)
    #             all_labels.append(y)
    #             all_features.append(f)

    #     logits = torch.cat(all_logits).squeeze()
    #     labels = torch.cat(all_labels).squeeze()
    #     features = torch.cat(all_features)

    #     # 回归指标（转为 CPU + NumPy）
    #     y_pred = logits.detach().cpu().numpy()
    #     y_true = labels.detach().cpu().numpy()
    #     mse = mean_squared_error(y_true, y_pred)
    #     mae = mean_absolute_error(y_true, y_pred)

    #     # 表征指标
    #     erank = calculate_erank(features)
    #     sim = calculate_patch_sim(features)

    #     return mse, mae, erank, sim

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def eval_env(loader, label_scaler, return_pred = None):
        mlp.eval()
        all_logits, all_labels, all_features = [], [], []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.float().to(device)
                f, logits = mlp(x, return_data="feat")

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())
                all_features.append(f.detach().cpu())

        # 拼接所有 batch
        logits = torch.cat(all_logits).squeeze()      # [N]
        labels = torch.cat(all_labels).squeeze()      # [N]
        features = torch.cat(all_features)            # [N, D]

        # ========= 标签 & 预测结果反归一化 =========
        y_pred = logits.numpy().reshape(-1, 1)
        y_true = labels.numpy().reshape(-1, 1)

        y_pred_orig = label_scaler.inverse_transform(y_pred).squeeze()
        y_true_orig = label_scaler.inverse_transform(y_true).squeeze()

        # ========= 回归指标（原始尺度） =========
        mse = mean_squared_error(y_true_orig, y_pred_orig)
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig))) * 100

        # ========= 表征指标 =========
        erank = calculate_erank(features)
        sim = calculate_patch_sim(features)
        if return_pred == None:
            return mse, mae, erank, sim, mape
        else:
            return mse, mae, erank, sim, mape, y_true_orig, y_pred_orig

    best_test_mse = float("inf")
    best_metrics = {}
    for step in range(flags.steps):
        for i, batch in enumerate(train_loader):
            data, labels = batch
            data, labels = data.to(device), labels.float().to(device)
            # batch_env_idx = env_idx[:len(labels)]
            batch_env_idx = env_idx[i * flags.batch_size:i * flags.batch_size + labels.size(0)]
            env0 = get_env_splits(data, labels, batch_env_idx == 0)
            env1 = get_env_splits(data, labels, batch_env_idx == 1)

            train_nll = (env0['nll'] + env1['nll']) / 2
            train_acc = (env0['acc'] + env1['acc']) / 2
            train_erank = (env0['erank'] + env1['erank']) / 2
            train_similarity = (env0['similarity'] + env1['similarity']) / 2

            # weight decay
            weight_norm = torch.tensor(0.).to(device)
            for name, w in mlp.named_parameters():
                if w.requires_grad and "proj" not in name:
                    weight_norm += w.norm().pow(2)

            loss = train_nll + flags.l2_regularizer_weight * weight_norm

            # 正则项计算
            irm_penalty = (env0['irm'] + env1['irm']) / 2
            rex_penalty = (env0['nll'] - env1['nll']).pow(2)
            ssl_penalty = (env0['ssl_loss'] + env1['ssl_loss']) / 2 if flags.ssl_weight != 0 else torch.tensor(0.)
            
            dict_grads_variance_avg = OrderedDict({
                name: torch.stack([env0['grads_variance'][name], env1['grads_variance'][name]]).mean(dim=0)
                for name in env0['grads_variance']
            })
            fishr_penalty = (
                l2_between_grads_variance(env0['grads_variance'], dict_grads_variance_avg) +
                l2_between_grads_variance(env1['grads_variance'], dict_grads_variance_avg)
            )

            # 选择正则项并加权
            if flags.algorithm == "irm":
                train_penalty = irm_penalty
            elif flags.algorithm == "rex":
                train_penalty = rex_penalty
            elif flags.algorithm.startswith("fishr"):
                train_penalty = fishr_penalty
            else:
                train_penalty = torch.tensor(0.).to(device)

            penalty_weight = flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= penalty_weight
            if flags.ssl_weight != 0:
                loss += flags.ssl_weight * ssl_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 测试集评估
            test_mse, test_mae, test_erank, test_sim, test_mape = eval_env(test_loader, main_scaler)
            ood_mse, ood_mae, ood_erank, ood_sim, ood_mape, y_true, y_pred = eval_env(ood_loader, main_scaler, return_pred="yes")

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                torch.save({'Y_true':y_true, 'Y_pred':y_pred}, os.path.join('/data/home/wxl22/NIROOD/logs/best_result/rw', 'divil.pt'))
                best_metrics = {
                    "step": step,
                    "test_mse": test_mse.item(),
                    "test_mae": test_mae,
                    "test_erank": test_erank,
                    "test_sim": test_sim,
                    "ood_mse": ood_mse.item(),
                    "ood_mae": ood_mae,
                    "ood_erank": ood_erank,
                    "ood_sim": ood_sim,
                    "ood_mape": ood_mape
                }
                # torch.save(mlp.state_dict(), f"best_model_{flags.algorithm}.pth")  # ✅ 可选：保存模型
        print(f"Step: {step}")
        print(best_metrics)
    print("\n=== Best Test Performance ===")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")
    all_best_test_mse.append(best_metrics["test_mse"])
    all_best_test_mae.append(best_metrics["test_mae"])
    all_best_ood_mse.append(best_metrics["ood_mse"])
    all_best_ood_mae.append(best_metrics["ood_mae"])
    all_best_ood_mape.append(best_metrics["ood_mape"])


print("\n=== Final Summary Across Restarts ===")
print(f"Test MSE (mean ± std): {np.mean(all_best_test_mse):.5f} ± {np.std(all_best_test_mse):.5f}")
print(f"Test MAE (mean ± std): {np.mean(all_best_test_mae):.5f} ± {np.std(all_best_test_mae):.5f}")
print(f"OOD MSE (mean ± std): {np.mean(all_best_ood_mse):.5f} ± {np.std(all_best_ood_mse):.5f}")
print(f"OOD MAE (mean ± std): {np.mean(all_best_ood_mae):.5f} ± {np.std(all_best_ood_mae):.5f}")
print(f"OOD MAPE (mean ± std): {np.mean(all_best_ood_mape):.5f} ± {np.std(all_best_ood_mape):.5f}")


