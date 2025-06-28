from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import Informer, Autoformer, DLinear, MSGNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import torch
import torch.nn as nn
from torch import optim, autograd

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from torch.utils.data import Dataset
# 自定义 Dataset 类
class NIROOD(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], idx


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'MSGNet': MSGNet
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    #flag = 'train' or 'val' or 'test'
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_NIR_row_data(self, root, dataset, scaler_train=None):
        Transmittance_spec = []
        labels = []
        if dataset.lower() in ["manggo_set", "manggo_region", "manggo_harvesttime", "manggo_maturity"]:
            for i in range(len(root)):
                transmittance = eval(root[i]["spec"])
                Transmittance_spec.append(transmittance)
                labels.append(eval(root[i]['DM']))
        elif dataset.lower() in ["chunjian_size", "chunjian_area"]:
            for i in range(len(root)):
                transmittance = eval(root[i]["processed_Transmittance_spec"])[80:]
                Transmittance_spec.append(transmittance)
                labels.append(eval(root[i]["brix"]))
        scaler = StandardScaler()
        transmittance_scaled = scaler.fit_transform(Transmittance_spec)
        labels = np.array(labels)
        if scaler_train == None:
            label_scaler = StandardScaler()
        else:
            label_scaler = scaler_train
        y_scaled = label_scaler.fit_transform(labels.reshape(-1, 1)).squeeze()
        return transmittance_scaled, y_scaled, eval(root[i]["wavelength"]), label_scaler

    def _get_NIR_data(self, flag = "None"):
        if flag == "train":
            root = os.path.join(self.args.data_root, self.args.data_path+".json")
            print(root)
            if not os.path.exists(root):
                raise FileNotFoundError(f"Dataset {self.args.data_root} not found in {self.args.root_path}")

            # 读取 JSON 文件并转换为字典
            with open(root, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 分别处理 'IID' 和 'OOD' 列表
            iid_df = pd.json_normalize(data['IID'])
            ood_df = pd.json_normalize(data['OOD'])
            iid_data = data["IID"]
            ood_data = data["OOD"]
            iid_transmittance_spec, iid_label, wavelength, self.label_scaler = self._get_NIR_row_data(iid_data, self.args.data_path)
            # 数据集划分
            iid_train, iid_test, iid_train_label, iid_test_label = train_test_split(
                iid_transmittance_spec, iid_label, test_size=0.2, random_state=self.args.seed
            )
            ood_val, ood_val_label, _, _ = self._get_NIR_row_data(ood_data, self.args.data_path, self.label_scaler)
            self.train_dataset = NIROOD(iid_train, iid_train_label)
            self.ood_dataset = NIROOD(ood_val, ood_val_label)
            self.test_dataset = NIROOD(iid_test, iid_test_label)

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.ood_loader = DataLoader(self.ood_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        if flag == "train":
            return self.train_dataset, self.train_loader, self.ood_dataset, self.ood_loader, self.test_dataset, self.test_loader, self.label_scaler
        elif flag == "val":
            return self.test_dataset, self.test_loader, self.label_scaler
        elif flag == "test":
            return self.ood_dataset, self.ood_loader, self.label_scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    import torch.nn.functional as F
    import numpy as np

    def vali(self, vali_data, vali_loader, criterion, label_scaler):
        total_loss = []
        all_preds = []
        all_trues = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, _ = batch
                batch_x, batch_y = batch_x.to(self.device), batch_y.float().to(self.device)

                # 模型推理
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)[0] if self.args.output_attention else self.model(batch_x)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)[0] if self.args.output_attention else self.model(batch_x)

                # 归一化下的 loss
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                # 收集预测值和真实值
                all_preds.append(outputs.detach().cpu().numpy().reshape(-1, 1))
                all_trues.append(batch_y.detach().cpu().numpy().reshape(-1, 1))

        # 合并所有 batch
        all_preds = np.concatenate(all_preds, axis=0)  # shape: (N, 1) or (N, seq_len, d)
        all_trues = np.concatenate(all_trues, axis=0)

        # 反归一化
        all_preds_denorm = label_scaler.inverse_transform(all_preds)
        all_trues_denorm = label_scaler.inverse_transform(all_trues)

        # 计算指标
        avg_loss = np.average(total_loss)
        mse = np.mean((all_preds_denorm - all_trues_denorm) ** 2)
        mae = np.mean(np.abs(all_preds_denorm - all_trues_denorm))
        mape = np.mean(np.abs((all_trues_denorm - all_preds_denorm) / (all_trues_denorm))) * 100

        self.model.train()
        return avg_loss, mse, mae, mape



    def train(self, setting):
        train_data, train_loader, ood_data, ood_loader, vali_data, vali_loader, label_scaler = self._get_NIR_data(flag="train")



        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        #use automatic mixed precision training
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            best_val_mse = float('inf')
            best_ood_mse = None
            best_ood_mae = None
            best_ood_mape = None
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                

                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, _ = batch
                batch_x, batch_y = batch_x.to(self.device), batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x)[0]
                            else:
                                outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            # print("Linear")
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention: #whether to output attention in ecoder
                            outputs = self.model(batch_x)[0]
                            
                        else:
                            outputs = self.model(batch_x).squeeze()
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    with autograd.detect_anomaly():
                        loss.backward()
                        model_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {} cost time: {} train loss: {}".format(epoch + 1, time.time() - epoch_time, train_loss))
            vali_loss, val_mse, val_mae, val_mape = self.vali(vali_data, vali_loader, criterion, label_scaler)
            test_loss, ood_mse, ood_mae, ood_mape = self.vali(ood_data, ood_loader, criterion, label_scaler)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # === 更新 best val mse & ood 性能 ===
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_ood_mse = ood_mse
                best_ood_mae = ood_mae
                best_ood_mape = ood_mape
                print(f"✓ New best val MSE: {best_val_mse:.6f} → saving corresponding OOD MSE: {best_ood_mse:.6f}, MAE: {best_ood_mae:.6f}")

            # early stopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        # === 重新加载最佳模型 ===
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # === 返回模型和 best OOD 性能 ===
        return self.model, best_ood_mse, best_ood_mae, best_ood_mape

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        #See utils / tools for usage
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        # print('preds_shape:', len(preds),len(preds[0]),len(preds[1]))

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        print('preds_shape:', preds.shape)
        print('trues_shape:', trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, nd, nrmse = metric(preds, trues)
        print('nd:{}, nrmse:{}, mse:{}, mae:{}, rse:{}, mape:{}'.format(nd, nrmse,mse, mae, rse, mape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('nd:{}, nrmse:{}, mse:{}, mae:{}, rse:{}, mape:{}'.format(nd, nrmse,mse, mae, rse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
