import argparse
import os
import time
from multiprocessing import freeze_support
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def set_seed(seed):
    """
    设置所有随机种子以确保结果的可重复性。

    参数:
        seed (int): 随机种子值。
    """
    random.seed(seed)  # Python 的随机模块
    np.random.seed(seed)  # NumPy 的随机模块
    torch.manual_seed(seed)  # PyTorch 的 CPU 随机模块
    torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机模块
    torch.cuda.manual_seed_all(seed)  # PyTorch 的多 GPU 随机模块

    # 为了确保 PyTorch 的结果是可重复的
    torch.backends.cudnn.deterministic = True  # 确保使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化



parser = argparse.ArgumentParser(description='MSGNet for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, mask, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int,  default=1, help='status')
parser.add_argument('--model_id', type=str,  default='test', help='model id')
parser.add_argument('--model', type=str,  default='MSGNet',
                    help='model name, options: [Autoformer, Informer, Transformer,MSGNet]')

# data loader
parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,'
                         ' S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task

parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')


parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock/ScaleGraphBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

parser.add_argument('--num_nodes', type=int, default=7, help='to create Graph')
parser.add_argument('--subgraph_size', type=int, default=3, help='neighbors number')
parser.add_argument('--tanhalpha', type=float, default=3, help='')

#GCN
parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
parser.add_argument('--gcn_depth', type=int, default=2, help='')
parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
parser.add_argument('--propalpha', type=float, default=0.3, help='')
parser.add_argument('--conv_channel', type=int, default=32, help='')
parser.add_argument('--skip_channel', type=int, default=32, help='')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default '
                                                              '1: value embedding + temporal embedding + positional embedding '
                                                              '2: value embedding + temporal embedding '
                                                              '3: value embedding + positional embedding '
                                                              '4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=4, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')

parser.add_argument('--patience', type=int, default=50, help='early stopping patience')

parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
# for NIR dataset
parser.add_argument('--data_root', default='/data/home/wxl22/NIROOD/datasets',type=str, help='root for datasets')
parser.add_argument('--data_path', choices=['Chunjian_size', 'Chunjian_area', 'manggo_set', 'manggo_region', 'manggo_HarvestTime', 'manggo_maturity'], default='Chunjian_size', type=str, help='dataset name')
parser.add_argument('--seed', default=2022, type=int, help='random seed')
parser.add_argument('--seq_len', type=int, default=380, help='input sequence length') ###
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    start = time.time()
    mse_seed = []
    mae_seed = []
    mape_seed = []
    seeds = [2022,2024,2025,2027]
    for ii in range(args.itr):
        set_seed(seeds[ii])
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        model, best_ood_mse, best_ood_mae, best_ood_mape = exp.train(setting)
        mse_seed.append(best_ood_mse)
        mae_seed.append(best_ood_mae)
        mape_seed.append(best_ood_mape)

        torch.cuda.empty_cache()
    mse_seed = np.array(mse_seed)
    mae_seed = np.array(mae_seed)
    mape_seed = np.array(mape_seed)

    mse_mean = np.mean(mse_seed)
    mse_std = np.std(mse_seed)
    mae_mean = np.mean(mae_seed)
    mae_std = np.std(mae_seed)
    mape_mean = np.mean(mape_seed)
    mape_std = np.std(mape_seed)

    print(f"Final OOD MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"Final OOD MAE: {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"Final OOD MAPE: {mape_mean:.4f} ± {mape_std:.4f}")

else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
