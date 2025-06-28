import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm
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



def get_data(root, label, dataset, scaler_train=None):
    Transmittance_spec = []
    labels = []
    if dataset.lower() in ["manggo_set", "manggo_region", "manggo_harvesttime", "manggo_maturity"]:
        for i in range(len(root)):
            transmittance = eval(root[i]["spec"])
            Transmittance_spec.append(transmittance)
            labels.append(eval(root[i]['DM']))
    elif dataset.lower() in ["chunjian_size", "chunjian_area", "chunjian_area_rw", "fina_chunjian_area_rw"]:
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


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(trueYAll, predYAll, metric=['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae', 'mape']):
    """
    评估回归模型的性能，计算并返回不同的评估指标。

    参数:
    - trueYAll: 真实标签列表（经过 .cpu().numpy() 处理后）
    - predYAll: 模型预测值列表（经过 .cpu().numpy() 处理后）
    - metric: 需要计算的评估指标（默认是 ['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae']）

    返回:
    - result: 各种评估指标的字典
    """
    # 转换为 numpy 数组
    # trueYAll = np.concatenate(trueYAll, axis=0)
    # predYAll = np.concatenate(predYAll, axis=0)

    result = {}

    if 'r2' in metric:
        result['r2'] = r2_score(trueYAll, predYAll)

    if 'rmse' in metric:
        rmse = np.sqrt(mean_squared_error(trueYAll, predYAll))
        result['rmse'] = rmse

    if 'rpd' in metric:
        std_true = np.std(trueYAll)
        result['rpd'] = std_true / result.get('rmse', np.sqrt(mean_squared_error(trueYAll, predYAll)))

    if 'rer' in metric:
        mean_true = np.mean(trueYAll)
        result['rer'] = mean_true / result.get('rmse', np.sqrt(mean_squared_error(trueYAll, predYAll)))

    if 'mse' in metric:
        result['mse'] = mean_squared_error(trueYAll, predYAll)

    if 'mae' in metric:
        result['mae'] = mean_absolute_error(trueYAll, predYAll)

    if 'mape' in metric:
        result['mape'] = np.mean(np.abs((np.array(trueYAll) - np.array(predYAll)) / np.array(trueYAll))) * 100

    return result

def validate_model(model, val_loader, criterion, device, args, label_scaler, return_pred = None):
    model.eval()
    val_loss = 0
    trueYAll = []
    predYAll = []

    with torch.no_grad():
        for batch in val_loader:
            if args.model.lower() != 'ours':
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                y_pred = model(data)
                loss = criterion(y_pred, labels)

                # 反归一化
                y_pred_np = y_pred.detach().cpu().numpy().reshape(-1, 1)
                y_true_np = labels.detach().cpu().numpy().reshape(-1, 1)
            else:
                graph = batch.to(device)
                y_pred = model(graph, args)
                loss = criterion(y_pred, graph.y)

                # 反归一化（注意 graph.y 也是归一化后的）
                y_pred_np = y_pred.detach().cpu().numpy().reshape(-1, 1)
                y_true_np = graph.y.detach().cpu().numpy().reshape(-1, 1)

            # 使用 label_scaler 恢复原始值
            y_pred_orig = label_scaler.inverse_transform(y_pred_np).squeeze()
            y_true_orig = label_scaler.inverse_transform(y_true_np).squeeze()
            

            trueYAll.extend(y_true_orig.tolist())
            predYAll.extend(y_pred_orig.tolist())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    eval_result = evaluate(trueYAll, predYAll, metric=['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae', 'mape'])
    if return_pred == None:
        return avg_val_loss, eval_result
    else:
        return avg_val_loss, eval_result, trueYAll, predYAll


# def validate_model(model, val_loader, criterion, device, args):
#     model.eval()  # 设置模型为评估模式
#     val_loss = 0
#     trueYAll = []
#     predYAll = []
#     with torch.no_grad():  # 不计算梯度
#         for batch in val_loader:
#             if args.model.lower() != 'ours':
#                 data, labels = batch
#                 data, labels = data.to(device), labels.to(device)
#                 # 前向传播
#                 y_pred = model(data)
#                 loss = criterion(y_pred, labels)
#                 trueYAll.append(labels.detach().cpu().numpy().tolist())
#                 predYAll.append(y_pred.detach().cpu().numpy().tolist())
#             else:
#                 graph = batch.to(device)
#                 y_pred = model(graph, args)
#                 loss = criterion(y_pred, graph.y.float())
#                 trueYAll.append(graph.y.float().detach().cpu().numpy().tolist())
#                 predYAll.append(y_pred.detach().cpu().numpy().tolist())
            
#             # 累加验证损失
#             val_loss += loss.item()
#     # 计算每个 epoch 的平均验证损失
#     avg_val_loss = val_loss / len(val_loader)
#     return avg_val_loss, evaluate(trueYAll, predYAll, metric=['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae'])




import logging
import os
import sys
from texttable import Texttable

class Logger:
    logger = None

    @staticmethod
    def get_logger(filename: str = None):
        if not Logger.logger:
            Logger.init_logger(filename=filename)
        return Logger.logger

    @staticmethod
    def init_logger(
            level=logging.INFO,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: \n %(message)s',
            filename: str = None):
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        fmt = logging.Formatter(fmt)
        
        if os.path.exists(filename):
            os.remove(filename)

        # file handler
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # stream handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.setLevel(level)
        Logger.logger = logger
        return logger


def args_print(args, logger):
    print('\n')
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())

