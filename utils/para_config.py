import argparse


def init_args():
    parser = argparse.ArgumentParser('graph Mutual Information for OOD')
    # base config
    parser.add_argument('--device', default=1, type=int, help='cuda device')
    parser.add_argument('--root', default='./NIROOD/datasets',type=str, help='root for datasets')
    parser.add_argument('--check_point', default='./NIROOD/check_point',type=str, help='check point for datasets')
    parser.add_argument('--seed', default="[2022, 2024, 2025, 2027]", type=str, help='random seed')
    parser.add_argument('--dataset', choices=['Chunjian_size', 'Chunjian_area', 'Fina_Chunjian_area_rw', 'Chunjian_area_rw', 'manggo_set', 'manggo_region', 'manggo_HarvestTime', 'manggo_maturity'], default='Chunjian_area', type=str, help='dataset name')
    parser.add_argument('--label', default='brix', type=str, help='label name')
    parser.add_argument('--taskMode', default='regression', type=str, help='task mode')
    parser.add_argument('--log_dir', default='./NIROOD/logs', type=str, help='log directory')
    parser.add_argument('--metric', default='mse', type=str, help='metric')
    parser.add_argument('--save_model', default=True, type=bool, help='save model')
    parser.add_argument('--drop_early', default=0, type=int, help='drop early')
    # training config
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training iterations')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate for the predictor')
    # model config
    parser.add_argument('--model', choices=['1dcnn','lstm','gru', 'mambanir', 'hashmambanir', 'invartsmodel', 'ours'], default='ours', type=str, help='model name')
    parser.add_argument('--bidirection', default=False, type=bool, help='lstm or gru or mamba bidirectional')
    #invariant model settings
    parser.add_argument('--num_envs', default=1, type=int, help='num of envs need to be partitioned')
    parser.add_argument('--irm_opt', choices=['irm', 'vrex', 'ib-irm', 'eiil'], default="eiil", help='irm algorithms to use')
    #Ours config
    parser.add_argument('--threshold', default=0.05, type=float, help='the threshold for generating graph')
    parser.add_argument('--window_size', default=2, type=int, help="the number of windows for node")
    parser.add_argument('--rate', type=float, default=0.75, help="select ratio of graph_rep")
    parser.add_argument('--batch_rate', type=float, default=0.1)
    parser.add_argument('--spec_coe', type=float, default=1)
    parser.add_argument('--Odcnn_coe', type=float, default=1)
    parser.add_argument('--invMode_coe', type=float, default=1)
    parser.add_argument('--invCont_coe', type=float, default=1.6)
    parser.add_argument('--drop_rate', type=float, default=0.4)
    #graid search
    parser.add_argument('--current_time', type=str, default=None)

    args = parser.parse_args()
    return args