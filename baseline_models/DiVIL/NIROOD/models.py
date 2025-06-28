import torch.nn as nn

class TOneDCNN(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(TOneDCNN, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv1d(1, hidden_size//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size//2, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)  # 增加额外卷积层

        # 计算卷积层输出的特征维度
        self.fc1 = nn.Linear(hidden_size * 2 * input_dim, hidden_size)  # 输出通道数增加到 256
        self.fc2 = nn.Linear(128, 1)  # 对于回归任务，输出维度为 1，表示预测的 SSC

        # 定义 Dropout
        self.dropout = nn.Dropout(0.3)

        # 激活函数（ReLU）
        self.relu = nn.ReLU()

    def forward(self, x, return_data = "pred"):
        # 确保输入数据是 float32
        x = x.float()
        x = x.unsqueeze(1)  # 增加通道维度 (batch_size, 1, feature_length)

        # 卷积层和激活函数
        x = self.relu(self.conv1(x))  # 使用 ReLU 激活
        x = self.dropout(x)
        x = self.relu(self.conv2(x))  # 使用 ReLU 激活
        x = self.dropout(x)
        x = self.relu(self.conv3(x))  # 使用 ReLU 激活

        # 展平层
        x = x.view(x.size(0), -1)  # 将卷积层的输出展平

        # 全连接层
        x_rep = self.relu(self.fc1(x))  # 使用 ReLU 激活
        output = self.fc2(x_rep)  # 输出层，不应用 softmax

        if return_data == "pred":
            return output.squeeze()  # 去掉最后的维度，输出为 (batch_size,)
        else:
            return x_rep, output.squeeze()
import torch
import torch.nn as nn
from backpack import extend

class OneDCNN(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(OneDCNN, self).__init__()

        # 整个特征提取流程打包进 _main
        self._main = nn.Sequential(
            nn.Conv1d(1, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2 * input_dim, hidden_size),
            nn.ReLU()
        )

        # 分类器用于预测（需要 extend 用于 BackPACK）
        self.classifier = extend(nn.Linear(hidden_size, 1))

        # 投影头（用于 InfoNCE）
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size)
        )

    def forward(self, x, return_data="pred"):
        x = x.float().unsqueeze(1)  # shape: [B, 1, L]
        features = self._main(x)
        logits = self.classifier(features)

        if return_data == "pred":
            return logits.squeeze()
        elif return_data == "feat":
            return features, logits.squeeze()
        else:
            raise ValueError("return_data must be 'pred' or 'feat'")
