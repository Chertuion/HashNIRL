import torch.nn as nn
import torch
from model.gnn import GNN
from model.base_models import LSTM, OneDCNN
from torch_geometric.data import Batch
import numpy as np
from utils.generate_pyg import get_pyg_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class HashEncoder(nn.Module):
    def __init__(self, input_dim, hash_dim=32, hidden_dim=128, rate = 0.5):
        super(HashEncoder, self).__init__()
        k = int(hash_dim * rate)
        self.k = k
        self.hash_dim = hash_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hash_dim),
            nn.Tanh()
        )

    def topk_binary(self, z, k):
        B, D = z.shape
        out = torch.zeros_like(z)
        topk = torch.topk(z, k=k, dim=1).indices
        out.scatter_(1, topk, 1.0)
        return out

    def forward(self, x):
        z = self.encoder(x)
        z_soft = (z + 1) / 2
        z_hard = self.topk_binary(z, self.k)
        hash_code = z_hard + (z_soft - z_soft.detach())
        return hash_code, z_soft

class Ours(nn.Module):
    def __init__(self,
                input_dim,
                out_dim,
                spec_dim,
                args,
                hash_bit = 128,
                edge_dim=-1,
                emb_dim=128,
                num_layers=2,
                gnn_type='gin',
                virtual_node=True,
                residual=True,
                drop_ratio=0.3,
                JK="last",
                graph_pooling="mean"):
        super(Ours, self).__init__()
        self.classifier = GNN(gnn_type=gnn_type,
                            input_dim=input_dim,
                            num_class=out_dim,
                            num_layer=num_layers,
                            emb_dim=emb_dim,
                            drop_ratio=drop_ratio,
                            virtual_node=virtual_node,
                            graph_pooling=graph_pooling,
                            residual=residual,
                            JK=JK,
                            edge_dim=edge_dim)
        self.hash_encoder = HashEncoder(input_dim=emb_dim, hash_dim=hash_bit, hidden_dim=emb_dim, rate = args.rate)
        self.OneDcnn = OneDCNN(input_dim=spec_dim, hidden_size=emb_dim)
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)
        self.dropout = nn.Dropout(args.drop_rate)
        self.relu = nn.ReLU()
        self.shuffle_projection = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, batch, args, return_data="pred"):
        graph_pred, graph_rep = self.classifier(batch, get_rep=True)
        if not isinstance(batch.spec, torch.Tensor):
            batch.spec = torch.tensor(np.array(batch.spec), dtype=torch.float32).to(batch.edge_index.device)
        if not isinstance(batch.spec, torch.Tensor):
            batch.spec = torch.tensor(np.array(batch.y), dtype=torch.float32).to(batch.edge_index.device)
        cnn_pred, cnn_rep = self.OneDcnn(batch.spec, return_data=="rep")
        invMode_code, soft_code = self.hash_encoder(graph_rep)
        invMode_loss = self.bit_balance_loss(hash_code = invMode_code, target_ratio=args.batch_rate)
        inv_fusion_feature = invMode_code * cnn_rep
        spuMode_code = 1 - invMode_code
        spu_fusion_feature = spuMode_code * cnn_rep
        perm = torch.randperm(spu_fusion_feature.size(0))
        shuffled_spu_fusion_feature = spu_fusion_feature[perm]
        env_fusion_feature = torch.cat([inv_fusion_feature, shuffled_spu_fusion_feature], dim=1)
        env_fusion_feature_projection = self.shuffle_projection(env_fusion_feature)
        _, inv_s1 = self.hash_encoder(inv_fusion_feature)
        _, env_s2 = self.hash_encoder(env_fusion_feature_projection)
        dis_loss = self.distill_loss_l2(inv_s1, soft_code)
        ctst_loss = self.contrastive_soft_neg_loss(inv_s1, soft_code, env_s2)
        x = self.fc1(env_fusion_feature_projection)
        x = self.dropout(x)
        output = self.fc2(x)
        if return_data == "pred":
            return output.squeeze()
        else:
            return graph_pred.squeeze(), output.squeeze(), cnn_pred.squeeze(), invMode_loss,  dis_loss + ctst_loss

    def bit_balance_loss(self, hash_code, target_ratio=0.25):
        bit_mean = hash_code.mean(dim=0)
        return torch.mean((bit_mean - target_ratio) ** 2)

    def similarity_loss(self, p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()
    
    def contrastive_soft_neg_loss(self, inv_s1, soft_code, env_s2, temperature=0.07):
        inv_s1 = F.normalize(inv_s1, dim=1)
        soft_code = F.normalize(soft_code, dim=1)
        env_s2 = F.normalize(env_s2, dim=1)
        pos_sim = (inv_s1 * soft_code).sum(dim=1) / temperature
        neg_sim = (inv_s1 * env_s2).sum(dim=1) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(inv_s1.size(0), dtype=torch.long, device=inv_s1.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    def distill_loss_l2(self, student_feat, soft_code):
        return F.mse_loss(student_feat, soft_code)





