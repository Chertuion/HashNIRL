import torch
import torch.nn.functional as F


# def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.mse_loss):
#     device = causal_pred.device
#     dummy_w = torch.tensor(1.).to(device).requires_grad_()
#     loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
#     loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
#     grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
#     grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
#     irm_loss = torch.sum(grad_0 * grad_1)
#     # print(irm_loss)
#     return irm_loss


def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.mse_loss, lambda_irm=1.0):
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    

    loss_0 = torch.tensor(0., device=device)
    loss_1 = torch.tensor(0., device=device)
    grad_0 = torch.tensor(0., device=device)
    grad_1 = torch.tensor(0., device=device)
    

    if torch.sum(batch_env_idx == 0) > 0:
        loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
        grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    

    if torch.sum(batch_env_idx == 1) > 0:
        loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
        grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    

    grad_0_norm = grad_0 / (torch.norm(grad_0) + 1e-8)  # 防止除零
    grad_1_norm = grad_1 / (torch.norm(grad_1) + 1e-8)
    irm_penalty = (grad_0_norm * grad_1_norm) ** 2
    

    total_loss = loss_0 + loss_1 + lambda_irm * irm_penalty
    return total_loss