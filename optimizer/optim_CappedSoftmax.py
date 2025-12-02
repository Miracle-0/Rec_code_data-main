from optimizer.optim_Base import IROptimizer
from torch import nn
import torch

class CappedSoftmaxOptimizer(IROptimizer):
    def __init__(self, model, config):
        """
        初始化 Softmax 优化器。
        # lr: 学习率
        # weight_decay: 权重衰减，用于正则化
        # temp: 温度参数，用于调整 softmax 分布的平滑度
        """
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]
        self.temp           =  config['ssm_temp']
        self.eta = config.get('eta', 0.1)  # 建议初始值设为 0.1 或 0.2 试试

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr)

    def cal_loss(self, y_pred):
        # 取出正负样本分数
        # y_pred shape: [batch_size, 1 + n_negs]
        pos_scores = y_pred[:, 0]      # [batch_size]
        neg_scores = y_pred[:, 1:]     # [batch_size, n_negs]

        # === 核心修改开始 ===
        
        # 1. 计算负样本的 Softmax 概率分布 (即 w_j)
        # 这里的 softmax 是在 dim=1 (负样本维度) 上进行的
        # 注意：要除以温度 temp
        neg_probs = torch.softmax(neg_scores / self.temp, dim=-1)
        
        # 2. 关键步骤：截断权重 (Weight Clipping)
        # clamp(max=self.eta) 把大于 eta 的权重压死在 eta
        # .detach() 非常重要！这告诉 PyTorch：计算梯度时，把这个 w 当作常数，不要对它求导
        neg_weights = torch.clamp(neg_probs, max=self.eta).detach()
        
        # 3. 构造 Surrogate Loss (代理损失)
        # 这是一个数学技巧。
        # 我们想要：对 neg_scores 的梯度 = (1/temp) * neg_weights
        # 所以我们需要构造一个线性 Loss： Sum(weight * score) / temp
        
        # 正样本部分 Loss (梯度是 -1/temp)
        pos_term = - (pos_scores / self.temp)
        
        # 负样本部分 Loss (梯度是 neg_weights/temp)
        neg_term = torch.sum(neg_weights * (neg_scores / self.temp), dim=-1)
        
        # 最终 Loss
        loss = (pos_term + neg_term).mean()
        
        # === 核心修改结束 ===

        return loss

    # 正则化
    def regularize(self,users_emb, pos_emb, neg_emb):
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        return regularize

    def cal_loss_graph(self, users, pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss            =  self.cal_loss(y_pred)
        emb_loss        =  self.weight_decay * self.regularize(users_emb, pos_emb, neg_emb) / batch_size
        additional_loss =  self.model.additional_loss(
                                usr_idx = users.long(), 
                                pos_idx = pos.long(), 
                                embedding_user = embedding_user, 
                                embedding_item = embedding_item
                            )
        return loss, emb_loss + additional_loss

    def step(self, user, pos, neg):
        ssm_loss,emb_loss = self.cal_loss_graph(user, pos, neg)
        loss = ssm_loss + emb_loss
        self.optimizer_descent.zero_grad() # 清空之前的梯度

        loss.backward() # 反向传播求导

        self.optimizer_descent.step() # 更新参数
        return ssm_loss.cpu().item()
    
    def save(self,path):
        all_states = self.model.state_dict()
        torch.save(obj = all_states, f = path)

