#%%
import torch
from torch import nn
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, $-\alpha (1-\hat{y})^{\gamma} * CrossEntropyLoss(\hat{y}, y)$
        alpha: 类别权重. 当α是列表时, 为各类别权重, 当α为常数时, 类别权重为[α, 1-α, 1-α, ....]
        gamma: 难易样本调节参数.
        num_classes: 类别数量
        size_average: 损失计算方式, 默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入, 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为[α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds: 预测类别. size:[B, C] or [B, S, C] B 批次, S长度, C类别数
        labels: 实际类别. size:[B] or [B, S] B批次, S长度
        """
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        print(labels)
        print(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels)   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.76, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
                - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

#%%
if __name__ == "__main__":
    
    loss = AsymmetricLossOptimized()
    pred = torch.rand(4,1)
    labels = torch.tensor([[0],
                            [1],
                            [1],
                            [0]])
    losses = loss(pred, labels)
    print(losses)
# %%
