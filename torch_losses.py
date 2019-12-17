# coding=utf-8
# Author: Didia
# Date: 19-12-17
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy


class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets):
        if not isinstance(targets, torch.FloatTensor):
            targets = targets.float()
        outputs = outputs.view(-1)
        pred = self.sigmoid(outputs)

        targets = targets.view(-1)

        smooth = 1.
        intersection = (pred * targets).sum()

        return 1 - ((2. * intersection + smooth) /
                    (pred.sum() + targets.sum() + smooth))


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        # NLLLoss2d直接计算pixel-wise的loss,
        # 输入的维度分别为： output = [batch_size, num_class, shape, shape]
        #                   target = [batch_size, shape, shape]
        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)


class LossWithWeight(nn.Module):
    def __init__(self, weight=None):
        super(LossWithWeight, self).__init__()
        self.loss = nn.NLLLoss2d(weight)
        self.classify_loss = nn.BCELoss() # BE

    def forward(self, outputs, targets):
        # targets = targets.type(torch.DoubleTensor)
        # outputs, predicted = torch.max(outputs.data, 1)  # 此处参数选为1， 对第一维（num_class）操作
        outputs = F.sigmoid(outputs)
        pos_idcs = targets>0
        pos_output = outputs[:,1][pos_idcs]
        pos_target = targets[pos_idcs].type(torch.FloatTensor).cuda()
        neg_idcs = targets<1
        neg_output = outputs[:,1][neg_idcs]
        neg_target = targets[neg_idcs].type(torch.FloatTensor).cuda()
        pos_loss = self.classify_loss(pos_output, pos_target)
        neg_loss = self.classify_loss(neg_output, neg_target)
        # print pos_loss, neg_loss
        return pos_loss+neg_loss


def hard_mining(neg_output, neg_labels, num_hard):
    if num_hard==0:
        num_hard = 100
    l = len(neg_output)
    neg_output, idcs = torch.topk(neg_output, l)
    idcs = idcs[200:]
    interval = num_hard/5
    l_devide = l/5
    selected_idcs = idcs[:interval]
    for i in range(4):
        i+=1
        selected_idcs = torch.cat((selected_idcs, idcs[i*l_devide: i*l_devide+interval]))
    neg_output = torch.index_select(neg_output, 0, selected_idcs)
    neg_labels = torch.index_select(neg_labels, 0, selected_idcs)

    return neg_output, neg_labels


class LossWithHardmining(nn.Module):
    def __init__(self, weight=None):
        super(LossWithHardmining, self).__init__()
        self.bce = nn.BCELoss(weight)

    def forward(self, outputs, targets):
        # outputs = F.sigmoid(outputs[:,0])
        # outputs = F.sigmoid(outputs)
        # print outputs.size()
        pos_idcs = targets>0
        neg_idcs = targets<1
        pos_output = outputs[pos_idcs]
        pos_target = targets[pos_idcs].type(torch.FloatTensor).cuda()
        neg_output = outputs[neg_idcs]
        neg_target = targets[neg_idcs].type(torch.FloatTensor).cuda()

        neg_output, neg_target = hard_mining(neg_output, neg_target, num_hard=4*len(pos_target))

        pos_loss = self.bce(pos_output, pos_target)
        if len(pos_target)==0:
            pos_loss=0
        neg_loss = self.bce(neg_output, neg_target)
        return pos_loss+neg_loss


class LossBinary():
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        if weight is not None:
            weight = torch.from_numpy(np.asarray(weight))
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        input = input.permute(0,4,1,2,3).contiguous()
        # compute the negative likelyhood
        if self.weight is not None:
            weight = Variable(self.weight)
            logpt = -F.cross_entropy(input, target, weight, ignore_index=255)
        else:
            logpt = -F.cross_entropy(input, target, ignore_index=255)

        pt = torch.exp(logpt)
        loss = -((1-pt)**self.gamma) * logpt
        # averaging (or not) loss
        print(loss.mean())
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)
    input = input.type(torch.LongTensor)

    # expand the input tensor to Nx1xDxHxW
    src = input.unsqueeze(1)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)


def one_hot(labels, classes):
    labels = labels.contiguous()
    b, s, k = labels.size()
    labels = labels.view(b, s, k, -1)
    newsize = (b, s, k, classes)
    labels_one_hot = torch.FloatTensor(*newsize).zero_()
    if labels.is_cuda:
        labels_one_hot.scatter_(-1, labels.cpu().data, 1.0)
        target = Variable(labels_one_hot.float().cuda())
    else:
        labels_one_hot.scatter_(-1, labels, 1)
        target = Variable(labels_one_hot.float())
    return target.contiguous()


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target, weights=None):

        input = input.permute(0,4,1,2,3)
        log_probabilities = self.log_softmax(input).cpu()
        # loss1 = F.nll_loss(log_probabilities, target, ignore_index=255)  # === F.cross_entropy(input, target)

        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        target = target.type(torch.DoubleTensor)
        loss1 = self.bce(input,  target)
        print(loss1)

        if weights is not None:
            assert target.size() == weights.size()
            weights = weights.unsqueeze(1)
            weights = weights.type(torch.DoubleTensor)
            result = -weights * target * log_probabilities
        else:
            result = -target*log_probabilities

        print(result.mean())
        return result.mean()