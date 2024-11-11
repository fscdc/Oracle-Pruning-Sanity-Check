import numpy as np
import torch
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # shape [maxk, batch_size]
        correct = pred.eq(
            target.view(1, -1).expand_as(pred)
        )  # target shape: [batch_size] -> [1, batch_size] -> [maxk, batch_size]
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(test_loader, model, criterion, device, accelerator):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    train_state = model.training
    with torch.no_grad():
        model.eval()
        for i, (images, targets) in enumerate(tqdm(test_loader, desc="Testing Progress")):
            images = images.to(device=device)
            targets = targets.to(device=device).long()  # label should be long int
            outputs = model(images)

            logits = outputs.logits

            all_logits, all_targets = accelerator.gather_for_metrics((logits, targets))

            loss = criterion(all_logits, all_targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(all_logits, all_targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    if train_state:
        model.train()
    return top1.avg, top5.avg, losses.avg


def validate_noaccelerator(test_loader, model, criterion, device):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    train_state = model.training
    with torch.no_grad():
        model.eval()
        for i, (images, targets) in enumerate(tqdm(test_loader, desc="Testing Progress")):
            images = images.to(device=device)
            targets = targets.to(device=device).long()  # label should be long int
            outputs = model(images)

            logits = outputs.logits

            loss = criterion(logits, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    if train_state:
        model.train()
    return top1.avg, top5.avg, losses.avg


def validate_tv(test_loader, model, criterion, device):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    train_state = model.training
    with torch.no_grad():
        model.eval()
        for i, (images, target) in enumerate(tqdm(test_loader, desc="Testing Progress")):
            images = images.to(device=device)
            target = target.to(device=device).long()
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    if train_state:
        model.train()
    return top1.avg, top5.avg, losses.avg


def validate_tv_accelerator(test_loader, model, criterion, device, accelerator):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    train_state = model.training
    with torch.no_grad():
        model.eval()
        for i, (images, target) in enumerate(tqdm(test_loader, desc="Testing Progress")):
            images = images.to(device=device)
            target = target.to(device=device).long()
            output = model(images)

            all_logits, all_targets = accelerator.gather_for_metrics((output, target))

            loss = criterion(all_logits, all_targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(all_logits, all_targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    if train_state:
        model.train()
    return top1.avg, top5.avg, losses.avg