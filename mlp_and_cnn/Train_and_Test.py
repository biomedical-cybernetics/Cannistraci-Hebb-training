from tqdm import tqdm
import torch
import torch.nn as nn
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def Train(args, model, device, train_loader, optimizer, epoch, warmup_scheduler):
    model.train()
    CSE = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CSE(output, target)
        loss.backward()
        if "gradient" in args.regrow_method:
            for layer in model.sparse_layers:
                layer.core_grad = layer.weight_core.grad
        optimizer.step()
        if args.warmup and epoch < 1:
            warmup_scheduler.step()
        # print(torch.mean(model.sparse_layers[2].weight.data))
        

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.4f}'.format(
        epoch, batch_idx, len(train_loader.dataset)//args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


def Test(model, device, val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    print('Test:\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Prec@1 {top1.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}'.format(batch_time=batch_time, loss=losses,
            top1=top1, top5=top5))

    top1 = top1.avg.cpu().numpy()
    top5 = top5.avg.cpu().numpy()
    loss = losses.avg
    return top1, top5, loss


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res