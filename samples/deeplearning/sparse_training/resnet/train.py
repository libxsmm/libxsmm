import sys
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision.models as models

from dataset import data_loader
from PrunedModel import UnstructuredPrunedModel

from fastai2.data.external import *
from fastai2.data.transforms import *
from fastai2.data.block import *
from fastai2.vision.models.xresnet import *
from fastai2.vision.core import *
from fastai2.vision.data import *
from fastai2.vision.augment import *

from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter
from utils import prune_scheduler

NUM_EPOCHS = 50

from ResNetLearner import ResNetLearner


def report_sparsity(model, writer, it, verbose=False):
    pruned_modules = []
    for name, module in model.named_modules():
        if type(module) == torch.nn.modules.conv.Conv2d:
            nz = float(torch.sum(module.weight == 0))
            nelement = float(module.weight.nelement())
            pruned_modules.append((name, nz, nelement))

    t_nz = 0
    t_nelement = 0

    # Report
    for name, nz, nelement in pruned_modules:
        t_nz += nz
        t_nelement += nelement
        if verbose:
            print("Sparsity in {}: {:2f}%".format(name, 100. * nz / nelement))

    print("Global sparsity: {:2f}%".format(100. * t_nz / t_nelement))
    writer.add_scalar("Global sparsity", t_nz / t_nelement, it)
    return 0.

def prune_model(model, amount=0.159):
    # modules to prune
    modules_to_prune = []
    for module in model.modules():
        if type(module) == torch.nn.modules.conv.Conv2d:
            modules_to_prune.append((module, 'weight'))
    #prune.global_unstructured(modules_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    prune.global_unstructured(modules_to_prune, pruning_method=prune.RandomUnstructured, amount=amount)

def train():
    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model
    model = models.resnet50()

    # Modify the fc layer to accomodate n_labels
    model.fc.out_features = 10

    cls_crit = torch.nn.CrossEntropyLoss()

    # Send to device
    model.to(device)
    cls_crit.to(device)

    # optimizer and learning rate scheduler
    optim = torch.optim.SGD(
            model.parameters(),
            lr=0.005,
            momentum=0.9)
    #TODO lr scheduler

    # Load datasets using fastai2
    dls = get_dls(128, 0, 64, sh=0., workers=8)

    spartsity_scheduler = prune_scheduler(0.2, NUM_EPOCHS, start_epoch=1)

    print(spartsity_scheduler)

    for epoch in range(NUM_EPOCHS):
        model.train()
        # start training
        n_iter = len(dls.train)

        tr_loss = 0.

        for i, batch in enumerate(dls.train):
            optim.zero_grad()
            # x, y are tuples of size 1
            x, y = batch[:1], batch[1:]
            pred = model(*x)
            loss = cls_crit(pred, y[0])
            loss.backward()
            optim.step()
            tr_loss += loss

        print("Epoch: {} / {}, Training loss: {}.".format(epoch, NUM_EPOCHS, loss / len(dls.train)))

        model.eval()

        y_correct = 0
        num_eval = 0

        te_loss = 0.

        for i, batch in enumerate(dls.valid):
            # x, y are tuples of size 1
            x, y = batch[:1], batch[1:]
            logit = model(*x)
            pred = torch.argmax(logit, dim=1)
            #loss = cls_crit(logit, y[0])

            #te_loss += loss

            y_correct += torch.sum(pred == y[0]).item()
            num_eval += len(*y)

        writer.add_scalar("Loss/train", tr_loss / len(dls.train), epoch)
        #writer.add_scalar("Loss/test", te_loss / len(dls.valid), epoch)

        acc = y_correct / num_eval
        writer.add_scalar("Accuracy/test", acc, epoch)

        print("Accuracy: {}, {} / {}".format(acc, y_correct, num_eval))

        # Prune model on CPU
        if spartsity_scheduler[epoch] == 1.:
            print("Skipping pruning...")

        else:
            model.to('cpu')
            prune_model(model, spartsity_scheduler[epoch])
            report_sparsity(model, writer, epoch)
            model.to(device)

if __name__ == "__main__":
    target_sparsity = float(sys.argv[1])
    learner = ResNetLearner(target_sparsity=target_sparsity)

    for i in range(NUM_EPOCHS):
        learner.train(i)
        learning.eval()
