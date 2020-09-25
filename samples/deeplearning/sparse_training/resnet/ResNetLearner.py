import torch
import torchvision.models as models

from fastai2.data.external import *
from fastai2.data.transforms import *
from fastai2.data.block import *
from fastai2.vision.models.xresnet import *
from fastai2.vision.core import *
from fastai2.vision.data import *
from fastai2.vision.augment import *

from torch.utils.tensorboard import SummaryWriter
from utils import prune_scheduler

"""Helper function that loads ImageNette dataset for us
    size (int): size of the input image dimensions
"""
def get_dls(size, woof, bs, sh=0., workers=None):
    if size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else        : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    source = untar_data(path)
    if workers is None: workers = 8
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    if sh: batch_tfms.append(RandomErasing(p=0.3, max_count=3, sh=sh))
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files, get_y=parent_label,
                       item_tfms=[RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)],
                       batch_tfms=batch_tfms)
    return dblock.dataloaders(source, path=source, bs=bs, num_workers=workers)


class ResNetLearner():
    def __init__(self, n_classes=10, num_epochs=50, target_sparsity=0.8):
        self.model = models.resnet50()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter()
        self.num_epochs = num_epochs

        # Modify resnet to have 10 layers
        self.model.fc.out_features = n_classes
        self.cls_crit = torch.nn.CrossEntropyLoss()

        # Send stuff to device
        self.model.to(self.device)
        self.cls_crit.to(self.device)

        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=0.005,
            momentum=0.9)

        self.dls = get_dls(128, 0, 64, sh=0., workers=8)

        # Array of sparsity rates
        self.sparsity_scheduler = prune_scheduler(1-target_sparsity, num_epochs, start_epoch=1)


    def prune_weights(self, epoch):
        # Send model weights to cpu
        self.model.to("cpu")
        sparsity_rate = self.sparsity_scheduler[epoch]

        self.modules_to_prune = []
        for module in self.model.modules():
            if type(module) == torch.nn.modules.conv.Conv2d:
                self.modules_to_prune.append((module, 'weight'))
        prune.global_unstructured(
            self.modules_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity_rate)

        report_sparsity(self.model, self.writer, epoch)

        # Send model weights back to designated device
        self.model.to(self.device)

    def train(self, epoch):
        for epoch in range(self.num_epochs):
            self.model.train()

            tr_loss = 0.

            for i, batch in enumerate(self.dls.train):
                self.optim.zero_grad()
                x, y = batch[:1], batch[1:]
                pred = self.model(*x)
                loss = self.cls_crit(pred, y[0])
                loss.backward()
                self.optim.step()
                tr_loss += loss

            print("Epoch: {} / {}, Training loss: {}.".format(
                epoch, self.num_epochs, loss / len(self.dls.train)))

    def eval(self):
        self.model.eval()
        y_correct = 0
        num_eval = 0

        te_loss = 0.

        for i, batch in enumerate(self.dls.valid):
            # x, y are tuples of size 1
            x, y = batch[:1], batch[1:]
            logit = self.model(*x)
            pred = torch.argmax(logit, dim=1)
            #loss = cls_crit(logit, y[0])

            #te_loss += loss

            y_correct += torch.sum(pred == y[0]).item()
            num_eval += len(*y)

        print("Accuracy: {}, {} / {}".format(acc, y_correct, num_eval))

