import torch
import torchvision.models as models
import time

class UnstructuredPrunedModel():
    def __init__(self, base_model="resnet50"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Model initiated, running on {}".format(self.device))

        self.net = getattr(models, base_model)()
        self.cls_crit = torch.nn.CrossEntropyLoss()

        # Send to device
        self.net.to(self.device)
        self.cls_crit.to(self.device)

        # optimizer and learning rate scheduler
        self.optim = torch.optim.SGD(
            self.net.parameters(),
            lr=0.001,
            momentum=0.9)
        #TODO lr scheduler


    def update(self, tr_data_loader, epoch, logger):
        # switch net to train mode
        self.net.train()

        # training
        train_loss = 0.
        num_eval = 0
        start_time = time.time()

        for x, y in tr_data_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logit_y = self.net(x)

            # gradient of loss and update weights
            self.optim.zero_grad()

            # compute loss and gradient
            loss = self.cls_crit(logit_y, y)
            loss.backward()

            # update weights
            self.optim.step()
            num_eval += len(x)

            #log loss
            train_loss += loss

            #TODO Print every so often
        print("Epoch #{} Complete".format(epoch))
        print("Elapsed time: %2.3f, Train_loss: %2.5f" % (time.time() - start_time, train_loss / num_eval))

        return train_loss

    def eval(self, val_data, logger):
        print("Evaluating model on test data")
        self.net.eval()
        y_correct = 0

        num_eval = 0
        for x, y in val_data:
            x = x.to(self.device)
            y = y.to(self.device)

            logit_y = self.net(x)
            prediction = torch.argmax(logit_y, dim=1)

            y_correct += torch.sum(prediction == y).item()
            num_eval += len(y)

        acc = y_correct / num_eval * 100

        print("Accuracy: {}, {} / {}".format(acc, y_correct, num_eval))


    def prune_weights(self):
        pass

    def org_model(self):
        pass

    def summary(self):
        pass
