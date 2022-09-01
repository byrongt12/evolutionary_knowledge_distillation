import torch
from torch import nn
import torch.nn.functional as F

from NeuralNetwork.Helper import rescale


def accuracy(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predictions == actual).item() / len(predictions))

def accuracy_vanilla_kd(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    _, actual_grouped = torch.max(actual, dim=1)
    return torch.tensor(torch.sum(predictions == actual_grouped).item() / len(predictions))

class BaseModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return loss, acc

    def training_step_vanilla_kd(self, batch, teacher_model):

        images, labels = batch

        teacher_out = teacher_model(images)
        teacher_out = rescale(teacher_out)

        out = self(images)
        out = rescale(out)

        loss = F.cross_entropy(out, teacher_out)
        acc = accuracy_vanilla_kd(out, teacher_out)
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [loss["val_loss"] for loss in outputs]
        loss = torch.stack(batch_losses).mean()
        batch_accuracy = [accuracy["val_acc"] for accuracy in outputs]
        acc = torch.stack(batch_accuracy).mean()
        return {"val_loss": loss.item(), "val_acc": acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, train_acc: {:.4f} , val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))