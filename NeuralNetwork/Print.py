import pylab as pl
import torch
from torch import nn

from torchvision.utils import make_grid
from torchviz import make_dot
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork.Helper import getModelWeights


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def printModel(model, t_loader):
    batch = next(iter(t_loader))
    yhat = model(batch[0].cuda())  # Give dummy batch to forward()
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("print/rnn_torchviz_student")
    print("Model printed.")


def printModelSummary(nn_model, firstConvWeight=False, allWeightsShape=False, summaryDisplay=True):
    if firstConvWeight or allWeightsShape:
        model_weights, conv_layers = getModelWeights(nn_model)
        print(f"Total convolution layers: {len(conv_layers)}")
        print("conv_layers")
        print(len(conv_layers))
        if firstConvWeight:
            print(model_weights[0])
        if allWeightsShape:
            for i in range(len(model_weights)):
                print(model_weights[i].shape)

    if summaryDisplay:
        summary(nn_model, (3, 32, 32))


def printFeatureMaps(model, device, train_loader):
    dataIter = iter(train_loader)
    imgs, labels = next(dataIter)

    image = imgs[0]

    # print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    # print(f"Image shape after: {image.shape}")
    image = image.to(device)

    outputs = []
    names = []
    model_weights, conv_layers = getModelWeights(model)

    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    # print(len(outputs))
    # print feature_maps
    # for feature_map in outputs:
    #    print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    # for fm in processed:
    #    print(fm.shape)

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(10, 11, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
    print("Feature maps printed.")


def show_batch(dl):
    for batch in dl:
        images, labels = batch
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20], nrow=5).permute(1, 2, 0))
        pl.show()
        break


def plot_acc(history):
    plt.plot([x.get("train_acc") for x in history], "-bx")
    plt.plot([x["val_acc"] for x in history], "-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train acc", "val acc"])
    plt.show()


def plot_loss(history):
    plt.plot([x.get("train_loss") for x in history], "-bx")
    plt.plot([x["val_loss"] for x in history], "-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss", "val loss"])
    plt.show()


def plot_lrs(history):
    plt.plot(np.concatenate([x.get("lrs", []) for x in history]))
    plt.xlabel("Batch number")
    plt.ylabel("Learning rate")
    plt.show()

def plot_ga_best(best_arr):
    plt.title("Best fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Best validation accuracy")
    generations_arr = np.array(range(1, len(best_arr) + 1))
    plt.plot(generations_arr, best_arr, color="red", marker="o", label="Best val accuracy")
    plt.legend()
    plt.savefig('best_over_generations.png')


