import random

import torch  # need this for eval function
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import psnr_loss, lovasz_softmax_loss, ssim_loss
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchmetrics.functional import pairwise_euclidean_distance
from torch.nn.functional import normalize

# from pytorch_msssim import ms_ssim
# from ignite.engine import Engine
# from ignite.metrics import SSIM, PSNR
import kornia.metrics as metrics


def printLayerAndGradientBoolean(student_model):
    model_children = list(student_model.children())
    counter = 0

    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            for parameter in model_children[i].parameters():
                print("Conv layer number: " + str(counter) + ". Requires gradient: " + str(parameter.requires_grad))

        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        for parameter in child.parameters():
                            print("Conv layer number: " + str(counter) + ". Requires gradient: " + str(
                                parameter.requires_grad))


def printLayerAndGradient(student_model):
    model_children = list(student_model.children())
    counter = 0

    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            for parameter in model_children[i].parameters():
                print("Conv layer number: " + str(counter) + ". Gradient: " + str(parameter.grad))

        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        for parameter in child.parameters():
                            print("Conv layer number: " + str(counter) + ". Gradient: " + str(
                                parameter.grad))


def changeGradientBoolean(featureMapNumForStudent, student_model):
    model_children = list(student_model.children())
    counter = 0

    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
        if counter > featureMapNumForStudent:
            for parameter in model_children[i].parameters():
                parameter.requires_grad = False

        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                    if counter > featureMapNumForStudent:
                        for parameter in child.parameters():
                            parameter.requires_grad = False


def resetGradientBoolean(student_model):
    for child in student_model.children():
        for parameter in child.parameters():
            parameter.requires_grad = True


def getModelWeights(model):
    # save the convolutional layer weights
    m_weights = []
    # save the convolutional layers
    c_layers = []
    # get all the model children as list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0

    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            m_weights.append(model_children[i].weight)
            c_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        m_weights.append(child.weight)
                        c_layers.append(child)

    return m_weights, c_layers


def getNumberOfConvolutionLayers(nn_model):
    model_weights, conv_layers = getModelWeights(nn_model)
    return len(conv_layers)


def getFeatureMaps(model, device, image):
    # dataIter = iter(train_loader)
    # imgs, labels = next(dataIter)

    # image = imgs[0]

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
    #     print(feature_map.shape)

    return outputs


def printSingularFeatureMap(featureMap):
    feature_map = featureMap.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]

    plt.imshow(gray_scale.data.cpu().numpy())
    plt.show()


def convertLayerToCode(student_model_number, featureMapNumForStudent, layerOnly=False):
    layer = None
    block = None
    conv = None

    divisor = student_model_number * 2
    quotient = featureMapNumForStudent // divisor
    if featureMapNumForStudent % divisor == 0:
        layer = quotient
    else:
        layer = quotient + 1

    if layerOnly:
        return layer

    if featureMapNumForStudent % divisor == 0:
        block = student_model_number
        conv = 2
    else:
        if (featureMapNumForStudent % divisor) % 2 == 0:
            block = ((featureMapNumForStudent % divisor) // 2)
        else:
            block = ((featureMapNumForStudent % divisor) // 2) + 1

    if conv is None:
        if ((featureMapNumForStudent % divisor) % 2) == 0:
            conv = 2
        else:
            conv = 1

    return layer, block, conv


def differentSizeMaps(featureMapForTeacher, featureMapForStudent):
    # If matrices have different shapes: downsize to small one + shave off values make matrix size identical.
    A = featureMapForTeacher  # .detach().clone()
    B = featureMapForStudent  # .detach().clone()

    if featureMapForTeacher.size() != featureMapForStudent.size():

        if A.size() < B.size():  # if the total Student tensor is bigger but inner tensors smaller
            A = transforms.functional.resize(A, B.size()[3])
            B = B.narrow(1, 0, A.size()[1])

        elif A.size() > B.size():  # if the total Teacher tensor is bigger but inner tensors smaller
            B = transforms.functional.resize(B, A.size()[3])
            A = A.narrow(1, 0, B.size()[1])

    return A, B


def creatParametersList(student_model, layerForStudent, blockForStudent, convForStudent):
    params = []
    # Add all the layers from the start until current layer
    for i in range(1, layerForStudent):
        executeStr = 'list(student_model.layer' + str(i) + '.parameters())'
        params += eval(executeStr)

    for j in range(1, blockForStudent):
        for x in range(1, 3):
            executeStr = 'list(student_model.layer' + str(layerForStudent) + '[' + str(j - 1) + '].conv' + str(
                x) + '.parameters())'
            params += eval(executeStr)

    for k in range(1, convForStudent):
        executeStr = 'list(student_model.layer' + str(layerForStudent) + '[' + str(
            blockForStudent - 1) + '].conv' + str(k) + '.parameters())'
        params += eval(executeStr)

    # Add current layer parameters
    executeStr = 'list(student_model.layer' + str(layerForStudent) + '[' + str(blockForStudent - 1) + '].conv' + str(
        convForStudent) + '.parameters())'
    params += eval(executeStr)

    return params


def distill(heuristicString, heuristicToLayerDict, kd_loss_type, distill_optimizer, distill_lr, batch,
            student_model,
            student_model_number, teacher_model, teacher_model_number, device):
    # Step through GA string.
    # abcdefghijklmnopqr = represents student feature map.
    # Use random to get corresponding teacher block(1-18) and conv (1-2).
    # Compute loss from mappings between teacher and student feature map.
    # Add add loss values from all feature map comparisons and then calculate gradients and update weights.

    student_model.train()  # put the model in train mode

    kd_loss_arr = []
    featureMapNumForStudentArr = []
    featureMapNumForTeacherArr = []
    distill_optimizer_implemented = distill_optimizer(student_model.parameters(), lr=distill_lr)

    for i in range(0, len(heuristicString)):
        if (i + 1) % 2 != 0:
            # student layer
            student_layer_number = heuristicToLayerDict[heuristicString[i]] % student_model_number
            if student_layer_number == 0:
                student_layer_number = student_model_number
            featureMapNumForStudentArr.append(student_layer_number)
        else:
            # teacher layer
            featureMapNumForTeacherArr.append(heuristicToLayerDict[heuristicString[i]])

    if len(heuristicString) % 2 != 0:
        # featureMapNumForStudentArr > featureMapNumForTeacherArr, so add random teacher layer
        layerForStudent, blockForStudent, convForStudent = convertLayerToCode(student_model_number,
                                                                              featureMapNumForStudentArr[-1])
        random.seed(len(featureMapNumForStudentArr))
        layerForTeacher = layerForStudent
        blockForTeacher = random.randint(1, teacher_model_number)
        convForTeacher = random.randint(1, 2)

        featureMapNumForTeacher = ((layerForTeacher - 1) * (teacher_model_number * 2)) + (
                (blockForTeacher - 1) * 2) + convForTeacher

        featureMapNumForTeacherArr.append(featureMapNumForTeacher)

    for i in range(0, (len(featureMapNumForStudentArr))):
        featureMapNumForStudent = featureMapNumForStudentArr[i]
        featureMapNumForTeacher = featureMapNumForTeacherArr[i]

        images, labels = batch

        for image in images:

            featureMapForTeacher = getFeatureMaps(teacher_model, device, image)[featureMapNumForTeacher]
            featureMapForStudent = getFeatureMaps(student_model, device, image)[featureMapNumForStudent]

            # Normalize tensor so NaN values do not get produced by loss function
            t = normalize(featureMapForTeacher, p=1.0, dim=2)
            t = normalize(t, p=1.0, dim=3)
            s = normalize(featureMapForStudent, p=1.0, dim=2)
            s = normalize(s, p=1.0, dim=3)

            # Loss functions: Cosine, SSIM, PSNR and Euclidean dist
            distill_loss = 0
            if kd_loss_type == 'ssim':
                distill_loss = 1 - ssim_loss(s, t, max_val=1.0, window_size=1)
            elif kd_loss_type == 'psnr':
                distill_loss = 1 - psnr_loss(s, t, max_val=1.0)
            elif kd_loss_type == 'cosine':
                distill_loss = F.cosine_similarity(t.reshape(1, -1), s.reshape(1, -1))
            elif kd_loss_type == 'euclidean':
                distill_loss = pairwise_euclidean_distance(t.reshape(1, -1), s.reshape(1, -1))

            kd_loss_arr.append(distill_loss)

        total_loss = sum(kd_loss_arr)
        if i == len(featureMapNumForStudentArr) - 1:
            total_loss.backward()
        else:
            total_loss.backward(retain_graph=True)

    distill_optimizer_implemented.step()
    distill_optimizer_implemented.zero_grad()
