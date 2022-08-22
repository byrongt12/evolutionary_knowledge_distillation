import torch  # need this for eval function
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import psnr_loss, ssim_loss
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchmetrics.functional import pairwise_euclidean_distance
from torch.nn.functional import normalize


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

def getRandomBatches(numOfBatches, dataLoader):
    randomBatches = []

    for _ in range(numOfBatches):
        randomBatches += [next(iter(dataLoader))]

    return randomBatches


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)


def distill(heuristicString, heuristicToLayerDict, kd_loss_type, optimizer, distill_optimizer, distill_lr, batch,
            student_model,
            student_model_number, teacher_model, teacher_model_number, device, lossOnly=False):
    student_model.train()  # put the model in train mode

    kd_loss_arr = []
    featureMapNumForStudentArr = []
    featureMapNumForTeacherArr = []
    distill_optimizer_implemented = distill_optimizer(student_model.parameters(), lr=distill_lr)

    for i in range(0, len(heuristicString)):
        if (i + 1) % 2 != 0:
            # student layer
            student_layer_number = heuristicToLayerDict[heuristicString[i]] % (student_model_number * 3 * 2)
            if student_layer_number == 0:
                student_layer_number = (student_model_number * 3 * 2)
            featureMapNumForStudentArr.append(student_layer_number)
        else:
            # teacher layer
            layerForStudent, blockForStudent, convForStudent = convertLayerToCode(student_model_number,
                                                                                  student_layer_number)
            layerForTeacher = layerForStudent

            teacher_layer_number = ((layerForTeacher - 1) * teacher_model_number * 2) + heuristicToLayerDict[
                heuristicString[i]]

            featureMapNumForTeacherArr.append(teacher_layer_number)

    if len(heuristicString) % 2 != 0:
        # featureMapNumForStudentArr > featureMapNumForTeacherArr, so add teacher layer
        layerForStudent, blockForStudent, convForStudent = convertLayerToCode(student_model_number,
                                                                              featureMapNumForStudentArr[-1])

        layerForTeacher = layerForStudent

        teacher_layer_number = ((layerForTeacher - 1) * teacher_model_number * 2) + (2 * teacher_model_number)

        featureMapNumForTeacherArr.append(teacher_layer_number)

    images, labels = batch

    for image in images:

        featureMapForTeacherArr = getFeatureMaps(teacher_model, device, image)
        featureMapForStudentArr = getFeatureMaps(student_model, device, image)

        for i in range(0, (len(featureMapNumForStudentArr))):

            featureMapNumForStudent = featureMapNumForStudentArr[0]
            featureMapNumForTeacher = featureMapNumForTeacherArr[0]

            featureMapForTeacher = featureMapForTeacherArr[featureMapNumForTeacher]
            featureMapForStudent = featureMapForStudentArr[featureMapNumForStudent]

            # Normalize tensor so NaN values do not get produced by loss function
            t = normalize(featureMapForTeacher, p=1.0, dim=2)
            t = normalize(t, p=1.0, dim=3)
            s = normalize(featureMapForStudent, p=1.0, dim=2)
            s = normalize(s, p=1.0, dim=3)

            # Loss functions: Cosine, SSIM, PSNR and Euclidean dist
            distill_loss = 0
            if kd_loss_type == 'ssim':
                distill_loss = -1 * ssim_loss(s, t, max_val=2.0, window_size=1)
            elif kd_loss_type == 'psnr':
                distill_loss = psnr_loss(s, t, max_val=1.0)
            elif kd_loss_type == 'cosine':  # best
                distill_loss = F.cosine_similarity(s.reshape(1, -1), t.reshape(1, -1))
            elif kd_loss_type == 'euclidean':
                distill_loss = pairwise_euclidean_distance(s.reshape(1, -1), t.reshape(1, -1))

            kd_loss_arr.append(distill_loss)

    if not lossOnly:
        for kd_loss in kd_loss_arr:
            kd_loss.backward(retain_graph=True)

        distill_optimizer_implemented.step()

        for param in student_model.parameters():  # instead of: optimizer.zero_grad()
            param.grad = None

    return kd_loss_arr


def distill56(heuristicString, heuristicToLayerDict, kd_loss_type, optimizer, distill_optimizer, distill_lr, batch,
              student_model,
              student_model_number, teacher_model, teacher_model_number, device, lossOnly=False):
    student_model.train()  # put the model in train mode

    kd_loss_arr = []
    featureMapNumForStudentArr = []
    featureMapNumForTeacherArr = []
    distill_optimizer_implemented = distill_optimizer(student_model.parameters(), lr=distill_lr)

    for i in range(0, len(heuristicString)):
        if (i + 1) % 2 != 0:
            # student layer
            student_layer_number = heuristicToLayerDict[heuristicString[i]]
            featureMapNumForStudentArr.append(student_layer_number)
        else:
            # teacher layer
            layerForStudent, blockForStudent, convForStudent = convertLayerToCode(student_model_number,
                                                                                  student_layer_number)
            layerForTeacher = layerForStudent
            # 1 - 36
            OldRange = (56 - 1)
            NewRange = (36 - 1)
            newValue = (((heuristicToLayerDict[heuristicString[i]] - 1) * NewRange) / OldRange) + 1

            teacher_layer_number = ((layerForTeacher - 1) * teacher_model_number * 2) + int(newValue)

            featureMapNumForTeacherArr.append(teacher_layer_number)

    if len(heuristicString) % 2 != 0:
        # featureMapNumForStudentArr > featureMapNumForTeacherArr, so add teacher layer
        layerForStudent, blockForStudent, convForStudent = convertLayerToCode(student_model_number,
                                                                              featureMapNumForStudentArr[-1])

        layerForTeacher = layerForStudent

        teacher_layer_number = ((layerForTeacher - 1) * teacher_model_number * 2) + (2 * teacher_model_number)

        featureMapNumForTeacherArr.append(teacher_layer_number)

    images, labels = batch

    for image in images:

        featureMapForTeacherArr = getFeatureMaps(teacher_model, device, image)
        featureMapForStudentArr = getFeatureMaps(student_model, device, image)

        for i in range(0, (len(featureMapNumForStudentArr))):

            featureMapNumForStudent = featureMapNumForStudentArr[0]
            featureMapNumForTeacher = featureMapNumForTeacherArr[0]

            featureMapForTeacher = featureMapForTeacherArr[featureMapNumForTeacher]
            featureMapForStudent = featureMapForStudentArr[featureMapNumForStudent]

            # Normalize tensor so NaN values do not get produced by loss function
            t = normalize(featureMapForTeacher, p=1.0, dim=2)
            t = normalize(t, p=1.0, dim=3)
            s = normalize(featureMapForStudent, p=1.0, dim=2)
            s = normalize(s, p=1.0, dim=3)

            # Loss functions: Cosine, SSIM, PSNR and Euclidean dist
            distill_loss = 0
            if kd_loss_type == 'ssim':
                distill_loss = -1 * ssim_loss(s, t, max_val=2.0, window_size=1)
            elif kd_loss_type == 'psnr':
                distill_loss = psnr_loss(s, t, max_val=1.0)
            elif kd_loss_type == 'cosine':  # best
                distill_loss = F.cosine_similarity(s.reshape(1, -1), t.reshape(1, -1))
            elif kd_loss_type == 'euclidean':
                distill_loss = pairwise_euclidean_distance(s.reshape(1, -1), t.reshape(1, -1))

            kd_loss_arr.append(distill_loss)

    if not lossOnly:
        for kd_loss in kd_loss_arr:
            kd_loss.backward(retain_graph=True)

        distill_optimizer_implemented.step()

        for param in student_model.parameters():  # instead of: optimizer.zero_grad()
            param.grad = None

    return kd_loss_arr
