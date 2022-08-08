import torch
import torch.nn as nn

from NeuralNetwork.Helper import distill

# Speed up training
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(model, test_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_dl]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_student(student_model, numOfEpochs, train_dl, test_dl, optimizer, max_lr, weight_decay, scheduler,
                  grad_clip=None):
    if numOfEpochs == 0:
        return student_model

    torch.cuda.empty_cache()
    history = []

    optimizer = optimizer(student_model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, max_lr, epochs=numOfEpochs, steps_per_epoch=len(train_dl))

    for epoch in range(numOfEpochs):
        student_model.train()  # put the model in train mode
        train_loss = []
        train_acc = []
        lrs = []
        batch_count = 0

        for batch in train_dl:

            batch_count += 1
            # print(batch_count)

            # Normal error and update
            loss, acc = student_model.training_step(batch)
            train_loss.append(loss)
            train_acc.append(acc)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(student_model.parameters(), grad_clip)

            optimizer.step()
            for param in student_model.parameters():  # instead of: optimizer.zero_grad()
                param.grad = None

            # Step scheduler
            scheduler.step()
            lrs.append(get_lr(optimizer))

        # Add results:
        result = evaluate(student_model, test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["train_acc"] = torch.stack(train_acc).mean().item()
        result["lrs"] = lrs
        student_model.epoch_end(epoch, result)
        history.append(result)

    torch.save(student_model.state_dict(), "../../../NeuralNetwork/resnet20.ckpt")
    print("Student model partially trained for: " + str(numOfEpochs) + " epochs.")
    return student_model


def train_model(epochs, train_dl, test_dl, model, optimizer, max_lr, weight_decay, scheduler, grad_clip=None):
    torch.cuda.empty_cache()
    history = []

    optimizer = optimizer(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        model.train()  # put the model in train mode
        train_loss = []
        train_acc = []
        lrs = []

        for batch in train_dl:
            loss, acc = model.training_step(batch)
            train_loss.append(loss)
            train_acc.append(acc)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            scheduler.step()
            lrs.append(get_lr(optimizer))

        result = evaluate(model, test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["train_acc"] = torch.stack(train_acc).mean().item()
        result["lrs"] = lrs

        model.epoch_end(epoch, result)
        history.append(result)

    return history


def train_model_with_distillation(heuristicString, heuristicToLayerDict, epochs, train_dl, test_dl, student_model,
                                  student_model_number, teacher_model,
                                  teacher_model_number, device, optimizer, max_lr,
                                  weight_decay, scheduler, kd_loss_type, distill_optimizer,
                                  distill_lr,
                                  grad_clip=None):
    torch.cuda.empty_cache()
    history = []

    optimizer = optimizer(student_model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        student_model.train()  # put the model in train mode
        train_loss = []
        train_acc = []
        lrs = []
        batch_count = 0

        for batch in train_dl:

            batch_count += 1
            # print(batch_count)

            # Normal error and update
            loss, acc = student_model.training_step(batch)
            train_loss.append(loss)
            train_acc.append(acc)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(student_model.parameters(), grad_clip)

            optimizer.step()
            for param in student_model.parameters():  # instead of: optimizer.zero_grad()
                param.grad = None

            # Step scheduler
            scheduler.step()
            lrs.append(get_lr(optimizer))

        distill(heuristicString, heuristicToLayerDict, kd_loss_type, distill_optimizer, distill_lr,
                next(iter(train_dl)),
                student_model,
                student_model_number, teacher_model, teacher_model_number, device)

        # Add results:
        result = evaluate(student_model, test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["train_acc"] = torch.stack(train_acc).mean().item()
        result["lrs"] = lrs
        student_model.epoch_end(epoch, result)
        history.append(result)

    return history


def train_model_distill_only(numberOfEpochs, heuristicString, heuristicToLayerDict, train_dl, test_dl,
                             student_model, student_model_number, teacher_model,
                             teacher_model_number, device, kd_loss_type, distill_optimizer,
                             distill_lr):
    count = 0

    for batch in train_dl:

        count += 1

        distill(heuristicString, heuristicToLayerDict, kd_loss_type, distill_optimizer, distill_lr,
                batch,  # HERE change to use with random batch
                student_model,
                student_model_number, teacher_model, teacher_model_number, device)

        if count >= numberOfEpochs:
            break


def get_model_distill_loss_only(numberOfEpochs, heuristicString, heuristicToLayerDict, train_dl, test_dl,
                                student_model, student_model_number, teacher_model,
                                teacher_model_number, device, kd_loss_type, distill_optimizer,
                                distill_lr):
    count = 0
    lossArr = []
    for batch in train_dl:

        count += 1

        lossArr += distill(heuristicString, heuristicToLayerDict, kd_loss_type, distill_optimizer, distill_lr,
                           batch,
                           student_model,
                           student_model_number, teacher_model, teacher_model_number, device, lossOnly=True)

        if count >= numberOfEpochs:
            break

    return lossArr


def train_model_normal_and_distill(heuristicString, heuristicToLayerDict, epochs, train_dl, test_dl, student_model,
                                   student_model_number, teacher_model,
                                   teacher_model_number, device, optimizer, max_lr,
                                   weight_decay, scheduler, kd_loss_type, distill_optimizer,
                                   distill_lr,
                                   grad_clip=None):
    torch.cuda.empty_cache()

    optimizer = optimizer(student_model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        student_model.train()  # put the model in train mode
        lrs = []

        # Normal error and update

        for batch in train_dl:
            loss, acc = student_model.training_step(batch)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(student_model.parameters(), grad_clip)

            optimizer.step()
            for param in student_model.parameters():  # instead of: optimizer.zero_grad()
                param.grad = None

            # Step scheduler
            scheduler.step()
            lrs.append(get_lr(optimizer))

        distill(heuristicString, heuristicToLayerDict, kd_loss_type, distill_optimizer, distill_lr,
                next(iter(train_dl)),
                student_model,
                student_model_number, teacher_model, teacher_model_number, device, False)
