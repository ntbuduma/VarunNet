import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.001)

    epoch = 1
    while epoch <= num_epochs:
        scheduler.step()
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
        # count = 0
        # accuracy_top1 = 0
        # accuracy_top5 = 0
        # for batch_num, (inputs, labels) in enumerate(train_loader, 1):
        #     prediction = model(inputs)
        #     prediction = prediction.to('cpu')
        #     _, cls = torch.max(prediction, dim=1)
        #     _, top5 = torch.topk(prediction, k=5, dim=1)
        #     for i in range(len(cls)):
        #         accuracy_top1 += int(cls[i] == labels[i])
        #         count += 1
        #     for i in range(len(top5)):
        #         accuracy_top5 += int(labels[i] in top5[i])           

        # accuracy_top1 /= count
        # accuracy_top5 /= count
        # print(accuracy_top1, accuracy_top5)

        # accuracy_top1 = 0
        # accuracy_top5 = 0
        # for batch_num, (inputs, labels) in enumerate(val_loader, 1):
        #     prediction = model(inputs)
        #     prediction = prediction.to('cpu')
        #     _, cls = torch.max(prediction, dim=1)
        #     _, top5 = torch.topk(prediction, k=5, dim=1)
        #     for i in range(len(cls)):
        #         accuracy_top1 += int(cls[i] == labels[i])
        #     for i in range(len(top5)):
        #         accuracy_top5 += int(labels[i] in top5[i])
        #     count += 1

        # accuracy_top1 /= count
        # accuracy_top5 /= count
        # print(accuracy_top1, accuracy_top5)

        gc.collect()
        epoch += 1
    

print('Starting training')
run()
print('Training terminated')
