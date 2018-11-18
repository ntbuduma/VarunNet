import torch

model = torch.load("/models/model.10")
train_loader, val_loader = dataset.get_data_loaders(batch_size)

count = 0
accuracy_top1 = 0
accuracy_top5 = 0
for batch_num, (inputs, labels) in enumerate(train_loader, 1):
    prediction = model(inputs)
    prediction = prediction.to('cpu')
    _, cls = torch.max(prediction, dim=1)
    _, top5 = torch.topk(prediction, k=5, dim=1)
    for i in range(len(cls)):
        accuracy_top1 += int(cls[i] == labels[i])
        count += 1
    for i in range(len(top5)):
        accuracy_top5 += int(labels[i] in top5[i])           

accuracy_top1 /= count
accuracy_top5 /= count
print("training error: ", accuracy_top1, accuracy_top5)

count = 0
accuracy_top1 = 0
accuracy_top5 = 0
for batch_num, (inputs, labels) in enumerate(val_loader, 1):
    prediction = model(inputs)
    prediction = prediction.to('cpu')
    _, cls = torch.max(prediction, dim=1)
    _, top5 = torch.topk(prediction, k=5, dim=1)
    for i in range(len(cls)):
        accuracy_top1 += int(cls[i] == labels[i])
    for i in range(len(top5)):
        accuracy_top5 += int(labels[i] in top5[i])
    count += 1

accuracy_top1 /= count
accuracy_top5 /= count
print("validation error: ", accuracy_top1, accuracy_top5)
