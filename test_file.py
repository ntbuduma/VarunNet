import os

model = torch.load("/models/model.10")
_, test_loader = dataset.get_data_loaders(batch_size)

# Mapping of incorrect to correct labels
dict_map = {}
dir = "./data/train"
ind = len(dir) + 1
arr = os.listdir(dir)
arr = sorted(arr)[1:]
for i in range(100):
    dict_map[str(i)] = arr[i]

filepath = 0
f = open("test.txt", "w")
for batch_num, (inputs, labels) in enumerate(test_loader, 1):
    prediction = model(inputs)
    prediction = prediction.to('cpu')
    _, top5 = torch.topk(prediction, k=5, dim=1)
    
    for i in range(len(top5)):
        filepath += 1
        curr_top5 = top5[filepath-1]
        filepathstr = 'test/' + str(filepath).zfill(8)+".jpg"
        actual_top5 = []
        for category in curr_top5:
            actual_category = dict_map[category]
            actual_top5.append(actual_category)
        actual_top5_str = " ".join(actual_top5)
        filepathstr += actual_top5_str + "\n"
        f.write(filepathstr)
