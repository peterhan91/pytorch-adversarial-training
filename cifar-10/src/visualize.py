
import os
import torch
import torchvision as tv
import numpy as np

from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from src.utils import makedirs, tensor2cuda, load_model, LabelDict
from src.argument import parser
from src.visualization import VanillaBackprop
from src.model.madry_model import WideResNet

import matplotlib.pyplot as plt 

img_folder = 'img'
makedirs(img_folder)
out_num = 5


args = parser()

label_dict = LabelDict(args.dataset)

te_dataset = tv.datasets.CIFAR10(args.data_root, 
                               train=False, 
                               transform=tv.transforms.ToTensor(), 
                               download=True)

te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=48)


for data, label in te_loader:

    data, label = tensor2cuda(data), tensor2cuda(label)


    break


# model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
model = models.resnet50(pretrained=False)
num_classes=10
model.fc = nn.Linear(model.fc.in_features, num_classes)
load_model(model, args.load_checkpoint)

if torch.cuda.is_available():
    model.cuda()

VBP = VanillaBackprop(model)

grad = VBP.generate_gradients(data, label)

grad_flat = grad.view(grad.shape[0], -1)
mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3)
std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3)

mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
std = std.repeat(1, 1, data.shape[2], data.shape[3])

grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)

print(grad.min(), grad.max())

grad -= grad.min()

grad /= grad.max()

grad = grad.cpu().numpy().squeeze()  # (N, 28, 28)

grad *= 255.0

label = label.cpu().numpy()

data = data.cpu().numpy().squeeze()

data *= 255.0

out_list = [data, grad]

types = ['Original', 'Your Model']

fig, _axs = plt.subplots(nrows=len(out_list), ncols=out_num)

axs = _axs

for j, _type in enumerate(types):
    axs[j, 0].set_ylabel(_type)

    # if j == 0:
    #     cmap = 'gray'
    # else:
    #     cmap = 'seismic'

    for i in range(out_num):
        axs[j, i].set_xlabel('%s' % label_dict.label2class(label[i]))
        img = out_list[j][i]
        # print(img)
        img = np.transpose(img, (1, 2, 0))

        img = img.astype(np.uint8)
        axs[j, i].imshow(img)

        axs[j, i].get_xaxis().set_ticks([])
        axs[j, i].get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'cifar_grad_%s.jpg' % args.affix))