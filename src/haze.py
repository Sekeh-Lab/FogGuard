import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import cv2
import torch
from torchvision import transforms

def add_haz_1(img_f, beta):
    img_f = img_f / 255
    (row, col, chs) = img_f.shape
    center = (row // 2, col // 2)  
    size = math.sqrt(max(row, col)) 

    x, y = np.meshgrid(np.linspace(0, row, row, dtype=int),
                       np.linspace(0, col, col, dtype=int))
    d = -0.04 * np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) + size
    d = np.tile(d, (3, 1, 1)).T
    trans = np.exp(-d * beta)

    # A = 255
    A = 0.5
    hazy = img_f * trans + A * (1 - trans)
    # hazy = np.array(hazy, dtype=np.uint8)

    return hazy

def add_haz_0(img_f, beta):
    img_f = img_f / 255
    (row, col, chs) = img_f.shape
    center = (row // 2, col // 2)  
    size = math.sqrt(max(row, col)) 
    A = 0.5  

    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

def add_haze_tensor(img, beta):
    img_f = transforms.ToTensor()(img)
    # img_f = img_f #/ 255
    (chs, row, col) = img_f.shape
    center = (row // 2, col // 2)  
    size = math.sqrt(max(row, col)) 

    x, y = torch.meshgrid(torch.linspace(0, row, row, dtype=int),
                          torch.linspace(0, col, col, dtype=int), indexing='ij')
    d = -0.04 * torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) + size
    d = torch.tile(d, (3, 1, 1))
    trans = torch.exp(-d * beta)

    # A = 255
    A = 0.5
    hazy = img_f * trans + A * (1 - trans)
    # hazy = np.array(hazy, dtype=np.uint8)

    return hazy.permute(1, 2, 0)


img_dir = "/home/soheil/gpu/adverse_weather/torch-yolov3/data/samples/herd_of_horses.jpg"
# img_numpy = cv2.imread(img_dir) / 255
img_numpy = np.array(Image.open(img_dir).convert('RGB'))

beta = 0.06
fig, ax = plt.subplots(2)
ax[0].imshow(add_haze_tensor(Image.open(img_dir).convert('RGB'), beta=beta))
ax[1].imshow(add_haz_1(img_numpy, beta=beta))
ax[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
ax[1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
fig.tight_layout()
fig.savefig(img_dir + 'haze.jpg')
