import os
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchvision import datasets, models
from torchvision import transforms
import matplotlib.pyplot as plt
import datasets as ds
# import torchvision.transforms.v2 as transforms
from collections import defaultdict
import PIL.Image
from torchinfo import summary

class LCANet(nn.Module):
    def __init__(self, in_channel, h_in, w_in):
        "docstring"
        super(LCANet, self).__init__()
        self.encoder = nn.Sequential(
            # (3, 512, 512)
            nn.Conv2d(in_channel, 50, 3, padding='same'),
            # (b, 50, 512, 512)
            nn.ReLU(),
            nn.AvgPool2d(2),
            # (b, 50, 256, 256)
            nn.Conv2d(50, 50, 3, padding='same'),
            # (b, 50, 256, 256)
            nn.ReLU(),
            nn.AvgPool2d(2)
            # (b, 50, 128, 128)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(50, 10, 3, padding='same'),
            # (b, 10, 256, 256)
            nn.ReLU(),
            nn.AvgPool2d(2),
            # (b, 10, 64, 64)
            # (50, 10, 128, 128)
            # (h_in // 4 * w_in // 4), (h_in // 4 * w_in // 4)
            nn.Flatten(),
            # nn.Linear(50 * h_in // 4 * w_in // 4, 10* h_in // 4 * w_in // 4),
            nn.Linear(10 * 64 * 64, 50 * 64 * 64),
            # (10, 10, 128, 128)
            nn.ReLU(),
            # nn.Linear(10, 50 * h_in // 4 * w_in // 4),
            # # (10, 10, 128, 128)
            # nn.ReLU()
            nn.Upsample(scale_factor=2),
            # (10, 50, 128, 128)
        )
        self.decoder = nn.Sequential(
            # (10, 10, 128, 128)
            nn.Conv2d(50, 50, 3, padding='same'),
            # (10, 50, 128, 128)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            # (10, 50, 256, 256)
            nn.Conv2d(50, 50, 2, padding='same'),
            # (50, 50, 256, 256)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            # (50, 50, 512, 512)
            nn.Conv2d(50, 3, 3, padding='same'),
            # (50, 3, 512, 512)
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.bottleneck(x.view(x.size(0), -1))
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
        

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channel):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(in_channel, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, in_channel, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train(model, loader, loss_function, optimizer, device, epoch):
    losses = []
    # outputs = []
    loss = 0
    for images in tqdm(loader):
        clear_image = images[0].to(device)
        for image in images:
            image = image.to(device)

            # Output of Autoencoder
            reconstructed = model(image)

            # Calculating the loss function
            loss = loss_function(reconstructed, clear_image)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.detach().item())
            # outputs.append((epoch, image, reconstructed))

    return losses

def main():
    ckpt_dir = "../checkpoints/autoencoder.pt"
    batch_size = 8
    image_size = 256
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Transforms images to a PyTorch Tensor
    transform = transforms.Compose(
        [
            # transforms.ToImageTensor(),
            # transforms.ConvertImageDtype(),
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ]
    )
    inv_transform = transforms.Compose(
        [
            transforms.Normalize(mean=[-m/s for m, s in zip(norm_mean, norm_std)],
                                 std=[1/x for x in norm_std])
        ]
    )
    train_dataset = ds.CustomImageDataset(img_dir="/home/soheil/data/coco/images/train2014/",
                                          transform=transform,
                                          fog_level=35)
    # transform = transforms.Compose(
    #     [
    #         transforms.RandomPhotometricDistort(),
    #         transforms.RandomZoomOut(
    #             fill=defaultdict(lambda: 0, {PIL.Image.Image: (123, 117, 104)})
    #         ),
    #         transforms.RandomIoUCrop(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToImageTensor(),
    #         transforms.ConvertImageDtype(torch.float32),
    #         transforms.SanitizeBoundingBox(),
    #     ]
    # )

    # train_dataset = ds.load_example_coco_detection_dataset(transforms=transform)
    # train_dataset = datasets.CocoDetection("/home/soheil/data/coco/images/train2014",
    #                                        "/home/soheil/data/coco/annotations/instances_train2014.json")

    # train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset)
    # sample = train_dataset[0]
    # ds.show(sample)
    # DataLoader is used to load the dataset for training
    train_dl = torch.utils.data.DataLoader(dataset = train_dataset,
                                        batch_size = batch_size,
                                        # collate_fn=lambda batch: tuple(zip(*batch)),
                                        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model Initialization
    # model = ConvAutoencoder(in_channel=3).to(device)
    model = LCANet(3, 512, 512)
    summary(model, input_size=(batch_size, 3, image_size, image_size))

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-4,
                                weight_decay = 1e-8)

    # if os.path.exists(ckpt_dir):
    #     checkpoint = torch.load(ckpt_dir)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = 10
    losses = []
    for epoch in trange(epochs):
        loss = train(model, train_dl, loss_function, optimizer, device,
                             epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,},
                   ckpt_dir)

        losses.append(loss)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot([l for loss in losses for l in loss])
    plt.savefig('../output/autoencoder-loss.png')

    num_samples = 4
    # rnd_id = torch.randint(0, len(output), (num_samples,))
    # outputs = [out for idx, out in enumerate(output) if idx in rnd_id]

    fig, axs = plt.subplots(num_samples, 2)

    images = next(iter(train_dl))[:num_samples][0]
    output = model(images.to(device)).detach().cpu()

    images = inv_transform(images)
    output - inv_transform(output)


    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=num_samples, ncols=2)
    # , sharex=True, sharey=True, figsize=(25,4))
    # input images on top row, reconstructions on bottom
    # for images, row in zip([images, output], axes):
    #     for img, ax in zip(images, row):
    #         # ax.imshow(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
    #         ax.imshow(img.permute(1, 2, 0))
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)


    for i in range(num_samples):
        axes[i, 0].imshow(images[i].permute(1, 2, 0))
        axes[i, 1].imshow(output[i].permute(1, 2, 0))

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.savefig('../output/autoencoder.png')

if __name__ == '__main__':
    main()
