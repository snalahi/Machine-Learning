'''
Names (Please write names in <Last Name, First Name> format):
1. Alahi, Sk Nasimul

TODO: Project type: Image Denoising
Built a convolutional neural network to denoise images using the VOC dataset

TODO: Report what each member did in this project
All the works associated with coding the entire program and writing the project report
were done by myself

'''


import os
import math
import torch
import argparse 
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import Axes3D


# Constants
NOISE_FACTOR = 0.5

parser = argparse.ArgumentParser()

# Commandline arguments
parser.add_argument('--train_network',
    action='store_true', help='If set, then trains network')
parser.add_argument('--batch_size',
    type=int, default=10, help='Number of samples per batch')
parser.add_argument('--n_epoch',
    type=int, default=20, help='Number of times to iterate through dataset')
parser.add_argument('--learning_rate',
    type=float, default=1e-3, help='Base learning rate (alpha)')


args = parser.parse_args()


class FullyConvolutionalNetwork(torch.nn.Module):
    '''
    Fully convolutional network

    Args:
        Please add any necessary arguments
    '''
    
    def __init__(self):
        super(FullyConvolutionalNetwork, self).__init__()
        
        # encoder layers
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        # TODO: Implement forward function
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation

        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.out(x))
        return x
    
def train(net,
          dataloader,
          n_epoch,
          optimizer):
    '''
    Trains the network using a learning rate scheduler

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights

        Please add any necessary arguments

    Returns:
        torch.nn.Module : trained network
    '''
    
    # The loss function
    criterion = nn.MSELoss()
    
    train_loss = []
    for epoch in range(n_epoch):
        running_loss = 0.0
        for data in dataloader:
            img, _ = data # we do not need the image labels
            # add noise to the image data
            img_noisy = img + NOISE_FACTOR * torch.randn(img.shape)
            # clip to make the values fall between 0 and 1
            img_noisy = np.clip(img_noisy, 0., 1.)
            img_noisy = img_noisy.to(device)
            optimizer.zero_grad()
            outputs = net(img_noisy)
            loss = criterion(outputs, img_noisy)
            # backpropagation
            loss.backward()
            # update the parameters
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(dataloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, n_epoch, loss))
    return train_loss

def evaluate(net, dataloader):
    '''
    Evaluates the network on a dataset

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data

        Please add any necessary arguments
    '''
    
    for batch in dataloader:
        img, _ = batch
        img_noisy = img + NOISE_FACTOR * torch.randn(img.shape)
        img_noisy = np.clip(img_noisy, 0., 1.)
        img_noisy = img_noisy.to(device)
        outputs = net(img_noisy)
        save_image(img_noisy, 'noisy_image.png')
        save_image(outputs, 'denoised_clear_image.png')
        break

    psnr = peak_signal_to_noise_ratio(img_noisy, outputs)
    print('Peak Signal to Noise Ratio (PSNR): {:.4f} dB'.format(psnr))
    return img_noisy, outputs

def intersection_over_union(prediction, ground_truth):
    '''
    Computes the intersection over union (IOU) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : intersection over union
    '''

    # TODO: Computes intersection over union score
    # Implement ONLY if you are working on semantic segmentation

    return 0.0

def peak_signal_to_noise_ratio(prediction, ground_truth):
    '''
    Computes the peak signal to noise ratio (PSNR) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : peak signal to noise ratio
    '''

    # TODO: Computes peak signal to noise ratio
    # Implement ONLY if you are working on image reconstruction or denoising

    mse = mean_squared_error(prediction, ground_truth)
    if mse == 0:
        return 100
    
    PIXEL_MAX = 255.0
    
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mean_squared_error(prediction, ground_truth):
    '''
    Computes the mean squared error (MSE) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : mean squared error
    '''

    # TODO: Computes mean squared error
    # Implement ONLY if you are working on image reconstruction or denoising

    loss = nn.MSELoss()

    return loss(prediction, ground_truth)

def plot_images(X, Y, n_row, n_col, fig_title_X, fig_title_Y):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy
            N x h x w input data
        Y : numpy
            N x h x w predictions
        n_row : int
            number of rows in figure
        n_col : list[str]
            number of columns in figure
        fig_title_X : str
            title of input plot
        fig_title_Y : str
            title of output plot

        Please add any necessary arguments
    '''

    X = X / 2 + 0.5
    Y = Y / 2 + 0.5
    
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(fig_title_X)
    for idx_x in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, idx_x+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(X[idx_x], (1, 2, 0)))
    
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(fig_title_Y)
    for idx_y in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, idx_y+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(Y[idx_y], (1, 2, 0)))

    plt.show()
        

### Helper method to facilitate the program execution
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


if __name__ == '__main__':
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((140, 140)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])
    
    voc_train = torchvision.datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=True,
        transform=transform,
        target_transform=transform
    )
    voc_test = torchvision.datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=True,
        transform=transform,
        target_transform=transform
    )
    dataloader_train = torch.utils.data.DataLoader(
        voc_train, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    dataloader_test = torch.utils.data.DataLoader(
        voc_test, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    net = FullyConvolutionalNetwork()
    
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    
    if args.train_network:
        device = get_device()
        net.to(device)
        
        train_loss = train(
                    net=net,
                    dataloader=dataloader_train,
                    n_epoch=args.n_epoch,
                    optimizer=optimizer)
        
        plt.figure()
        plt.plot(train_loss)
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        torch.save({ 'state_dict' : net.state_dict()}, "./checkpoint.pth")
    else:
        checkpoint = torch.load("./checkpoint.pth")
        net.load_state_dict(checkpoint['state_dict'])
        
    noisy_image, clear_image = evaluate(
                                net=net,
                                dataloader=dataloader_test)
    
    noisy_image = noisy_image.detach().numpy()
    clear_image = clear_image.detach().numpy()
    
    plot_images(X=noisy_image, Y=clear_image, n_row=2, n_col=5,
            fig_title_X='Noisy Images', fig_title_Y='Denoised Clear Images')

    