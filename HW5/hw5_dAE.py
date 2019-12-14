import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


# hyperparameters initialization
epoch_num = 10
batch_size = 64
lr = 0.0002


# read data
transform = transforms.Compose([transforms.ToTensor(),])
trainset = MNIST(root = "./data/", transform = transform, train = True, download = True)
testset = MNIST(root="./data/", transform = transform, train = False, download = True)
dataloader_train = DataLoader(dataset=trainset, batch_size = batch_size, shuffle=True)
dataloader_test = DataLoader(dataset=testset, batch_size = batch_size, shuffle=True)

class AutoEncoder(nn.Module):
    """
    This is a class for auto-encoder
    we have an encoder layer: fc-ReLu-fc-ReLu
    and a decoder layer: fc-ReLu-fc-Sigmoid
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid())

    def forward(self, x): # forward propagation
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def plot_img(im1, im2):
    """
    This function is used to plot comparison image of
    noise test images and reconstructed test images
    2 rows, 5 columns of images
    """
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(im1[i].reshape(28,28), cmap='gray') # reshape image 1 and show
        frame = plt.gca() # eliminate the axises
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

    for i in range(5):
        plt.subplot(2, 5, i+1+5)
        plt.imshow(im2[i].reshape(28,28), cmap='gray') # reshape image 2 and show
        frame = plt.gca() # eliminate the axises
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    
    plt.subplots_adjust(wspace =0, hspace=0, top=0.6)
    plt.savefig("p5_comparison.eps")
    plt.show()

# parameters initialization
dAE = AutoEncoder()
dAE_optimizer = torch.optim.Adam(dAE.parameters(), lr=lr) # using adam as optimizer
criterion = nn.BCELoss() # using BCE Loss as loss funciton


def train_dAE():
    """
    This is a function used to train autoencoder
    return - losses: a list consisting average loss of each epoch
    """
    losses = []
    for epoch in range(epoch_num):
        sum_loss = 0 # total loss for this epoch
        for step, (img, _) in enumerate(dataloader_train): 
            # img is the training image, we need to reshape it first
            img = img.view(img.size(0), -1) # [64, 1, 28, 28] --> [64, 784]

            noisy_img = img + torch.randn(img.size()) * 0.5 # add some gaussian noise to original image

            # re_img is the reconstructed image, 
            # which is generated through forward propagation of autoencoder
            re_img = dAE(noisy_img)
            # get the loss for this iteration
            loss = criterion(re_img, img)
            sum_loss += loss.item() * batch_size

            # Backpropagation, update weights
            dAE_optimizer.zero_grad()
            loss.backward()
            dAE_optimizer.step()

        # print some info since an epoch is finished.
        print('Epoch [{}/{}], Loss:{:.6f}'.format(epoch+1, epoch_num, sum_loss / 60000))
        # Save average loss for each epoch
        losses.append(sum_loss / 60000)

    return losses

# Get the loss list
losses = train_dAE()


# Plot loss for each epoch
plt.plot(np.arange(1, epoch_num+1),losses)
plt.savefig("p5_loss.eps")
plt.show()

torch.save(dAE, 'hw5_dAE.pth')



"""
The following code are used to plot noise image and reconstructed image
"""
model = torch.load('hw5_dAE.pth')
img_test = next(iter(dataloader_test))[0].data[:5].view(5, -1) # true test image
noisy_img_test = img_test + torch.randn(img_test.size()) * 0.5 # noise test image
re_img_test = model(noisy_img_test) # reconstructed test image
plot_img(noisy_img_test.data, re_img_test.data)