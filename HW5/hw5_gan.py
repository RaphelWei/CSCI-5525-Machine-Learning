import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST


# hyperparameters initialization
epoch_num = 50
batch_size = 100
lr = 0.0002
noise_dim = 128

# read data
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=(0.5,), std=(0.5,))]) # normalize the data

mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=False)

class Generator(torch.nn.Module):
    """
    This is a class of our discriminator
    structure: fc-leakyReLu-fc-leakyReLu-fc-Tanh
    """
    def __init__(self):
        super(Generator, self).__init__()    
        self.layer1 = nn.Sequential(
            nn.Linear(128, 256), # FC layer, input 128, output 256
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Discriminator(torch.nn.Module):
    """
    This is a class of our discriminator
    structure: fc-leakyReLu-fc-leakyReLu-fc-Sigmoid
    """
    def __init__(self):
        super(Discriminator, self).__init__()      
        self.layer1 = nn.Sequential( 
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def plot_img(img, iter):
    """
    This is a function to plot generated images
    @params:
    img - generated images
    iter - used to name the plots
    """
    plt.figure(figsize=[6, 6])
    for i in range(4*4):
        plt.subplot(4, 4, i+1)
        plt.imshow(img[i].reshape(28,28), cmap='gray')
        frame = plt.gca() # eliminate axises
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace =0.05, hspace=0.05)
    # name plots with number of iter
    plt.savefig("p4_images_"+str(iter)+".eps")
    plt.show()


# parameters initialization
G = Generator() # Generator model we would like to train
D = Discriminator() # Discriminator model we would like to train

G_optimizer = torch.optim.Adam(G.parameters(), lr=lr) # using ADAM as optimizer
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr) # using ADAM as optimizer
criterion = nn.BCELoss()


def train_GAN():
    """
    This is a function used to train GAN
    return - losses: a list consisting average loss of each epoch
    """
    G_losses = [] # initialize losses as an empty list
    D_losses = []
    for epoch in range(epoch_num):
        D_sum_loss = 0
        G_sum_loss = 0
        for _, (img, _) in enumerate(dataloader):
            """
            First, we do the preparation work.
            input random Gaussian noise for Generator to generate some fake images
            we combine the fake images and real images together
            next step, we will input the mixed data of fake and real images into Discriminator
            """
            # img denotes the real images
            img_num = img.size(0) # number of images
            real_img = img.view(img_num, -1) # reshape the real images
            real_label = torch.ones(img_num).view(-1,1) # initialize the labels of real images as all 1's 
        
            noise = torch.randn(img_num, noise_dim) # noise
            fake_img = G(noise) # using generator to generate fake images from noise
            fake_label = torch.zeros(img_num).view(-1,1) # initialize the labels of fake images as all 0's 

            combined_img = torch.cat((real_img, fake_img))
            combined_label = torch.cat((real_label, fake_label))

            """
            In this step, we will compute the loss of Discriminator
            we first use forward propagation of Discriminator to predict labels
            of mixed data.
            """
            # Loss of Discriminator
            D_predicts = D(combined_img) # prediction of Discriminator on mixed data
            D_loss = criterion(D_predicts, combined_label) # compute loss      
            D_sum_loss += D_loss.item() * batch_size
        
            # Backpropagation, update weights of Discriminator
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            """
            In this step, we will compute loss of Generator
            Our purpose is to make fake images look "real"
            so we compute loss between (prediction of fake images) and (real labels)
            """
            # Loss for Generator
            noise = torch.randn(img_num, noise_dim) # noise
            fake_img = G(noise) # generated fake images
            D_predicts_fake = D(fake_img)
            G_loss = criterion(D_predicts_fake, real_label)
            G_sum_loss += G_loss.item() * batch_size

            # Backpropagation, update weights of Generator
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()


            """
            our steps for 1 iteration finishes.
            """



        print('Epoch [{}/{}], D_loss: {:.6f}, G_loss: {:.6f}'.format(epoch+1, epoch_num, D_sum_loss / 60000, G_sum_loss / 60000))

    
        # Save Loss Every epoch
        D_losses.append(D_sum_loss / 60000)
        G_losses.append(G_sum_loss / 60000)

        # plot 16 generated images every 10 epochs
        if (epoch+1) % 10 == 0: # epoch from 0 to 49, we plot when epoch=9,19,29,39,49
            noise = torch.randn(16, noise_dim)
            plot_img(G(noise).data, (epoch+1) / 10)

    return D_losses, G_losses



# Get the loss lists
D_losses, G_losses = train_GAN()

# Plot average loss for each epoch
plt.plot(np.arange(1, epoch_num+1), D_losses, label="Discriminator")
plt.plot(np.arange(1, epoch_num+1), G_losses, label="Generator")
plt.legend()
plt.savefig("p4_loss.eps")
plt.show()

torch.save(D, 'hw5_gan_dis.pth')
torch.save(G, 'hw5_gan_gen.pth')

