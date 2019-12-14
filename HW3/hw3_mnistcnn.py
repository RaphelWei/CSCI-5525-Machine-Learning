import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time

train = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
test = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
# default training and testing set, batch_size = 32
trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

class CNN(nn.Module):
    """
    Define our CNN model
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(1, 20, 3, 1), # convolution layer, input 1 channel, output 20 channels, kernel 3x3, stride=1
                    nn.MaxPool2d(2, 2), # max pooling layer
                    nn.ReLU()) # ReLU layer
        self.fc  = nn.Sequential(
                    nn.Linear(13*13*20, 128), # fc layer 1, after flatten layer, input 13*13*20, output 128
                    nn.ReLU(),
                    nn.Linear(128, 10)) # fc layer 2,  input 128, output 10
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1) # flatten layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1) # softmax layer

def train_cnn(model, train_set, optimizer, losses, acces):
    """@params:
    model: The model you want to train
    train_set: By varying the batch_size, we might have different training set. The default value is training set with batch_size=32
    optimizer: Optimizer we use for training. Default value is SGD.
    losses: A list which is used to store loss in each epoch.
    acces: A list which is used to store training accuracy in each epoch.
    """
    last_loss = 10000   # initialize last_loss as a large numer
                        # this variable will be used to determine whether our model converges
    for e in range(100):
        train_loss = 0
        train_acc = 0
        model.train()
        for im, label in train_set:
            # forward broadcast
            out = model(im)
            # calculating loss using cross entropy loss
            loss = F.nll_loss(out, label)
            # backward broadcast to update parameters
            optimizer.zero_grad() # set the gradient to 0
            loss.backward()
            optimizer.step()
            # calculating total loss in each epoch
            train_loss += loss.item()
            # calculating training accuracy in each epoch
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item() # numbers of correct predict
            acc = num_correct / im.shape[0]
            train_acc += acc
        # calculating loss and accuracy respectively
        current_loss = train_loss / len(train_set)
        current_acc = train_acc / len(train_set)
        print("epoch: {}, Train loss: {:.6f}, Train accuracy: {:.2f}%".format(e, current_loss, current_acc*100))
        # If the difference of losses between last epoch and current epoch is less than a threshold,
        # and current_loss is less than a threshold,
        # we will consider the model has converged.
        if abs(last_loss-current_loss) < 0.001 and current_loss < 0.05:
            print("********************It converges********************")
            break # quit training
        else: # model not converges yet
            last_loss = current_loss # update 
            losses.append(current_loss) # add loss and accuracy to the list respectively
            acces.append(current_acc)

def main():
    """
    our main function to train CNN model
    after the model converges, it will save the model
    """
    # initialize parameters
    losses = []
    acces = []
    cnn = CNN()
    # train our model
    train_cnn(cnn, 
              trainset, 
              optim.SGD(cnn.parameters(), lr=0.01), 
              losses, 
              acces)
    # save model
    torch.save(cnn, './mnist-cnn.pt')
    # plot loss and accuracy
    plt.figure(1)
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Loss", fontsize = 16)
    plt.savefig("loss_cnn.png")
    plt.figure(2)
    plt.plot(acces)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.savefig("acc_cnn.png")
    plt.show()

main()

def test_cnn():
    """
    This function is used to reload our model
    and test the reloaded model on test set to get the accuracy
    """
    model = torch.load('./mnist-cnn.pt')
    test_acc=0
    for im, label in testset:
        output = model(im)
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        test_acc += acc
    print("The accuracy of test data is {0:.2f}%".format(test_acc/len(testset)*100))


test_cnn()

"""
The following part is used for problem 4 and 5
vary batch sizes and optimizers respectively.
You could comment the codes below this line
If you want to run my code
Because they are much too time consuming
"""








def compare_batch_size():
    """
    this is a function to test CNN
    on different batch sizes
    """
    # params initialization
    batch_size = [32, 64, 96, 128]
    model_list = [CNN(), CNN(), CNN(), CNN()]
    times = []
    loss_list = []
    acc_list = []
    # training
    for i in range(4):
        losses = []
        acces = []
        size = batch_size[i]
        model_compare_batch = model_list[i]
        # given batch size, get the new train and test set
        new_trainset = torch.utils.data.DataLoader(train, batch_size = size, shuffle=True)
        new_testset = torch.utils.data.DataLoader(test, batch_size = size, shuffle=False)
        # record start time of training
        start_time = time()
        train_cnn(model_compare_batch, 
                  new_trainset, 
                  optim.SGD(model_compare_batch.parameters(), lr=0.01), # Using SGD as optimizer
                  losses, 
                  acces)
        end_time = time()
        # record running time in minute
        time0 = (end_time - start_time) / 60
        print("batch_size: {}, converge time: {:.2f}(minitue).".format(size, time0))
        times.append(time0)
        loss_list.append(losses)
        acc_list.append(acces)
    return times, loss_list, acc_list



"""
The following code is used to test 
func `compare_batch_size()`
and plot figures
"""
times, loss_batch, acc_batch = compare_batch_size()
x_tick = [32, 64, 96, 128]
plt.plot(x_tick, times, marker="o")
plt.xlabel("Batch size", fontsize = 16)
plt.ylabel("Converge time(min)", fontsize = 16)
plt.xticks(x_tick)
plt.savefig("batch-time.png")
plt.figure(2)
plt.plot(loss_batch[0], label = "32")
plt.plot(loss_batch[1], label = "64")
plt.plot(loss_batch[2], label = "96")
plt.plot(loss_batch[3], label = "128")
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Loss", fontsize = 16)
plt.legend()
plt.savefig("batch-loss.png")




def compare_optim(loss_list, acc_list):
    """
    This function is used to test performances of our model
    on different optimizers
    """
    # params initialization
    model_list = [CNN(), CNN(), CNN()]
    SGD = optim.SGD(model_list[0].parameters(), lr=0.01)
    ADAM = optim.Adam(model_list[1].parameters(), lr=0.001)
    ADAGRAD = optim.Adagrad(model_list[2].parameters(), lr=0.01)
    optim_list = [SGD, ADAM, ADAGRAD]
    losses = []
    acces = []
    # training
    for i in range(3):
        model_compare_optim = model_list[i]
        optimizer = optim_list[i]
        losses = []
        acces = []
        train_cnn(model_compare_optim, 
                  trainset, 
                  optimizer, 
                  losses, 
                  acces)
        loss_list.append(losses)
        acc_list.append(acces)



"""
The following code is used to test func `compare_optim()`
and plot figures
"""
loss_list = []
acc_list = []
compare_optim(loss_list, acc_list)

plt.figure(1)
plt.plot(loss_list[0], label = "SGD")
plt.plot(loss_list[1], label = "ADAM")
plt.plot(loss_list[2], label = "ADAGRAD")
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Loss", fontsize = 16)
plt.legend()
plt.savefig("loss-optim.png")

plt.figure(2)
plt.plot(acc_list[0], label = "SGD")
plt.plot(acc_list[1], label = "ADAM")
plt.plot(acc_list[2], label = "ADAGRAD")
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
plt.legend()
plt.savefig("acc-optim.png")
plt.show()