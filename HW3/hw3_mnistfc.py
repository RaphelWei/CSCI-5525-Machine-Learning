import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Read data
train = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
test = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
# set batch size of train set to 32                      
trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

# define our FC model
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) #fc layer 1, input 28*28, output 128
        self.fc2 = nn.Linear(128, 10) #fc layer 2, input 128, output 10

    def forward(self, x):
        x = F.relu(self.fc1(x)) # ReLU layer after fc layer 1
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # Softmax layer


def train_fc(model, losses, acces, train_set, optimizer):
    """@params:
    model: model we want to train
    losses: A list which is used to store loss in each epoch.
    acces: A list which is used to store training accuracy in each epoch.
    train_set: By varying the batch_size, we might have different training set. The default value is training set with batch_size=32
    optimizer: Optimizer we use for training. Default value is SGD.
    """
    last_loss = 10000   # initialize last_loss as a large numer
                        # this variable will be used to determine whether our model converges
    for e in range(100):
        train_loss = 0
        train_acc = 0
        model.train()
        for im, label in train_set:
            # forward broadcast
            out = model(im.view(-1,784))
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
        # and current loss is less than a threshold
        # we will consider the model has converged.
        if last_loss-current_loss < 0.001 and current_loss < 0.05:
            print("********************It converges********************")
            break # quit training
        else: # model not converges yet
            last_loss = current_loss # update last loss
            losses.append(current_loss) # add loss and accuracy to the list respectively
            acces.append(current_acc)

def main():
    """
    main function
    we will train our network
    save model and plot here
    """
    # initialize parameters
    fc = FC()
    losses = []
    acces = []
    SGD = optim.SGD(fc.parameters(), lr=0.01) # using SGD as optimizer
    train_fc(fc, losses, acces, trainset, SGD)
    torch.save(fc, './mnist-fc.pt')
    # plot figures
    plt.figure(1)
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize = 20)
    plt.ylabel("Loss", fontsize = 20)
    plt.savefig("loss.png")
    plt.figure(2)
    plt.plot(acces)
    plt.xlabel("Epochs", fontsize = 20)
    plt.ylabel("Accuracy", fontsize = 20)
    plt.savefig("acc.png")
    plt.show()

main()


def test_fc():
    """
    This is a function where we load our model and test the 
    accuracy of our model on test set.
    """
    model = torch.load('./mnist-fc.pt')
    test_acc=0
    for im, label in testset:
        output = model(im.view(-1,784))
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item() # number of correct predict
        acc = num_correct / im.shape[0]
        test_acc += acc
    print("The accuracy of test data is {0:.2f}%".format(test_acc/len(testset)*100))

test_fc()