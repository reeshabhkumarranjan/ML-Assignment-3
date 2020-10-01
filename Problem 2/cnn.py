# -*- coding: utf-8 -*-

# imports
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

from google.colab import drive
drive.mount('/content/drive')

import torch, torchvision, torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
fashiontrain = FashionMNIST(root='/content/drive/My Drive/Colab Notebooks/ML Assignment 3', train=True, download=True, transform=transform)
fashiontest = FashionMNIST(root='/content/drive/My Drive/Colab Notebooks/ML Assignment 3', train=False, download=True, transform=transform)
svm_train_y = fashiontrain.targets.detach().numpy().reshape(-1, 1)
svm_test_y = fashiontest.targets.detach().numpy().reshape(-1, 1)
print(svm_train_y.shape)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainloader = torch.utils.data.DataLoader(fashiontrain, batch_size=100, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(fashiontest, batch_size=100, shuffle=False, num_workers=2)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Defining CNN

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.svm_x = None
        self.svm_x_is_None = True
        self.conv1 = nn.Conv2d(1, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.count = 0
        self.create_svm_x = False
        self.create_svm_test_x = False
        self.svm_test_x = None
        self.svm_test_x_is_None = True
        
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        self.count += 1
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.fc1(x)
        _x = x.cpu().detach().numpy()

        if self.create_svm_x:
            if self.svm_x_is_None:
                self.svm_x = _x
                self.svm_x_is_None = False
            else:
                self.svm_x = np.concatenate((self.svm_x, _x), axis=0)
        
        if self.create_svm_test_x:
            if self.svm_test_x_is_None:
                self.svm_test_x = _x
                self.svm_test_x_is_None = False
            else:
                self.svm_test_x = np.concatenate((self.svm_test_x, _x), axis=0)
            
        x = F.relu(x)
        x = self.fc2(x)
        return x


net = Net()
net.to(device)

# Training CNN

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
print(len(trainloader))

net.create_svm_x = True
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    
    net.create_svm_x = False

print(net.svm_x.shape)
print(net.count)
print('Finished Training')

# Testing CNN

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)

correct = 0
total = 0
net.create_svm_test_x = True
with torch.no_grad():
    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(net.svm_test_x.shape)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

import torch
from sklearn.metrics import confusion_matrix
trainloader1 = torch.utils.data.DataLoader(fashiontrain, batch_size=60000, shuffle=True, num_workers=2)
testloader1 = torch.utils.data.DataLoader(fashiontest, batch_size=10000, shuffle=False, num_workers=2)

with torch.no_grad():
    for data in testloader1:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        cm_test = confusion_matrix(labels,predicted)
        
with torch.no_grad():
    for data in trainloader1:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        cm_train = confusion_matrix(labels,predicted)
        
print(cm_train)
print(cm_test)

# Defining SVM
import sklearn.svm as svm
svClassifier = svm.SVC(kernel='rbf', gamma='scale', verbose=True)
svClassifier.fit(net.svm_x, svm_train_y.reshape(-1, ))

# Generating test dataset for SVM
svClassifier.score(net.svm_test_x, svm_test_y.reshape(-1, ))

svm_test_y.reshape(-1, )

svm_test_y.shape
