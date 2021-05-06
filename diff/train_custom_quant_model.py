import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
from modeling import NetQuant,NetCustomQuant
import argparse
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            data=torch.reshape(data,(-1,784))
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_model(model_file):
    model = NetCustomQuant()
    # state_dict = torch.load(model_file)
    # model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def train( model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=torch.reshape(data,(-1,784))
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
          

def main():

    train_batch_size = 30
    eval_batch_size = 50

    transform=transforms.Compose([
        transforms.ToTensor()
        # ,
        # transforms.Normalize((0.1307,), (0.3081,))
        ])

    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    device=None

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    criterion = nn.CrossEntropyLoss()

    saved_model_dir='/home/azureuser/projects/SimpleQuantization/mnist_cnn.pt'
    qat_model = load_model(saved_model_dir)


    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    # qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # torch.quantization.prepare_qat(qat_model, inplace=True)
    print(qat_model)

    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    epochs=14
    for nepoch in range(epochs):
        train(qat_model, device, train_loader, optimizer, nepoch)
        # Check the accuracy after each epoch
        # quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
        qat_model.eval()
        # print(qat_model)
        test(qat_model, device, test_loader)
    torch.save(qat_model.state_dict(), "mnist_custom_quant.pt")

if __name__ == '__main__':
    main()
    