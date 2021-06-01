from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from modeling import Net,NetCustomQuant,NetQuant

def train( model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=torch.reshape(data,(-1,784))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data=torch.reshape(data,(-1,784))
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_train_setting():
    import argparse
    parser = argparse.ArgumentParser(description='simple quant')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001,help='learning rate')
    parser.add_argument('--epochs', type=int, default=14,help='learning rate')
    parser.add_argument('--use_cuda', type=bool, default=True,help='learning rate')
    parser.add_argument('--quant_type', type=str, default='custom',help='[origin,custom,official]')
    args = parser.parse_args()
    args.use_cuda =  torch.cuda.is_available()
    torch.manual_seed(1)
    

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if args.use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    return train_kwargs, test_kwargs,args

def load_data(train_kwargs, test_kwargs):
    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                       transform=transform)


    train_loader = torch.utils.data.DataLoader(train_data,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    return train_loader, test_loader


def main():
    # Training settings
    train_kwargs, test_kwargs,args= load_train_setting()
    
    train_loader, test_loader = load_data(train_kwargs, test_kwargs)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    if args.quant_type=='origin':
        print('origin quant mode')
        model = Net().to(device)
    elif args.quant_type=='custom':
        print('custom quant mode')
        model = NetCustomQuant().to(device)
    elif args.quant_type=='official':
        print('official quant mode')
        device=torch.device('cpu')
        model= NetQuant().to(device)


    if args.quant_type=='official':

        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)

        # QAT takes time and one needs to train over a few epochs.
        # Train and check accuracy after each epoch
        # device = torch.device("cuda" if use_cuda else "cpu")
        for nepoch in range(args.epochs):
            train(model, device, train_loader, optimizer, nepoch)
            # Check the accuracy after each epoch
            quantized_model = torch.quantization.convert(model.eval(), inplace=False)
            quantized_model.eval()
            test(quantized_model, device, test_loader)
        
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()


if __name__ == '__main__':
    main()
    