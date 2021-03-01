import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# Or better: make a custom docker image with dependencies already installed

torch.backends.cudnn.enabled = False
test_env = os.environ.get("TEST_ENV")
# print("TEST_ENV=", test_env)
DIR_ROOT = 'output'  # All saved data goes in this directory
print("Starting mnist.py")


# The mnist program will save progress periodically
def save_progress(network=None, optimizer=None, losses=None):
    if network is not None:
        torch.save(network.state_dict(), os.path.join(DIR_ROOT, 'model.pth'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(DIR_ROOT, 'optimizer.pth'))
    if losses is not None:
        with open(os.path.join(DIR_ROOT, 'losses.json'), 'w') as f:
            json.dump(losses, f)


# On start, mnist will check for existing progress to continue training
def load_progress(network, optimizer, losses):
    os.makedirs(DIR_ROOT, exist_ok=True)
    try:
        network.load_state_dict(torch.load(os.path.join(DIR_ROOT, 'model.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(DIR_ROOT, 'optimizer.pth')))
        with open(os.path.join(DIR_ROOT, 'losses.json'), 'r') as f:
            data = json.load(f)
            for key in data:
                losses[key] = data[key]
    except FileNotFoundError:
        print('No progress to load')
    else:
        print('Loaded previous progress')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(network, optimizer, train_loader, batch_size_train, losses, epoch,
          save_interval):
    if losses['train']['counter']:
        last_batch = (losses['train']['counter'][-1] -
                      ((epoch - 1) * len(train_loader.dataset))) // batch_size_train + 1
    else:
        last_batch = -1

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx <= last_batch:
            continue  # Only train if not already in saved progress
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % save_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size_train, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            losses['train']['counter'].append((batch_idx * batch_size_train) +
                                              ((epoch - 1) * len(train_loader.dataset)))
            losses['train']['losses'].append(loss.item())
            save_progress(network=network, optimizer=optimizer, losses=losses)


def test(network, test_loader, losses, train_examples_seen):
    if losses['test']['counter'] and train_examples_seen <= losses['test']['counter'][-1]:
        return  # Only test if not already in saved progress

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    losses['test']['counter'].append(train_examples_seen)
    losses['test']['losses'].append(test_loss)
    save_progress(losses=losses)


def drawgraph(losses, epoch):
    fig = plt.figure()
    plt.plot(losses['train']['counter'], losses['train']['losses'], color='blue')
    plt.scatter(losses['test']['counter'], losses['test']['losses'], color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.savefig(os.path.join(DIR_ROOT, 'progress.png'))


def run(n_epochs=3,
        batch_size_train=64,
        batch_size_test=1000,
        learning_rate=0.01,
        momentum=0.5,
        save_interval=10,
        random_seed=1):
    print("mnist learning rate: ", learning_rate)
    torch.manual_seed(random_seed)

    # Initialize, and load any progress from previous runs
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    losses = {
        'train': {
            'counter': [],
            'losses': [],
        },
        'test': {
            'counter': [],
            'losses': [],
        },
    }
    load_progress(network, optimizer, losses)

    # Prepare data
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
                                               batch_size=batch_size_train,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
                                              batch_size=batch_size_test,
                                              shuffle=True)

    # Train model
    test(network, test_loader, losses, 0)
    for epoch in range(1, n_epochs + 1):
        train(network, optimizer, train_loader, batch_size_train, losses, epoch,
              save_interval)
        test(network, test_loader, losses, epoch * len(train_loader.dataset))
        drawgraph(losses, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--n-epochs', type=int, default=2)
    run(**parser.parse_args().__dict__)
    print("DONE!")
