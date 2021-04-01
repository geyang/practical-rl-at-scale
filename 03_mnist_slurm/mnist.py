import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# Or better: make a custom docker image with dependencies already installed
from tqdm import tqdm

torch.backends.cudnn.enabled = False
test_env = os.environ.get("TEST_ENV")
# print("TEST_ENV=", test_env)
DIR_ROOT = 'output'  # All saved data goes in this directory
print("Starting mnist.py")

# if __name__ == '__main__':
#     import argparse
#
#     parser.add_argument('--learning-rate', type=float, default=0.01)
#     parser.add_argument('--random-seed', type=int, default=1)
#     parser.add_argument('--n-epochs', type=int, default=2)
#     run(**parser.parse_args().__dict__)
#     print("DONE!")

from params_proto.neo_proto import ParamsProto


class Args(ParamsProto):
    lr = 0.01
    seed = 100
    n_epochs = 20
    batch_size = 32
    momentum = 0.5

    # for checkpointing
    checkpoint_interval = 10


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
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


def train(network, optimizer, train_loader):
    from ml_logger import logger
    network.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        logger.log(loss=loss.cpu().item())


def evaluation(network, test_loader):
    from ml_logger import logger

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
    logger.log(test_loss=test_loss)


def run(args=None):
    from ml_logger import logger

    Args._update(args)
    logger.log_params(Args=vars(Args))

    torch.manual_seed(Args.seed)

    # Initialize, and load any progress from previous runs
    network = LeNet()
    optimizer = optim.SGD(network.parameters(), lr=Args.lr, momentum=Args.momentum)
    # Prepare data
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        "../data", train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])), batch_size=Args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        "../data", train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=Args.batch_size, shuffle=True)

    # Train model
    for epoch in range(1, Args.n_epochs + 1):
        logger.log(epoch=epoch)
        train(network, optimizer, train_loader)
        evaluation(network, test_loader)
        logger.flush()
        # drawgraph(losses, epoch)


if __name__ == '__main__':
    from ml_logger import logger

    logger.configure(prefix=f"../output/{logger.now('%m-%d/%H.%M.%S')}")
    run()
