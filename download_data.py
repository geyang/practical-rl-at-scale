#!/usr/bin/env python
import torchvision

torchvision.datasets.MNIST("data",
                           train=True,
                           download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307, ),
                                                                (0.3081, ))
                           ]))
