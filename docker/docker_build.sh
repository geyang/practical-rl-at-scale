#!/bin/bash
git_root=$(git rev-parse --show-toplevel)
cd $git_root

echo -e "Building Docker Image from: " $(pwd)
echo -e "with command: docker build -f docker/Dockerfile -t alamp/mnist .\n"
docker build -f docker/Dockerfile \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t alamp/mnist .

