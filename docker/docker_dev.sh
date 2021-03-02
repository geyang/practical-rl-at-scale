#!/bin/bash
# Runs the docker image with the current codebase
git_root=$(git rev-parse --show-toplevel)
cd $git_root

echo -e "Running Docker Image with Current Codebase... "
echo -e "args: $@\n"
# This will run the image with a live mounted codebase in the docker environment
docker run -it --entrypoint bash --mount type=bind,source=$(pwd),target=/code -t alamp/mnist

# This will run the image with a the current mounted codebase with mnist.py args
# docker run --mount type=bind,source=$(pwd),target=/code -t alamp/mnist  $@
