#!/bin/bash
echo -e "Running Docker Image Bash... "

# This docker run command overrides the entrypoint to bash
# When the container runs, bash will open in an interactive
# shell, at the working directory /code
docker run -it --entrypoint bash alamp/mnist:latest

