#!/bin/bash
echo -e "Running Docker Image... "
echo -e "args: $@\n"
docker run --mount type=bind,source=$(pwd)/output,target=/code/output -t alamp/mnist $@

