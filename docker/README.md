# Using Docker for Reproducibility

https://docs.docker.com/get-started/

### Building an Image

To build a docker image, you have to write the instructions to create the image from another base image.  This instruction file is called a Dockerfile.  Docker will read through the Dockerfile and change the state of the image at every step to produce a final image at the end.

There is an exmple Dockerfile included for the mnist sample.  `docker_build.sh` will build the Docker image with the correct context (the root of the git repo).  By including a dataset folder or results folder in the context, Docker build will make copies of all files past the root, so we supply a .dockerignore which will ignore `output` `data` `venv` and other directories similar to a gitignore.

When building the image, we use an ARG (`USER_ID` and `GROUP_ID`) that is supplied to match the local machine's id.  This allows the files created within the image to match the host machine's id credentials and then there will be no need to chown the output of an image.  

The entrypoint of the image is `python3 mnist.py`, which means that any extra args provided after a `docker run` or `docker exec` will be postpended.  We can then run the docker image in one step such as `docker run -t <image tag> --n-epochs 10`

### Running an Image (and dealing with your Codebase)

Included in this repo are a couple of different ways to run the image, which will allow you to do different things.  The image itself will contain a pegged version of the codebase, from whenever you built it.  This means that if you use the docker image as a source of your codebase, you will need to continually update the docker registry with new images or update the inside of the image by git pulling new updates.  

I believe that this complicates the development process, but is good for sweeping hyperparameters, when you have a stable codebase and just need to run it on many computers.  Then you can uniquely tag the docker image, to keep it for reproducibilty in the future.  Using the docker image to contain the codebase, I find often slows down the development process because there is always the confusion as to whether the codebase in this image is up to date and credential issues in pulling updates or continually needing to rebuild images.

To deal with this codebase updating issue during development, I find that it is often best to just mount your codebase directly into the docker image from the host machine.  This allows for a couple of benefits:
- IDE usage on the host machine
- One source of codebase
- No need to recredential git within a docker image to pull updates
- Running on random other machines can be credential-less as you can `rsync` the codebase over, then run the docker image by pulling from a docker registry.

In the sample, I have included a couple of different ways to run the docker image with mounts. 
`docker_run_bash.sh` - Runs the pegged image with bash entrypoint
usage: > `./docker_run_bash.sh` 
    user@3eda7e8f3827:/code$ 
`docker_run.sh` - Runs a mounted codebase image with mnist.py arguments
usage: > `./docker_run.sh --n-epochs 5 --learning-rate 0.14` 
`docker_dev.sh` - Runs a mounted codebase image with bash entrypoing
usage: > `./docker_dev.sh` 
    user@3eda7e8f3827:/code$ 

