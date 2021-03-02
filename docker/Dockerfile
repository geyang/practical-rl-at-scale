FROM ubuntu:20.04
# Starting with Ubuntu as the base image.
# 
# Other common images in ML include:
# nvidia/cuda
# tensorflow/tensorflow
# 

# Upgrades ubuntu packages and installs python3
# Run in "noninteractive" to prevent prompts
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip

# Add other environment dependencies to install here

# Sets the docker user to be the same as your local user
# (makes it easier to not have to change permisisons of results)
ARG USER_ID
ARG GROUP_ID

# Change working directory to /code
RUN mkdir -p /code

# Adds the new user and chowns working directory
RUN addgroup --gid $GROUP_ID user

RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN chown -R user:user /code

USER user


WORKDIR /code

# Add allows the docker image to hash the file directly and 
# only repeat the below steps if the file changes
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Copy will recursively copy the whole directory
# Copy does not get cached (as the copy size can be large),
# so it should be one of your last steps
COPY . .


# Entrypoint is what will run by default when using docker run or docker exec
# If you do `docker run <image> < arg(s) ...>`, the arg(s) will be postpended to the entrypoint
# 
# To run bash and inspect the image:
# `docker run -it --entrypoint bash <image>`
ENTRYPOINT ["python3", "mnist.py"]
