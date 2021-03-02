USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)
export USER_ID
export GROUP_ID
PWD := $(shell pwd)
export PWD


# These are copied from docker/*.sh
docker-build:
	echo -e "Building Docker Image from: " $(PWD)
	echo -e "with command: docker build -f docker/Dockerfile -t alamp/mnist .\n"
	docker build -f docker/Dockerfile \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		-t alamp/mnist .

docker-run: 
	# To run the docker image with parameters use:
	# make docker-run ARGS="--n-epochs 1 --learning-rate 0.14"
	docker run --mount type=bind,source=$(PWD)/output,target=/code/output -t alamp/mnist $(ARGS)

docker-run-bash:
	echo -e "Running Docker Image Bash... "
	# This docker run command overrides the entrypoint to bash
	# When the container runs, bash will open in an interactive
	# shell, at the working directory /code
	docker run -it --entrypoint bash alamp/mnist:latest
docker-dev:
	echo -e "Running Docker Image with Current Codebase... "
	# This will run the image with a live mounted codebase in the docker environment
	docker run -it --entrypoint bash --mount type=bind,source=$(PWD),target=/code -t alamp/mnist
