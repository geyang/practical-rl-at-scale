# Experimentation At Scale

## Providers-Agnostic Launch

Jaynes is a lightweight runner that can connect to local machines, cloud machins, and clusters managed by SLURM and Kubernetes.  It helps manage running a workload on multiple different sources of compute.  Check out the different Jaynes examples here: 

- [**Jaynes Starter Kit**](https://github.com/geyang/jaynes-starter-kit)

## Environment Management
In the root of the repo we include a bunch of options for lightweight environment management.  There are many lightweight options available, but they generally all are quire similar in how they work.  The main differences between them is how you install them and how they manage multiple environments on your machine.  

### Python Virtual Environment
The Python Virtual Environment is included in python3, which is convenient when moving to new machines.  To create an environment:
`python3 -m venv <env_folder>`
The environment will be stored in the `<env_folder>`, which should be added to .gitignore .dockerignore, as it will get large and should not be commited to git as it is reproducible.  To activate the environment for usage, run `source <env_folder>/bin/activate`.  To deactivate run `deactivate`.  To save the installed pip packages to a file run `pip freeze > requirements.txt`.  Lastly, there will be many frozen pip packages that are installed as dependencies of other packages used by you, so you can simplify the pip dependencies by using `pipreq` to analyze code and only create a `requirements.txt` file for python imports that are specifically used.

### Pipenv
Pipenv is a little bit heavier of a dependency manager that can be pip installed as a user program designed to maintain multiple environments on a machine and the lockfiles associated with an environment.  It is designed to use as a pip replacement, where `pipenv` is used to install packages like `pipenv install matplotlib`.  When installing packages it will automatically add them to the Pipfile to be committed to the repo for recreation.  To lock all versions of packages, `pipenv lock` will create `Pipfile.lock` that will store the unique hashes of all packages dependencies.

### Conda
Conda is a heavyweight dependency manager, made for python and R packages that comes with two major forms Anaconda and Microconda.
Anaconda is conda packaged with a huge library of dependencies pre-installed.  It is most useful for R packages and pip packages
Miniconda comes with the minimal library of dependencies
When using conda, by default you are moved into a conda environment with shell.  To switch and create new environments `conda create -n envname python=3.8`
To export an environment run `conda env export > environment.yml`.  Conda will immediately activate when logging into a shell.

## Docker
Docker is a virtual machine alternative, using containers as a paradigm.  To view more docker information, go through the Docker Readme
[**Docker**](docker/README.md)




