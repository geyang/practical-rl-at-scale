# Introduction

This repository is an up-to-date collection of minimal jaynes usage examples. You can mix and match configurations between these included usecases for your particular infrastructure.

## To Get Started

First let's install Jaynes! This tutorial is written w.r.t version: [v0.6.0-rc15](https://github.com/geyang/jaynes/releases/tag/v0.6.0-rc15)

```bash
pip install jaynes==0.6.0-rc15
```

I would also recommend taking a look at [params-proto](https://github.com/geyang/params_proto), which is a pythonic hyperparameter + argparsing library that makes parameter management declaritive and error-free. We use params-proto and its sweep utility, `params_proto.hyper` in our parameter sweep example. To install params-proto, run

```bash
pip install params-proto waterbear
```

## Table of Contents

For detailed documentation on each usecases, refer to the in-dept tutorial bellow:

1. [**Multiple SSH Reacheable Machines**](fast-experimentation-at-scale/03_multiple_ssh_reacheable_machines.md)
2. [**Compute at Scale with SLRUM & Jaynes**](fast-experimentation-at-scale/04_slurm_configuration.md)
3. [**Advanced Multi-mode Example**](https://github.com/geyang/practical-rl-at-scale/tree/885c3305a903870fbdd48d09444c038d5f02c605/05_muti-mode_advanced_config/README.md)
4. [**SSH Docker Configuration**](fast-experimentation-at-scale/01_ssh_docker_configuration.md)
5. [**EC2 Docker Configuration**](fast-experimentation-at-scale/02_ec2_docker_configuration.md)

In this folder, we provide a collection of example configurations. Each example sits within its own folder. To run, follow the instruction in the README in that example project.

```text
01_ssh_docker_configuration
├── README.md
├── launch_entry.py
└── .jaynes.yml
```

## Reporting Issues \(on the [Jaynes Repo/issues](https://github.com/geyang/jaynes/issues)\)

Let's collect all issues on the [main `jaynes` repo's issue page](https://github.com/geyang/jaynes/issues), so that people can search for things more easily!

## How to Debug

`Jaynes` offer a way to transparently debug the launch via `verbose` mode, where it prints out all of the local and remote script that it generates. To debug a launch script, set `verbose` to `true` either in the yaml file, or through the `jaynes.config` call. To debug in the remote host where you intend to run your job, you can often copy and paste the generated script to see the error messages.

**Debugging Steps:**

1. **Turn on verbose mode**, by setting `verbose=True` in the jaynes call

   ```python
   #! launch_entry.py
   import jaynes

   jaynes.config(verbose=True)
   ```

   or

   ```yaml
   #! .jaynes.yml
   verbose: true
   runner:
   - ....
   ```

2. **Launch**

   ```python
   #! launch_entry.py
   if __name__ == "__main__":
      jaynes.run(train_fn, *args, **kwargs)

   # if in SLURM or SSH mode:
   jaynes.listen()  # to listen to the stdout/stderr pipe-back
   ```

3. **Debug** Suppose you have an error message. You can copy and paste the script ran by `jaynes`, that is printed out in the console either locally or on the EC2 instance you just launched to debug the specifics of it.
4. **Share with Lab mates** When you are done, you can share this repo with others who use the same infrastructure, so that they can run their code there too.

## Call for Contributors

Machine Learning infrastructure is an evolving problem, and would take the rest of the community to maintain, adopt and standardize.

Below are a few areas that current stands in need to contributions:

* [**Documentation on Configuration Schema** issue \#2](https://github.com/geyang/practical-rl-at-scale/tree/885c3305a903870fbdd48d09444c038d5f02c605/issues/2/README.md)
* [**GCE Support** issue \#3](https://github.com/geyang/practical-rl-at-scale/tree/885c3305a903870fbdd48d09444c038d5f02c605/issues/3/README.md)
* [**Pure SSH Host Support** issue \#4](https://github.com/geyang/practical-rl-at-scale/tree/885c3305a903870fbdd48d09444c038d5f02c605/issues/4/README.md)
* [**SLURM SBatch Support** issue \#5](https://github.com/geyang/practical-rl-at-scale/tree/885c3305a903870fbdd48d09444c038d5f02c605/issues/5/README.md)
* [**SLURM Singularity Support** issue \#6](https://github.com/geyang/practical-rl-at-scale/tree/885c3305a903870fbdd48d09444c038d5f02c605/issues/6/README.md)

