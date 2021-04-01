def launch(lr, model_name="LeNet"):
    print(f"training model {model_name} with {lr}")
    print('...')
    print('This is working!!')


if __name__ == "__main__":
    import jaynes
    from params_proto.neo_hyper import Sweep

    from mnist import Args, run

    with Sweep(Args) as sweep, sweep.product:
        Args.lr = [1e-2, 3e-2]
        Args.seed = [100, 200, 300]

    jaynes.config()
    jaynes.run(launch, lr=1e-3)

    # this line allows you to keep the pipe open and hear back from the remote instance.
    jaynes.listen(200)
