def estimparam_args(i, num_samples, ntrees, nthreads=1):
    return " ".join(
        [
            "-n ",
            str(num_samples),
            "--ntree " + str(ntrees),
            "--parameter t" + str(i),
            "--noob 0",
            "--chosenscen 1",
            "-j " + str(nthreads),
        ]
    )
