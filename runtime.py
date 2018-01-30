import numpy as np

def runtime(bs, epochs, gpus, iters_per_s):
    n_iters = epochs * 50000 / bs
    n_seconds = n_iters / iters_per_s / gpus
    #print (n_seconds, "seconds")
    print (np.round(n_seconds / 3600., 1), "hours")
    print (np.round(n_seconds / 3600. / 24., 1), "days")

