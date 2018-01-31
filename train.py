#!/usr/bin/env python3 
"""
Trains a SNAIL generative model on CIFAR-10 or Tiny ImageNet data.
Supports using multiple GPUs and machines (the latter using MPI).

modified by DK to do MAF (WIP)
TODO:
    Gaussian output (cost)
    init

"""

import numpy as np
def spherical_Gaussian_NLL(x):
    return .5 * (x**2 + np.log(2*np.pi))




def main(args):
    import os
    import sys
    import time
    import json
    from mpi4py import MPI
    import numpy as np
    import tensorflow as tf
    from tqdm import trange

    import pixel_cnn_pp.nn as nn
    import pixel_cnn_pp.plotting as plotting
    from pixel_cnn_pp import model as pxpp_models
    import data.cifar10_data as cifar10_data
    import data.imagenet_data as imagenet_data

    import tf_utils as tfu

    # TODO: from data.cifar10_data import ALPHA
    ALPHA = .05

    comm = MPI.COMM_WORLD
    num_tasks, task_id = comm.Get_size(), comm.Get_rank()
    save_dir = args.save_dir

    if task_id == 0:
        os.makedirs(save_dir)
        f_log = open(os.path.join(save_dir, 'print.log'), 'w')

    def lprint(*a, **kw):
        if task_id == 0:
            print(*a, **kw)
            print(*a, **kw, file=f_log)

    #lprint (num_tasks)
    lprint('input args:\n', json.dumps(vars(args), indent=4,
                                                                         separators=(',', ':')))    # pretty print args
    # -----------------------------------------------------------------------------
    # fix random seed for reproducibility
    rng = np.random.RandomState(args.seed + task_id)
    tf.set_random_seed(args.seed + task_id)
    
    # for MAF, we model pixels as real values, and must dequantize the discrete pixel values!
    if args.model == 'dk_CNN':
        dequantize = False
    else:
        dequantize = True

    n_ex = args.n_ex / (args.nr_gpu * args.batch_size) * (args.nr_gpu * args.batch_size) 

    # initialize data loaders for train/test splits
    if args.data_set == 'imagenet' and args.class_conditional:
        raise("We currently don't have labels for the small imagenet data set")
    DataLoader = {'cifar': cifar10_data.DataLoader,
                                'imagenet': imagenet_data.DataLoader}[args.data_set]
    if args.use_valid:
        train_data = DataLoader(args.data_dir, 'train_', args.batch_size,
                                                        rng=rng, shuffle=True, return_labels=args.class_conditional,
                                                        dequantize=dequantize,
                                                        n_ex=n_ex)
        test_data = DataLoader(args.data_dir, 'valid_', args.batch_size,
                                                     shuffle=False, return_labels=args.class_conditional,
                                                     dequantize=dequantize,
                                                     #n_ex=int(.25 * args.n_ex))
                                                     n_ex=n_ex)
    else:
        train_data = DataLoader(args.data_dir, 'train', args.batch_size,
                                                        rng=rng, shuffle=True, return_labels=args.class_conditional,
                                                        dequantize=dequantize,
                                                        n_ex=n_ex)
        test_data = DataLoader(args.data_dir, 'test', args.batch_size,
                                                     shuffle=False, return_labels=args.class_conditional,
                                                     dequantize=dequantize,
                                                     n_ex=n_ex)
    obs_shape = train_data.get_observation_size()    # e.g. a tuple (32,32,3)
    assert len(obs_shape) == 3, 'assumed right now'

    if args.nr_gpu is None:
        #assert False
        from tensorflow.python.client import device_lib
        args.nr_gpu = len([d for d in device_lib.list_local_devices()
                                             if d.device_type == 'GPU'])

    # data place holders
    x_init = tf.placeholder(tf.float32,
                                                    shape=(args.init_batch_size,) + obs_shape)
    xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
                for _ in range(args.nr_gpu)]

    def _get_batch(is_training):
        if is_training:
            x = train_data.__next__(args.batch_size)
        else:
            x = test_data.__next__(args.batch_size)
        #x = np.cast[np.float32]((x - 127.5) / 127.5)
        return dict(x=x)

    #trx = _get_batch(1)
    #tex = _get_batch(0)
    #import pdb; pdb.set_trace()

    batch_def = dict(x=tfu.vdef(args.batch_size, obs_shape))
    qr = tfu.Struct(
            train=tfu.PyfuncRunner(batch_def, 64, 8, True,
                                                         _get_batch, is_training=True),
            test=tfu.PyfuncRunner(batch_def, 64, 8, True,
                                                        _get_batch, is_training=False),
    )
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr.train)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr.test)

    if args.nr_gpu is None:
        assert False
        from tensorflow.python.client import device_lib
        args.nr_gpu = len([d for d in device_lib.list_local_devices()
                                             if d.device_type == 'GPU'])

    sess = tfu.Session(allow_soft_placement=True).__enter__()

    # if the model is class-conditional we'll set up label placeholders +
    # one-hot encodings 'h' to condition on
    if args.class_conditional:
        raise NotImplementedError
        num_labels = train_data.get_num_labels()
        _init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
        h_init = tf.one_hot(y_init, num_labels)
        y_sample = np.split(
                np.mod(np.arange(args.batch_size), num_labels), args.nr_gpu)
        h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels)
                                for i in range(args.nr_gpu)]
        ys = [tf.placeholder(tf.int32, shape=(args.batch_size,))
                    for i in range(args.nr_gpu)]
        hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
    else:
        h_init = None
        h_sample = [None] * args.nr_gpu
        hs = h_sample


    # create the model
    # TODO: don't use kwargs!?!?
    if args.model == "dk_CNN":
        model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
                                 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity,
                                 'attn_rep': args.attn_rep}
    else:
        assert args.n_flow_params is not None
        assert args.n_flows is not None
        model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
                                 'n_flow_params': args.n_flow_params, 'resnet_nonlinearity': args.resnet_nonlinearity,
                                 'n_flows': args.n_flows,
                                 'attn_rep': args.attn_rep}
    model = tf.make_template('model', getattr(pxpp_models, args.model + "_spec"))

    # TODO: init
    # run once for data dependent initialization of parameters
    with tf.device('/gpu:0'):
        _ = model(x_init, h_init, init=True,
                                        dropout_p=args.dropout_p, **model_opt)

    # keep track of moving average
    all_params = tf.trainable_variables()
    lprint('# of Parameters', sum(np.prod(p.get_shape().as_list())
                                                                for p in all_params))
    ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
    maintain_averages_op = tf.group(ema.apply(all_params))

    loss_gen, loss_gen_test, grads = [], [], []
    extra_bpd_term, extra_bpd_term_test = [], [] 
    # TODO: remove these two lines (etc.)
    loss_prior, loss_j = [], []
    loss_prior_test, loss_j_test = [], []
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):

            x = qr.train.batch().x
            #tr_x = x
            gen_par = model(x, hs[i], ema=None,
                                            dropout_p=args.dropout_p, **model_opt)
            if args.model == 'dk_CNN':
                if isinstance(gen_par, tuple) and len(gen_par) == 3:
                    loss_gen.append(nn.discretized_mix_logistic_loss_per_chn(x, *gen_par))
                else:
                    loss_gen.append(nn.discretized_mix_logistic_loss(x, gen_par))
            else:
                u, log_dets = gen_par
                loss_gen.append(tf.reduce_sum(spherical_Gaussian_NLL(u)) - tf.reduce_sum(log_dets))
                loss_prior.append(tf.reduce_sum(spherical_Gaussian_NLL(u)))
                loss_j.append(-tf.reduce_sum(log_dets))
                lprint (u.shape.as_list(), log_dets.shape.as_list())
                # eqn 27 MAF (does not depend on model parameters!!)
                extra_bpd_term.append(-np.log2(1 - 2*ALPHA) + 8 + (tf.reduce_mean(tf.log(tf.nn.sigmoid(x))) + tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(x)))) / np.log(2.))
            #import pdb; pdb.set_trace()
            grads.append(tf.gradients(loss_gen[i], all_params))

            x = qr.test.batch().x
            #te_x = x
            gen_par = model(x, hs[i], ema=ema, dropout_p=0., **model_opt)
            if args.model == 'dk_CNN':
                if isinstance(gen_par, tuple) and len(gen_par) == 3:
                    loss_gen_test.append(
                            nn.discretized_mix_logistic_loss_per_chn(x, *gen_par))
                else:
                    loss_gen_test.append(nn.discretized_mix_logistic_loss(x, gen_par))
            else:
                u, log_dets = gen_par
                loss_gen_test.append(tf.reduce_sum(spherical_Gaussian_NLL(u)) - tf.reduce_sum(log_dets))
                loss_prior_test.append(tf.reduce_sum(spherical_Gaussian_NLL(u)))
                loss_j_test.append(-tf.reduce_sum(log_dets))
                # eqn 27 MAF
                extra_bpd_term_test.append(-np.log2(1 - 2*ALPHA) + 8 + (tf.reduce_mean(tf.log(tf.nn.sigmoid(x))) + tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(x)))) / np.log(2.))


    # add losses and gradients together and get training updates
    tf_lr = tf.placeholder(tf.float32, shape=[])
    with tf.device('/gpu:0'):
        for i in range(1, args.nr_gpu): # add all the losses and gradients from GPUs1+ to those from GPU0
            loss_gen[0] += loss_gen[i]
            loss_gen_test[0] += loss_gen_test[i]
            if args.model != 'dk_CNN': # TODO: do we need to do it that way?  can't we just do a mean?
                extra_bpd_term[0] += extra_bpd_term[i]
                extra_bpd_term_test[0] += extra_bpd_term_test[i]
            for j in range(len(grads[0])):
                grads[0][j] += grads[i][j]

    if num_tasks > 1:
        assert False
        lprint('creating mpi optimizer')
        # If we have multiple mpi processes, average across them.
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g in grads[0]], axis=0)
        shapes = [g.shape.as_list() for g in grads[0]]
        sizes = [int(np.prod(s)) for s in shapes]
        buf = np.zeros(sum(sizes), np.float32)

        def _gather_grads(my_flat_grad):
            comm.Allreduce(my_flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_gather_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        grads[0] = [tf.reshape(g, v.shape) for g, v in zip(avg_grads, grads[0])]

    # training op
    optimizer = tf.group(nn.adam_updates(
            all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995, eps=1e-6), maintain_averages_op)

    # convert loss to bits/dim
    total_gpus = sum(comm.allgather(args.nr_gpu))
    lprint('using %d gpus across %d machines' % (total_gpus, num_tasks))
    norm_const = np.log(2.) * np.prod(obs_shape) * args.batch_size
    norm_const *= total_gpus / num_tasks
    bits_per_dim = loss_gen[0] / norm_const
    bits_per_dim_test = loss_gen_test[0] / norm_const
    # extra terms of eqn 27 (MAF) (FIXME: extra_bpd_term is a LIST!??!?)
    if args.model != 'dk_CNN':
        bits_per_dim += (extra_bpd_term[0] / total_gpus)
        bits_per_dim_test += (extra_bpd_term_test[0] / total_gpus)

    bits_per_dim = tf.check_numerics(bits_per_dim, 'train loss is nan')
    bits_per_dim_test = tf.check_numerics(bits_per_dim_test, 'test loss is nan')

    # TODO: MAF sampling
    if args.model == 'dk_CNN':
        new_x_gen = []
        for i in range(args.nr_gpu):
            with tf.device('/gpu:%d' % i):
                gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0, **model_opt)
                new_x_gen.append(
                        nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix))

        def sample_from_model(sess, n_samples=args.nr_gpu * args.batch_size):
            sample_x = np.zeros((0,) + obs_shape, dtype=np.float32)
            while len(sample_x) < n_samples:
                x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
                                 for i in range(args.nr_gpu)]
                for yi in range(obs_shape[0]):
                    for xi in range(obs_shape[1]):
                        new_x_gen_np = sess.run(new_x_gen,
                                                                        {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
                        for i in range(args.nr_gpu):
                            x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]

                sample_x = np.concatenate([sample_x] + x_gen, axis=0)

            img_tile = plotting.img_tile(
                    sample_x[:int(np.floor(np.sqrt(n_samples))**2)],
                    aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(
                    os.path.join(save_dir, '%s_samples.png' % args.data_set))
            np.save(os.path.join(save_dir, '%s_samples.npy' % args.data_set), sample_x)
            plotting.plt.close('all')

    # init & save
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)

    # turn numpy inputs into feed_dict for use with tensorflow
    # DK - this is only used for initialization! (I think)
    def make_feed_dict(data, init=False):
        if type(data) is tuple:
            x, y = data
        else:
            x = data
            y = None
        # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
        #x = np.cast[np.float32]((x - 127.5) / 127.5)
        if init:
            feed_dict = {x_init: x}
            if y is not None:
                feed_dict.update({y_init: y})
        else:
            x = np.split(x, args.nr_gpu)
            feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
            if y is not None:
                y = np.split(y, args.nr_gpu)
                feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
        return feed_dict
    
    if 0:
        for var in tf.global_variables():
            lprint(var)

    # //////////// perform training //////////////
    lprint('dataset size: %d' % len(train_data.data))
    train_bpd = []
    test_bpd = []
    times = []
    lr = args.learning_rate

    # manually retrieve exactly init_batch_size examples
    feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True)
    train_data.reset()    # rewind the iterator back to 0 to do one full epoch
    lprint('initializing the model...')

    sess.run(initializer, feed_dict)
    if args.load_params:
        assert False
        # ckpt_file = save_dir + '/params_' + args.data_set + '.ckpt'
        ckpt_file = args.load_params
        lprint('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    # Sync params before starting.
    my_vals = sess.run(all_params)
    vals = [np.zeros_like(v) for v in my_vals]
    [comm.Allreduce(mv, v, op=MPI.SUM) for mv, v in zip(my_vals, vals)]
    assign_ops = [var.assign(val / num_tasks)
                                for var, val in zip(all_params, vals)]
    sess.run(assign_ops)

    coord = tfu.start_queue_runners(sess)
    batch_size = args.batch_size * total_gpus
    iters_per_train_epoch = len(train_data.data) // batch_size
    iters_per_test_epoch = len(test_data.data) // batch_size

    lprint('starting training')
    for epoch in range(args.max_epochs):
        begin = time.time()
        # train for one epoch
        train_losses = []

        ti = trange(iters_per_train_epoch)
        for itr in ti:
            if coord.should_stop():
                tfu.stop_queue_runners(coord)

            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            l, _ = sess.run([bits_per_dim, optimizer], {tf_lr: lr})
            train_losses.append(l)
            ti.set_postfix(loss=l, lr=lr)

        train_loss_gen = np.mean(train_losses)
        train_bpd.append(train_loss_gen)

        # compute likelihood over test data
        test_losses = []
        for itr in trange(iters_per_test_epoch):
            if coord.should_stop():
                tfu.stop_queue_runners(coord)

            l = sess.run(bits_per_dim_test)
            test_losses.append(l)

        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        times.append(time.time() - begin)
        stats = dict(epoch=epoch, time=times[-1], lr=lr,
                                 train_bpd=train_loss_gen,
                                 test_bpd=test_loss_gen)
        all_stats = comm.gather(stats)
        if task_id == 0:
            lprint('-' * 16)
            for k in stats:
                lprint('%s:\t%s' % (k, np.mean([s[k] for s in all_stats])))
            if epoch % args.save_interval == 0:
                path = os.path.join(save_dir, str(epoch))
                os.makedirs(path, exist_ok=True)
                saver.save(sess, os.path.join(path, 'params_%s.ckpt' % args.data_set))
            # save performance and time info every epoch
            np.savetxt(os.path.join(save_dir, args.model + '_test_bpd.txt'), test_bpd, fmt='%1.3f')
            np.savetxt(os.path.join(save_dir, args.model + '_train_bpd.txt'), train_bpd, fmt='%1.3f')
            np.savetxt(os.path.join(save_dir, args.model + '_times.txt'), times, fmt='%1.2f')

        lprint("loss components")
        lprint(sess.run(loss_prior))
        lprint(sess.run(loss_j))
        lprint(sess.run(loss_prior_test))
        lprint(sess.run(loss_j_test))

        # TODO: sampling from MAF
        if args.model=="dk_CNN":
            sample_from_model(sess)


if __name__ == '__main__':
    import argparse
    import datetime
    import dateutil.tz
    import functools
    import os.path as osp

    parser = argparse.ArgumentParser()

    # DK (TODO: restore defaults!)
    parser.add_argument('--attn_rep', type=int, default=12)
    parser.add_argument('--n_ex', type=int, default=50000)
    parser.add_argument('--n_flows', type=int, default=2)
    parser.add_argument('--n_flow_params', type=int, default=2)
    parser.add_argument('--use_valid', type=int, default=1)
    # DK (modified)
    parser.add_argument('--model', type=str, default="dk_IAF",
            choices=["dk_CNN", "dk_IAF", "dk_DSF1", "dk_IAF_MADE", "dk_DSF1_MADE"], 
                                            help='name of the model')

    # data I/O
    parser.add_argument('-i', '--data_dir', type=str, default="/mnt/AIDATA/home/david.krueger/data/",
                                            help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='./save',
                                            help='Location for parameter checkpoints and samples')
    parser.add_argument('-d', '--data_set', type=str, default='cifar',
                                            help='Can be either cifar|imagenet')
    parser.add_argument('-t', '--save_interval', type=int, default=20,
                                            help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str,
                                            help='Restore training from previous model checkpoint?')

    # model
    #parser.add_argument('--model', type=str, default="dk_CNN", # alias for "h12_noup_smallkey"
    #                                        help='name of the model')
    parser.add_argument('-q', '--nr_resnet', type=int, default=4,
                                            help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=256,
                                            help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                                            help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='elu',
                                            help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
    parser.add_argument('-c', '--class_conditional', dest='class_conditional',
                                            action='store_true', help='Condition generative model on labels?')
    # optimization
    parser.add_argument('-l', '--learning_rate', type=float,
                                            default=1e-3, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999998,
                                            help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=3, # TODO: restore
                                            help='Batch size during training per GPU')
    parser.add_argument('-a', '--init_batch_size', type=int, default=3, # TODO: restore
                                            help='How much data to use for data-dependent initialization.')
    parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                                            help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
    parser.add_argument('-x', '--max_epochs', type=int, default=800,
                                            help='How many epochs to run in total?')
    parser.add_argument('-g', '--nr_gpu', type=int, default=None,
                                            help='How many GPUs to distribute the training across? Defaults to all.')

    # evaluation
    parser.add_argument('--polyak_decay', type=float, default=0.9995,
                                            help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
    # reproducibility
    parser.add_argument('-s', '--seed', type=int, default=42,
                                            help='Random seed to use')
    FLAGS = parser.parse_args()

    timestamp = datetime.datetime.now(
            dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    logdir = 'pixelsnail_%s_%s' % (FLAGS.data_set, timestamp)

    FLAGS.save_dir = osp.join(FLAGS.save_dir, logdir)
    main(FLAGS)
