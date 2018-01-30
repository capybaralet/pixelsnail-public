import os
import itertools
import shutil
import time

#env = " --image=images.borgy.elementai.lan/pixel_snail -e PYTHONPATH=/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public -v /mnt/AIDATA:/mnt/AIDATA --req-cores=8 --req-gpus=2 --req-ram-gbytes=24 "
env = " --image=images.borgy.elementai.lan/pixel_snail -e PYTHONPATH=/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public -v /mnt/AIDATA:/mnt/AIDATA --req-cores=4 --req-gpus=1 --req-ram-gbytes=10 "

grid = [] 
#grid.append(['--init=' + str(v) for v in [.3, 1., 3.]])
grid.append(['--n_flow_params=' + str(v) for v in [12,24,48]])
grid.append(['--nr_filters=128 --nr_resnet=4 --n_flows=2 --model=dk_DSF1',
             '--nr_filters=192 --nr_resnet=2 --n_flows=2 --model=dk_DSF1'  
            ])
print grid
grid = [" ".join(item) for item in itertools.product(*grid)]
print grid
grid += ['--nr_filters=128 --nr_resnet=4 --n_flows=2 --model=dk_IAF',
             '--nr_filters=192 --nr_resnet=2 --n_flows=2 --model=dk_IAF']
print grid
grid = [grid, ['--learning_rate=' + str(v) for v in [.0001, .0003, .001, .003]]]
print grid
exps = [" ".join(item) for item in itertools.product(*grid)]
#print (exps)
save_dir = "/mnt/AIDATA/home/david.krueger/experiments/"
save_dirs = [save_dir + '_'.join(item.split(" --")[1:]) for item in exps]
# AFTER we've made save_dirs, we'd like to add the extra settings (which are fixed!)
exps = ["/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public/train.py --max_epochs=500 --init_batch_size=16 --batch_size=16 --n_ex=16 --attn_rep=6 --n_flows=2 " + item for item in exps]


def mkdirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        time.sleep(.01)
        os.makedirs(path)

# TODO: names...
names = ['DK_MAF3' for _ in range(len(exps))]

i = 0
for name, exp, save_dir in zip(names, exps, save_dirs):
    print (i); i += 1
    launch_str = "borgy submit --name " + name + env + " -- " + exp + " --save_dir=" + save_dir + " 1>>" + save_dir + "/stdout 2>>" + save_dir + "/stderr"
    print (launch_str)
    if 1:
        mkdirs(save_dir)
        os.system(launch_str)
