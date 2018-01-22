from gen_experiments import gen_experiments_dir, find_variables
import time
import argparse
import os

os.environ['LANG'] = 'en_CA.UTF-8'

"""
# Mount AIDATA
sshfs david.krueger@8gpu01.elementai.lan:/mnt/AIDATA /mnt/AIDATA

# Force unmount when sshfs is stalled (i.e. laptop went in sleep mode)
sudo umount -f /mnt/AIDATA

# borgy processes without the command column
watch -n 1 borgy ps --fields=id,state,name,createdOn,exitCode

# kill a set of jobs matching a pattern.
borgy ps -r  | grep '2017-10-18 02:10' | awk '{print $1}' | xargs -n1  borgy kill

# my sync script (in deep_prior directory).

#!/bin/bash
rsync --delete -avz ./ david.krueger@8gpu03.elementai.lan:/mnt/AIDATA/home/david.krueger/dev/deep-prior

# cd to most recent experiment in experiments folder
cd $(ls | tail -1)
"""

username = 'david.krueger'

exp_description = 'test_pixelSNAIL'

if __name__ == "__main__":
    params = dict(
        data_set="cifar",
        model="h12_pool2_smallkey",
        nr_logistic_mix=10,
        nr_filters=256,
        #batch_size=8,
        #init_batch_size=8,
        batch_size=8,
        init_batch_size=8,
        dropout_p=0.5,
        polyak_decay=0.9995,
        save_interval=1,
        #
        nex=100,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aidata_home', type=str, default="/mnt/AIDATA/home/" + username + "/", help='The path of your home in /mnt/AIDATA.')

    aidata_home = parser.parse_known_args()[0].aidata_home
    exp_tag = '_'.join(find_variables(params))  # extract variable names
    exp_dir = os.path.join(aidata_home, "experiments",
                           "%s_pixelSNAIL_%s_%s" % (time.strftime("%y%m%d_%H%M%S"), exp_tag, exp_description))

    #print ("yes")
    repo_path = os.path.join(aidata_home, "dev/pixelsnail-public")
    borgy_args = [
        "--image=images.borgy.elementai.lan/pixel_snail",
        "-e", "PYTHONPATH=%s" % repo_path, # -e : environment variable
        #"-e", "DATA_PATH=/mnt/AIDATA/datasets",
        "-v", "/mnt/AIDATA:/mnt/AIDATA", # make this directory visible inside the docker
        #"-v", "/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public:/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public", # make this directory visible inside the docker
        "--req-cores=12",
        "--req-gpus=4",
        "--req-ram-gbytes=40"]

    cmd = os.path.join(repo_path, "train.py")

    print (cmd)
    print (borgy_args)
    #assert False

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)
