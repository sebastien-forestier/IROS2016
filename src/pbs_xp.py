import os
import sys
import subprocess
import time
import datetime
import cPickle
import json
import base64
import numpy as np

from config import config_list, configs

path = '/home/sforestier/software/IROS2016/src/'

start_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

pool_name = sys.argv[1]


def write_pbs(config_name, trial, log_dir, time=12):
    pbs =   """
#!/bin/sh

#PBS -l walltime={}:00:00
#PBS -l nodes=1:ppn=1
#PBS -N {}-{}
#PBS -o {}logs/log-{}-{}.output
#PBS -e {}logs/log-{}-{}.error

cd {}
time python run.py {} {} {}

""".format(time, config_name, trial, log_dir, config_name, trial, log_dir, config_name, trial, path, log_dir, config_name, trial)
    filename = '{}-{}.pbs'.format(config_name, trial)
    with open(log_dir + "pbs/" + filename, 'wb') as f:
        f.write(pbs)


xp_list = [
#           "xp1", 
#             "xp2", 
#             "xp_snoise",
#             "xp_long",
#             "xp_bootstrap",
            "xp_explo_noise",
]

log_dir = '/scratch/sforestier001/logs/' + start_date + '-TOOL2-' + pool_name + '-'



n_iter = 10
iter_list = range(1,n_iter + 1) 


for xp_name in xp_list:
    os.mkdir(log_dir + xp_name + "/")
    os.mkdir(log_dir + xp_name + "/" + "pbs")
    os.mkdir(log_dir + xp_name + "/" + "img")
    os.mkdir(log_dir + xp_name + "/" + "logs")
    os.mkdir(log_dir + xp_name + "/" + "configs")
    
    
    
    
    
    
    

# 
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist() # or map(int, obj)
#         return json.JSONEncoder.default(self, obj)
#     
#     
#     
# for xp_name in xp_list:
#     for config_name in config_list[xp_name]: 
#         filename = 'config-{}.pickle'.format(config_name)
#         with open(log_dir + xp_name + "/" + "configs/" + filename, 'wb') as f:
#             cPickle.dump(configs[config_name], f)
#         filename = 'config-{}.json'.format(config_name)
#         with open(log_dir + xp_name + "/" + "configs/" + filename, 'wb') as f:
#             json.dump(configs[config_name].__dict__, f, cls=NumpyEncoder, default=lambda obj:[])
        
        
for xp_name in xp_list:
    for trial in iter_list:
        for config_name in config_list[xp_name]:
            print xp_name, config_name, trial
            if xp_name == "xp_long":
                write_pbs(config_name, trial, log_dir + xp_name + "/", time=48)
            else:
                write_pbs(config_name, trial, log_dir + xp_name + "/")   
            filename = '{}-{}.pbs'.format(config_name, trial)
            
            print "Run qsub", config_name, trial
            print "qsub " + log_dir + xp_name + "/pbs/" + filename
            process = subprocess.Popen("qsub " + log_dir + xp_name + "/pbs/" + filename, shell=True, stdout=subprocess.PIPE)
            time.sleep(0.2)
