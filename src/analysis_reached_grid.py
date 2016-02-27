import cPickle
import numpy as np
from config import config_list
import os
import sys


config_type = sys.argv[1]

if config_type == "NONOISE":
    
    config_list = {"xp1":[
                        "RmB",
                        "F-NN",
                        "M-NN-RMB",
                        "M-NN-LP-AMB",
                        "F-LWLR",
                        "M-LWLR-RMB",
                        "M-LWLR-LP-AMB",
                          ]}
elif config_type == "ENVNOISE":
    config_list = {"xp1":[
                    "RmB-ENVNOISE",
                    "F-NN-ENVNOISE",
                    "M-NN-RMB-ENVNOISE",
                    "M-NN-LP-AMB-ENVNOISE",
                    "F-LWLR-ENVNOISE",
                    "M-LWLR-RMB-ENVNOISE",
                    "M-LWLR-LP-AMB-ENVNOISE",
                      ]}
else:
    raise NotImplementedError
    


########################################################
################### PARAMS #############################
########################################################
d = "2016-02-26_11-56-40-TOOL2-iros_100T_14C_100K-xp1"
trials = range(1,101)
n = 100000
########################################################
########################################################
########################################################





if os.environ.has_key("AVAKAS") and os.environ["AVAKAS"]:
    pref = ""
else:
    pref = "/home/sforesti/avakas"
    
log_dir = pref + '/scratch/sforestier001/logs/' + d + '/'





n_logs = 1


gss = [0, 10, 100, 20, 10, 10, 10, 5, 5, 3]


mins = np.array([-1.5] * (19 * 3))
maxs = np.array([1.5] * (19 * 3))

def compute_explo(data, mins, maxs, checkpoints=None):
    if checkpoints is None:
        checkpoints = [0, len(data)]
    n = len(mins)
    assert len(data[0]) == n
    gs = gss[n]
    epss = (maxs - mins) / gs
    grid = np.zeros([gs] * n)
    for c in range(1, len(checkpoints)):
        for i in range(checkpoints[c-1], checkpoints[c]):
            idxs = np.array((data[i] - mins) / epss, dtype=int)
            #print c, i, idxs
            idxs[idxs>=gs] = gs-1
            idxs[idxs<0] = 0
            #print idxs
            grid[tuple(idxs)] = grid[tuple(idxs)] + 1
        grid[grid > 1] = 1
        print np.sum(grid)
    return grid


def compute_list_pos(grid):
    l = []
    v = np.linspace(-1.35, 1.35, 10)
    for i1 in range(10):
        for i2 in range(10):
            for i3 in range(10):
                for i4 in range(10):
                    for i5 in range(10):
                        for i6 in range(10):
                            if grid[i1, i2, i3, i4, i5, i6]:
                                l.append([v[i1], v[i2], v[i3], v[i4], v[i5], v[i6]])
    return l


merged_grid1 = np.zeros([10] * 6)
merged_grid2 = np.zeros([10] * 6)


for config in config_list["xp1"]:

    print config
    

    for trial in trials: 
        
        print trial
        
        try:
            data = {}
            
            def get_data_topic(topic):
                data[topic] = []
                for i in range(n_logs):
                    print log_dir + config + "/log{}-".format(trial) + topic + "-{}.pickle".format(i)
                    with open(log_dir + config + "/log{}-".format(trial) + topic + "-{}.pickle".format(i), 'r') as f:
                        log = cPickle.load(f)
                        f.close()
                    data[topic] = data[topic] + log
            
            get_data_topic('agentS')
            print "data ok"
            data = np.array(data['agentS'])
        
            dims = dict(
                        obj1=range(21,27),
                        obj2=range(39,45))
            
            
            merged_grid1 = merged_grid1 + compute_explo(data[:,np.array(dims["obj1"])], mins[dims["obj1"]], maxs[dims["obj1"]])
            merged_grid2 = merged_grid2 + compute_explo(data[:,np.array(dims["obj2"])], mins[dims["obj2"]], maxs[dims["obj2"]])
        
        except IOError:
            print "Error loading files for config", config, "trial", trial
            
merged_grid1[merged_grid1 > 1] = 1
merged_grid2[merged_grid2 > 1] = 1

print np.sum(merged_grid1)
print np.sum(merged_grid2)



with open(log_dir + 'merged_grid1-{}.pickle'.format(config_type), 'wb') as f:
    cPickle.dump(merged_grid1, f)

with open(log_dir + 'merged_grid2-{}.pickle'.format(config_type), 'wb') as f:
    cPickle.dump(merged_grid2, f)

list1 = compute_list_pos(merged_grid1)
list2 = compute_list_pos(merged_grid2)

with open(log_dir + 'list1-{}.pickle'.format(config_type), 'wb') as f:
    cPickle.dump(list1, f)

with open(log_dir + 'list2-{}.pickle'.format(config_type), 'wb') as f:
    cPickle.dump(list2, f)
