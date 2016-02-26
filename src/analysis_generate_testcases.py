import cPickle
import numpy as np
from config import config_list
import os



########################################################
################### PARAMS #############################
########################################################
d = "2016-02-25_18-17-04-TOOL2-iros_50k_NN_LWR_sp-xp1"
n_testcases = 100
########################################################
########################################################
########################################################





if os.environ.has_key("AVAKAS") and os.environ["AVAKAS"]:
    pref = ""
else:
    pref = "/home/sforesti/avakas"
    
log_dir = pref + '/scratch/sforestier001/logs/' + d + '/'


with open(log_dir + 'list1.pickle', 'r') as f:
    list1 = cPickle.load(f)
    f.close()

with open(log_dir + 'list2.pickle', 'r') as f:
    list2 = cPickle.load(f)
    f.close()



l1 = len(list1)
l2 = len(list2)

print "Number of total explored cells in S_Magnetic1", l1
print "Number of total explored cells in S_HookLoop1", l2



v = np.linspace(-1.35, 1.35, 10)

idx1 = np.random.randint(l1, size=n_testcases)
idx2 = np.random.randint(l2, size=n_testcases)

test_cases1 = np.array(list1)[idx1]# + np.random.random_sample((n_testcases,6)) * 0.3 - 0.15
test_cases2 = np.array(list2)[idx2]# + np.random.random_sample((n_testcases,6)) * 0.3 - 0.15

print test_cases1
print test_cases2



with open(log_dir + 'test_cases1.pickle', 'wb') as f:
    cPickle.dump(test_cases1, f)

with open(log_dir + 'test_cases2.pickle', 'wb') as f:
    cPickle.dump(test_cases2, f)
