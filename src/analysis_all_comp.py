import os
from config import config_list
from multiprocessing import Process
from analysis_comp import main



d = "2016-02-24_15-32-03-TOOL2-iros_small-xp1"

if os.environ.has_key("AVAKAS") and os.environ["AVAKAS"]:
    pref = ""
else:
    pref = "/home/sforesti/avakas"
    
log_dirs = {"xp1":pref + '/scratch/sforestier001/logs/' + d
}


processes = []

log_dir = log_dirs["xp1"] + "/"


for trial in range(10):        
    processes.append(Process(target = main, args=(log_dir, trial)))

print "Begin Comp analysis"
for p in processes:
    p.start()
for p in processes:
    p.join()
print "Finished Comp analysis"
