import os
from config import config_list
from multiprocessing import Process
from analysis_explo import main as main_explo
from analysis_events import main as main_events

# 
# config_list = {"xp1":[
#                     "F-RmB",
#                     "F-RGB",
#                     "M-RMB",
#                     "M-P-AMB",
#                     "F-RGB-LWR",
#                     "M-RMB-LWR",
#                     "M-P-AMB-LWR",
#                       ]}

d = "2016-02-24_21-11-14-TOOL2-iros_nn100000_sgsnn-xp1"


if os.environ.has_key("AVAKAS") and os.environ["AVAKAS"]:
    pref = ""
else:
    pref = "/home/sforesti/avakas"
    
log_dirs = {"xp1":pref + '/scratch/sforestier001/logs/' + d
}



processes = []

log_dir = log_dirs["xp1"] + "/"
#log_dir = '/home/sforesti/scm/Flowers/explaupoppydiva/scripts/cogsci2016/test_dmp2/'


try:
    os.mkdir(log_dir + "img")
except:
    pass

for config in config_list["xp1"]:        
    processes.append(Process(target = main_explo, args=(log_dir, config)))
    processes.append(Process(target = main_events, args=(log_dir, config)))

print "Begin Explo and Events analysis"
for p in processes:
    p.start()
for p in processes:
    p.join()
print "Finished Explo and Events analysis"
