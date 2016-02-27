import cPickle
import matplotlib.pyplot as plt
import numpy as np
import sys

from experiment import ToolsExperiment
from config import configs
from explauto.experiment.log import ExperimentLog
from explauto.utils import rand_bounds
from evaluation import Evaluation
import os



########################################################
################### PARAMS #############################
########################################################
d = "2016-02-26_11-56-40-TOOL2-iros_100T_14C_100K-xp1"
n_checkpoints = 4
n_testcases = 1000
########################################################
########################################################
########################################################



plt.switch_backend('Agg')


n_logs = 1

if os.environ.has_key("AVAKAS") and os.environ["AVAKAS"]:
    pref = ""
else:
    pref = "/home/sforesti/avakas"
    
log_dirs = {"xp1":pref + '/scratch/sforestier001/logs/' + d
}
log_dir = log_dirs["xp1"] + "/"



n = 100000
p = 25000

gui = False
xp = None




with open(log_dir + 'testcases1.pickle', 'r') as f:
    testcases1 = cPickle.load(f)
    f.close()

with open(log_dir + 'testcases2.pickle', 'r') as f:
    testcases2 = cPickle.load(f)
    f.close()




testcases = {
             'obj1':(range(21, 27), testcases1[:n_testcases,:]),
              
             'obj2':(range(39, 45), testcases2[:n_testcases,:]),
              }
        

x = np.array(np.linspace(0,n,n/p+1), dtype=int)



def main(explo_config_name, trial):

    
    def mean_std(d):
        v = np.zeros((n/p,len(d)))
        for i,l in zip(range(len(d)), d.values()):
            for j,lj in zip(range(len(l)), l):
                v[j,i] = lj
        mean = np.mean(v, axis=1)
        std = np.std(v, axis=1) / np.sqrt(len(d))
        return mean, std
    
    
    
    comp = {}
    logs = {}
    
    keys = ["agentM", "agentS", "babbling_module"]
    
    
    def eval_comp(config_name, trial, i, log_i):
        global xp, testcases
        config = configs[config_name]
        for key in log_i._logs.keys():
            print key, len(log_i._logs[key])
        if i == 0:
            config.gui = gui
            config.env_cfg['gui'] = gui
            xp = ToolsExperiment(config, log_dir=log_dir + config_name + '/')
        else:
            xp.ag.fast_forward(log_i)
        xp.ag.eval_mode()
        
        evaluation = Evaluation(xp.log, xp.ag, xp.env, testcases, modes=["inverse"])
        result = evaluation.evaluate_comp()
        return result

        
    print "explo_config_name", explo_config_name
    
    for s_space in testcases.keys():
        comp[s_space] = {}
        
    for s_space in testcases.keys():
        comp[s_space][explo_config_name] = {}
        
    logs[explo_config_name] = {}
            
    print "trial", trial
    
    logs[explo_config_name][trial] = {}
    log = ExperimentLog(None, None, None)
    for key in keys:
        for i in range(n_logs):
            filename = log_dir + explo_config_name + '/log{}-'.format(trial) + key + '-{}.pickle'.format(i)
            with open(filename, 'r') as f:
                log_key = cPickle.load(f)
            log._logs[key] = log._logs[key] + log_key
        print key, len(log._logs[key])
    
    for s_space in testcases.keys():
        comp[s_space][explo_config_name][trial] = {}
        
    for regression_config_name in ["M-NN-RMB", "M-LWLR-RMB"]: 
        print "regression_config_name", regression_config_name
        
        for s_space in testcases.keys():
            comp[s_space][explo_config_name][trial][regression_config_name] = []
            
        for i in range(n_checkpoints+1):
            print "checkpoint", i
            
            log_i = ExperimentLog(None, None, None)
            for key in ["agentM", "agentS"]:
                if i > 0:
                    log_i._logs[key] = log._logs[key][(i-1) * n / n_checkpoints: (i) * n / n_checkpoints]
                else:
                    log_i._logs[key] = []
                print regression_config_name, trial, key, i, n, n_checkpoints, [i * n / n_checkpoints, (i+1) * n / n_checkpoints], len(log_i._logs[key])
                
                
            errors = eval_comp(regression_config_name, trial, i, log_i)[0]
            for s_space in testcases.keys():
                comp[s_space][explo_config_name][trial][regression_config_name] += [errors[s_space]]
        logs[explo_config_name][trial][regression_config_name] = xp.log._logs
        
        if True:
            fig, ax = plt.subplots()
            fig.canvas.set_window_title('Competence')
            for s_space in testcases.keys():
                print x, comp[s_space][explo_config_name][trial][regression_config_name], np.median(comp[s_space][explo_config_name][trial][regression_config_name])
                ax.plot(x, np.median(comp[s_space][explo_config_name][trial][regression_config_name]), label=s_space)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
                 
            plt.savefig(log_dir + "img/" + explo_config_name + '-log-{}-{}-comp.png'.format(regression_config_name, trial))
            plt.close(fig)
            
    
    
    with open(log_dir + explo_config_name + '/analysis_comp_eval-{}.pickle'.format(trial), 'wb') as f:
        cPickle.dump(comp, f)
        
    with open(log_dir + explo_config_name + '/analysis_comp_logs-{}.pickle'.format(trial), 'wb') as f:
        cPickle.dump(logs, f)

if __name__ == "__main__":

    explo_config_name = sys.argv[1]
    trial = sys.argv[2]
    main(explo_config_name, trial)
