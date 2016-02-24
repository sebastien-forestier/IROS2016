import cPickle
import matplotlib.pyplot as plt
import numpy as np
import sys

from experiment import ToolsExperiment
from config import config_list, configs
from explauto.experiment.log import ExperimentLog
from explauto.utils import rand_bounds
from evaluation import Evaluation

plt.switch_backend('Agg')


n_logs = 1

n_checkpoints = 5




n = 10000
p = 2000

gui = False
xp = None

n_testcases = 10

testcases = {
             'obj1':(range(21, 27), rand_bounds(np.array([[-0.3, -0.3, -1.5, 1.1, -1.5, -1.5], [-0.3, -0.3, 1.5, 1.1, 1.5, 1.5]]), 
                                                     n_testcases)),
              
             'obj2':(range(39, 45), rand_bounds(np.array([[0.3, 0.3, -1.5, 1.1, -1.5, -1.5], [0.3, 0.3, 1.5, 1.1, 1.5, 1.5]]), 
                                                     n_testcases)),
              }
        
print "testcases", testcases



def main(log_dir, trial):

    
    
    x = np.array(np.linspace(p,n,n/p+1), dtype=int)
    
    
    
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
            xp.ag.fast_forward(log_i)
        else:
            xp.ag.fast_forward(log_i)
        xp.ag.eval_mode()
        
        evaluation = Evaluation(xp.log, xp.ag, xp.env, testcases, modes=["inverse"])
        result = evaluation.evaluate_comp()
        return result
    
    
    
    for explo_config_name in ["M-P-AMB", "M-P-AMB-LWR"]: 
            
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
            
        for regression_config_name in ["M-P-AMB", "M-P-AMB-LWR"]: 
            print "regression_config_name", regression_config_name
            
            for s_space in testcases.keys():
                comp[s_space][explo_config_name][trial][regression_config_name] = []
                
            for i in range(n_checkpoints):
                print "checkpoint", i
                
                log_i = ExperimentLog(None, None, None)
                for key in ["agentM", "agentS"]:
                    log_i._logs[key] = log._logs[key][i * n / n_checkpoints: (i+1) * n / n_checkpoints]
                    #print config_name, trial, key, i, n, n_checkpoints, [i * n / n_checkpoints, (i+1) * n / n_checkpoints], len(log_i._logs[key])
                    
                    
                errors = eval_comp(regression_config_name, trial, i, log_i)[0]
                for s_space in testcases.keys():
                    comp[s_space][explo_config_name][trial][regression_config_name] += [np.mean(errors[s_space])]
            logs[explo_config_name][trial][regression_config_name] = xp.log._logs
            
            if True:
                fig, ax = plt.subplots()
                fig.canvas.set_window_title('Competence')
                for s_space in testcases.keys():
                    #print x, comp[s_space][config_name][trial]
                    ax.plot(x, comp[s_space][explo_config_name][trial][regression_config_name], label=s_space)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                     
                plt.savefig(log_dir + explo_config_name + '/log-{}-{}-comp.png'.format(regression_config_name, trial))
                plt.close(fig)
                
        
        
        with open(log_dir + explo_config_name + '/analysis_comp_eval-{}.pickle'.format(trial), 'wb') as f:
            cPickle.dump(comp, f)
            
        with open(log_dir + explo_config_name + '/analysis_comp_logs-{}.pickle'.format(trial), 'wb') as f:
            cPickle.dump(logs, f)

if __name__ == "__main__":

    log_dir = sys.argv[1]
    trial = sys.argv[2]
    main(log_dir, trial)

# 
# for s_space in testcases.keys():
#     fig1, ax = plt.subplots()
#     fig1.canvas.set_window_title(s_space)
#     for config in comp[s_space].keys():
#         color_cycle = ax._get_lines.color_cycle
#         next_color = next(color_cycle)
#         mean,std = mean_std(comp[s_space][config])
#         ax.plot(x, mean, label=config, color=next_color)
#         ax.fill_between(x, mean-std, mean+std, alpha=0.2, label = config, color=next_color)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles, labels, loc='upper left')
#     plt.xlim(xmin=0)
#     plt.ylim(ymin=0)
#              
#     plt.savefig(log_dir + '/comp-' + s_space + '.png')
