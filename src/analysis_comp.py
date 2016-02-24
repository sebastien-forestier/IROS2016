import cPickle
import matplotlib.pyplot as plt
import numpy as np
import sys

from experiment import ToolsExperiment
from config import config_list
from explauto.experiment.log import ExperimentLog
from explauto.utils import rand_bounds

plt.switch_backend('Agg')


log_dir = sys.argv[1]



n_logs = 1

trials = range(1, 11)
n_checkpoints = 10

n_testcases = 10



testcases1 = {
             'obj1':(range(14+21, 14+27), rand_bounds(np.array([[-0.3, -0.3, -0.3, 1.1, 1.1, -1.5], [-0.3, -0.3, -0.3, 1.1, 1.1, 1.5]]), 
                                                     n_testcases)),
              
             'obj2':(range(14+39, 14+45), rand_bounds(np.array([[-1.5, -0.5], [2., 2]]), 
                                                     n_testcases)),
              }
        
n = 50000
p = 5000

gui = False

x = np.array(np.linspace(0,n,n/p+1), dtype=int)



def mean_std(d):
    v = np.zeros((n/p,len(d)))
    for i,l in zip(range(len(d)), d.values()):
        for j,lj in zip(range(len(l)), l):
            v[j,i] = lj
    mean = np.mean(v, axis=1)
    std = np.std(v, axis=1) / np.sqrt(len(d))
    return mean, std


xp = None

comp = {}
logs = {}

keys = ["agentM", "agentS", "babbling_module"]


def eval_comp(config_name, trial, i, log_i):
    global xp
    config = configs[config_name]
    for key in log_i._logs.keys():
        print key, len(log_i._logs[key])
    if i == 0:
        config.gui = gui
        config.env_cfg['gui'] = gui
        xp = VrepDivaExperiment(config, log_dir=log_dir + config_name + '/')
        xp.ag.fast_forward(log_i, forward_im=config.babbling_name == 'goal')
    else:
        xp.ag.fast_forward(log_i, forward_im=config.babbling_name == 'goal')
    xp.ag.eval_mode()
    
    logs['testcases'] = testcases1
    evaluation = Evaluation(xp.log, xp.ag, xp.env, logs['testcases'], modes=["comp"])
    result = evaluation.evaluate_comp()
    return result



for config in config_list["xp1"]: 
    
    testcases = {'obj':0}
        
    for s_space in testcases.keys():
        comp[s_space] = {}
        
    for s_space in testcases.keys():
        comp[s_space][config_name] = {}
        
    logs[config_name] = {}
    
    if configs[config_name].babbling_name == 'goal':
        for mid in configs[config_name].modules.keys():
            keys.append('im_update_' + mid)
            
    for trial in trials:
        print trial
        
        try:
            log = ExperimentLog()
            for key in keys:
                for i in range(n_logs):
                    filename = log_dir + config_name + '/log{}-'.format(trial) + key + '-{}.pickle'.format(i)
                    with open(filename, 'r') as f:
                        log_key = cPickle.load(f)
                    log._logs[key] = log._logs[key] + log_key
                print key, len(log._logs[key])
            
            for s_space in testcases.keys():
                comp[s_space][config_name][trial] = [0]
                
            n_im_mid = {}
            for mid in configs[config_name].modules.keys():
                n_im_mid[mid] = 0
                
            for i in range(n_checkpoints):
                print "checkpoint", i
                
                log_i = ExperimentLog()
                for key in ["agentM", "agentS"]:
                    log_i._logs[key] = log._logs[key][i * n / n_checkpoints: (i+1) * n / n_checkpoints]
                    #print config_name, trial, key, i, n, n_checkpoints, [i * n / n_checkpoints, (i+1) * n / n_checkpoints], len(log_i._logs[key])
                    
                for mid in configs[config_name].modules.keys():
                    log_i._logs['im_update_' + mid] = []
                #print log._logs["babbling_module"]
                for mid_babbling in log._logs["babbling_module"][i * n / n_checkpoints: (i+1) * n / n_checkpoints]:
                    #print mid_babbling, n_im_mid[mid_babbling]
                    try:
                        log_i._logs['im_update_' + mid_babbling].append(log._logs['im_update_' + mid_babbling][n_im_mid[mid_babbling]])
                        n_im_mid[mid_babbling] += 1
                    except IndexError:
                        pass
                    
                errors = eval_comp(config_name, trial, i, log_i)[0]
                for s_space in testcases.keys():
                    comp[s_space][config_name][trial] += [np.mean(errors[s_space])]
            logs[config_name][trial] = xp.log._logs
            
            if True:
                fig, ax = plt.subplots()
                fig.canvas.set_window_title('Competence')
                for s_space in testcases.keys():
                    #print x, comp[s_space][config_name][trial]
                    ax.plot(x, comp[s_space][config_name][trial], label=s_space)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                     
                plt.savefig(log_dir + config_name + '/log{}-comp.png'.format(trial))
                plt.close(fig)
                
        except IOError:
            print "File not found for trial", trial
    
    
    with open(log_dir + config_name + '/analysis_comp_eval.pickle', 'wb') as f:
        cPickle.dump(comp, f)
        
    with open(log_dir + config_name + '/analysis_comp_logs.pickle', 'wb') as f:
        cPickle.dump(logs, f)


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
