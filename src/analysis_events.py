import cPickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import brewer2mpl

bmap = brewer2mpl.get_map('Dark2', 'qualitative', 6)
colors = bmap.mpl_colors

colors_config = {"hand":colors[1],
                 "stick":colors[2],
                 "object":colors[4],
                 }


plt.switch_backend('Agg')

sw = 1

def runningMeanFast(x, sw):
    return np.convolve(x, np.ones((sw,))/sw, mode="valid")

def n_behavior_change(x, smoothing=10, th=10):
    x = runningMeanFast(x, smoothing)
    l = np.linspace(0, len(x)-1, 1+int((len(x)-1) / smoothing))
    l = np.array(l, dtype=int)
    x = np.abs(np.diff(x[l]))
    return np.sum(x >= th)

    

def main(log_dir, config):

    
    trials = range(1, 11)
    n_logs = 1
    
    n = 50000
    p = 1000
    x = np.array(range(n/p)) * p
    
    def mean_std(d):
        v = np.zeros((n/p,len(d)))
        for i,l in zip(range(len(d)), d.values()):
            for li in l:
                v[li[0]/p:,i] = li[1]
        mean = np.mean(v, axis=1)
        std = np.std(v, axis=1) / np.sqrt(len(d))
        return mean, std
    
    
    
    events = {}
    events_margins = {}
    events['hand'] = {}
    events['stick'] = {}
    events['object'] = {}
    events_margins['hand'] = {}
    events_margins['stick'] = {}
    events_margins['object'] = {}
    nbc = {}
    
    print config
    
    events['hand'][config] = {}
    events['stick'][config] = {}
    events['object'][config] = {}
    events_margins['hand'][config] = {}
    events_margins['stick'][config] = {}
    events_margins['object'][config] = {}
    nbc['hand'] = {}
    nbc['stick'] = {}
    nbc['object'] = {}
        
        
    log_p = {config:{}}
        
    for trial in trials:
        print trial
        log_p[config][trial] = {}
        try:
            
            data = {}
            
            def get_data_topic(topic):
                data[topic] = []
                for i in range(n_logs):
                    with open(log_dir + config + "/log{}-".format(trial) + topic + "-{}.pickle".format(i), 'r') as f:
                        log = cPickle.load(f)
                        f.close()
                    data[topic] = data[topic] + log
            
            get_data_topic('agentS')
            get_data_topic('babbling_module')
            
            
            
         
            babbling_module = {}
            for mid in data["babbling_module"]:
                if not (mid in babbling_module.keys()):
                    babbling_module[mid] = 1
                else:
                    babbling_module[mid] = babbling_module[mid] + 1
                     
            print config, trial, "babbling modules", babbling_module
             
             
             
            events['hand'][config][trial] = [[0,0]]
            events['stick'][config][trial] = [[0,0]]
            events['object'][config][trial] = [[0,0]]
            events_margins['hand'][config][trial] = [[0,0]]
            events_margins['stick'][config][trial] = [[0,0]]
            events_margins['object'][config][trial] = [[0,0]]
             
            def near_obj1(x, y, margin=0.3):
                return (x + 0.3)**2. + (y - 1.1)**2. < margin*margin
            
            def near_obj2(x, y, margin=0.3):
                return (x - 0.3)**2. + (y - 1.1)**2. < margin*margin
                 
            def near_one_stick(x, y, margin=0.3):
                return (x- (-0.75))**2. + (y - 0.25)**2. < margin*margin or (x- (0.75))**2. + (y - 0.25)**2. < margin*margin
                 
             
            for i,s in zip(range(1, len(data['agentS'])+1), data['agentS']):
                #print i, s
                 
                if abs(s[26] - (1.1)) > 0.0001 or abs(s[44] - (1.1)) > 0.0001 or near_obj1(s[9], s[12]) or near_obj1(s[10], s[13]) or near_obj1(s[11], s[14]) or near_obj2(s[15], s[18]) or near_obj2(s[16], s[19] or near_obj2(s[17], s[20])) :
                    events_margins['object'][config][trial].append([i, events_margins['object'][config][trial][-1][-1] + 1])
                else:
                    if abs(s[14] - (0.60355)) > 0.0001 or abs(s[20] - (0.60355)) > 0.0001 or near_one_stick(s[0], s[3]) or near_one_stick(s[1], s[4]) or near_one_stick(s[2], s[5]):
                        events_margins['stick'][config][trial].append([i, events_margins['stick'][config][trial][-1][-1] + 1])          
                    else:
                        events_margins['hand'][config][trial].append([i, events_margins['hand'][config][trial][-1][-1] + 1])
              
                     
                if abs(s[26] - (1.1)) > 0.0001 or abs(s[44] - (1.1)) > 0.0001:
                    events['object'][config][trial].append([i, events['object'][config][trial][-1][-1] + 1])
                else:
                    if abs(s[14] - (0.60355)) > 0.0001 or abs(s[20] - (0.60355)) > 0.0001:
                        events['stick'][config][trial].append([i, events['stick'][config][trial][-1][-1] + 1])          
                    else:
                        events['hand'][config][trial].append([i, events['hand'][config][trial][-1][-1] + 1])
              
            #logs_c = {}
             
            get_data_topic("interests")
            for mid in babbling_module.keys():
                log_p[config][trial][mid] = []
            for i in data["interests"]:
                t = i[0]
                interests = i[1]
                for mid in babbling_module.keys():
                    log_p[config][trial][mid].append([t,interests[mid]])
            for mid in babbling_module.keys():
                log_p[config][trial][mid] = np.array(log_p[config][trial][mid])
            
                #get_data_topic("competence_" + mid)
                #logs_c[mid] = data["competence_" + mid]
        
            if True:
                #Plot competences
#                 fig, ax = plt.subplots()
#                 fig.canvas.set_window_title('Competences')
#                 for mid in babbling_module.keys():
#                     ax.plot(np.array(logs_c[mid])[:,0], np.array(logs_c[mid])[:,1], label=mid)
#                 handles, labels = ax.get_legend_handles_labels()
#                 ax.legend(handles, labels)
#                      
#                 plt.savefig(log_dir + "img/" + config + '-log{}-competences-'.format(trial)+str(n)+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#                 plt.close(fig)
                
                # Plot progresses
                fig, ax = plt.subplots()
                fig.canvas.set_window_title('Interests')
                for mid in babbling_module.keys():
                    #print "Plot", mid, logs_p, logs_p[mid], np.array(logs_p[mid])[:,0]
                    ax.plot(log_p[config][trial][mid][x,0], log_p[config][trial][mid][x,1], label=mid, lw=2, rasterized=True)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=18)                        
                plt.tick_params(labelsize=18)                                        
#                 x_ticks = np.arange(0, 100001, 50000)                                                               
#                 ax.set_xticks(x_ticks)                                        
#                 y_ticks = np.arange(0, 0.181, 0.06)                                                               
#                 ax.set_yticks(y_ticks)       
                 
                plt.savefig(log_dir + "img/" + config + '-log{}-interests-'.format(trial)+str(n)+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
                #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/" + config + '-log{}-interests-'.format(trial)+str(n)+'.pdf', format='pdf', dpi=100, bbox_inches='tight')
                 
                plt.close(fig)
                
                # plot events against time
                fig, ax = plt.subplots(figsize=(8,5))
                fig.canvas.set_window_title('Interests')
                for event in events.keys():
                    #print "Plot", mid, logs_p, logs_p[mid], np.array(logs_p[mid])[:,0]
                    mean,_ = mean_std({"1":events[event][config][trial]})
                    print event, mean
                    res = np.append([0], np.diff(mean))
                     
                    #nbc[event][trial] = n_behavior_change(res)
                     
                    if sw > 1:
                        ax.plot(x[:-(sw-1)], runningMeanFast(res, sw), label=event, lw=3, color=colors_config[event])
                    else:
                        ax.plot(x, res, label=event, lw=3, color=colors_config[event])
                    events[event][config][trial] = res # log only result
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=22)                        
                plt.tick_params(labelsize=18)                                        
#                 x_ticks = np.arange(0, 100001, 50000)                                                               
#                 ax.set_xticks(x_ticks)                                        
#                 y_ticks = np.arange(0, 101, 20)                                                               
#                 ax.set_yticks(y_ticks)   
                plt.ylim([0,100])    
                
                #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/" + config + '-log{}-events-'.format(trial)+str(n)+'.pdf', format='pdf', dpi=100, bbox_inches='tight')               
                plt.savefig(log_dir + "img/" + config + '-log{}-events-'.format(trial)+str(n)+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')
                plt.close(fig)
                
                # plot events_margins against time
                fig, ax = plt.subplots()
                fig.canvas.set_window_title('Interests')
                for event in events_margins.keys():
                    #print "Plot", mid, logs_p, logs_p[mid], np.array(logs_p[mid])[:,0]
                    mean,_ = mean_std({"1":events_margins[event][config][trial]})
                    res = np.append([0], np.diff(mean))
                    if sw > 1:
                        ax.plot(x[:-(sw-1)], runningMeanFast(res, sw), label=event, lw=3, color=colors_config[event])
                    else:
                        ax.plot(x, res, label=event, lw=3, color=colors_config[event])
                    events_margins[event][config][trial] = res # log only result
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=22)                        
                plt.tick_params(labelsize=18)               
                           
                plt.ylim([0,100])    
                plt.savefig(log_dir + "img/" + config + '-log{}-events_margins-'.format(trial)+str(n)+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')
                plt.close(fig)




                #print nbc


                
                



        except IOError:
            print "Error loading files for config", config, "trial", trial
            
     
    with open(log_dir + config + '/analysis_events.pickle', 'wb') as f:
        cPickle.dump(log_p, f)
     
    with open(log_dir + config + '/analysis_events.pickle', 'wb') as f:
        cPickle.dump(events, f)
    with open(log_dir + config + '/analysis_events_margins.pickle', 'wb') as f:
        cPickle.dump(events_margins, f)
    with open(log_dir + config + '/analysis_nbc.pickle', 'wb') as f:
        cPickle.dump(nbc, f)
         
    #print "nbc", nbc
        
if __name__ == "__main__":
    
    log_dir = sys.argv[1]
    config = sys.argv[2]
    main(log_dir, config)
