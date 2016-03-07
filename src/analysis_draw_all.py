import cPickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

import scipy.stats
import brewer2mpl
    
    
    
bmap = brewer2mpl.get_map('Dark2', 'qualitative', 6)
colors = bmap.mpl_colors

def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"
    
        
        
from config import config_list

# config_list = {"xp1":[
#                     "F-RmB",
#                     "F-RGB",
#                     "M-RMB",
#                     "M-P-AMB",
#                       ]}

modes = sys.argv[1:]
print "modes", modes



n_logs = 1

# xp_name = "xp1"
# d = "2016-02-28_13-54-46-TOOL2-iros_100T_14C_100K-xp1"
# xp_name = "xp_long"
# d = "2016-03-03_18-22-11-TOOL2-iros_complementary-" + xp_name
d = "2016-03-06_10-46-23-TOOL2-iros_complementary_explo_noise-xp_explo_noise"
xp_name = "xp_explo_noise"

n = 300000
p = 100
p_events = 100


sw = 10

def runningMeanFast(x, sw):
    return np.convolve(x, np.ones((sw,))/sw, mode="valid")


gss = [0, 10000, 100, 20, 10, 6, 5, 4, 3, 3]


x = np.array(np.linspace(0,n,n/p+1), dtype=int)
x_events = np.array(np.linspace(0,n,n/p_events+1), dtype=int)
x10 = np.array(np.linspace(0,n,n/10000+1), dtype=int)


# def mean_std_events(d):
#     v = np.zeros((n/p+1,len(d)))
#     for i,l in zip(range(len(d)), d.values()):
#         for li in l:
#             v[li[0]/p+1:,i] = li[1]
#     mean = np.mean(v, axis=1)
#     std = np.std(v, axis=1) / np.sqrt(len(d))
#     return mean, std
# 
# def mean_std_explo(d):
#     v = np.zeros((n/p+1,len(d)))
#     for i,l in zip(range(len(d)), d.values()):
#         for j,lj in zip(range(len(l)), l):
#             v[j+1,i] = lj
#     mean = np.mean(v, axis=1)
#     std = np.std(v, axis=1) / np.sqrt(len(d))
#     return mean, std
# 
# def mean_std_comp(d):
#     v = np.zeros((n/p+1,len(d)))
#     for i,l in zip(range(len(d)), d.values()):
#         for j,lj in zip(range(len(l)), l):
#             v[j,i] = lj
#     mean = np.mean(v, axis=1)
#     std = np.std(v, axis=1) / np.sqrt(len(d))
#     return mean, std

def mean_std_dic(d, add_0=False):
    if add_0:
        v = np.zeros((len(d),len(x_events)))
        v[:,1:] = np.array(d.values())
    else:
        v = np.array(d.values())
    mean = np.mean(v, axis=0)
    std = np.std(v, axis=0) / np.sqrt(len(d))
    return mean, std
#     

if os.environ.has_key("AVAKAS") and os.environ["AVAKAS"]:
    pref = ""
else:
    pref = "/home/sforesti/avakas"
    
log_dirs = {"xp1":pref + '/scratch/sforestier001/logs/' + d,
            "xp_bootstrap":pref + '/scratch/sforestier001/logs/' + d,
            "xp_long":pref + '/scratch/sforestier001/logs/' + d,
            "xp_snoise":pref + '/scratch/sforestier001/logs/' + d,
}

    
log_dir = log_dirs[xp_name] + '/'

events = {}
events_margins = {}
explo = {}
nbc = {}

for config in config_list[xp_name]:
    print config
    
    if "events" in modes:
        try:
            with open(log_dir + config + '/analysis_events.pickle', 'r') as f:
                ev = cPickle.load(f)
        except IOError:
            print "Warning: analysis_events not found for config", config, "file", log_dir + config + '/analysis_events.pickle'
        try:
            with open(log_dir + config + '/analysis_nbc.pickle', 'r') as f:
                nbc_ = cPickle.load(f)
                print config, nbc_
        except IOError:
            print "Warning: analysis_events not found for config", config, "file", log_dir + config + '/analysis_events.pickle'
#     try:
#         with open(log_dir + config + '/analysis_events_margins.pickle', 'r') as f:
#             ev_m = cPickle.load(f)
#     except IOError:
#         print "Warning: analysis_events not found for config", config, "file", log_dir + config + '/analysis_events.pickle'
    if "explo" in modes:
        try:
            with open(log_dir + config + '/analysis_explo.pickle', 'r') as f:
                ex = cPickle.load(f)
        except IOError:
            print "Warning: analysis_explo not found for config", config, "file", log_dir + config + '/analysis_explo.pickle'

    try:
        if "events" in modes:
            for event in ev.keys():
                if not events.has_key(event):
                    events[event] = {}
                events[event][config] = ev[event][config]
            for event in nbc_.keys():
                if not nbc.has_key(event):
                    nbc[event] = {}
                nbc[event][config] = nbc_[event]
    
#         for event in ev_m.keys():
#             if not events_margins.has_key(event):
#                 events_margins[event] = {}
#             events_margins[event][config] = ev_m[event][config]
    
        if "explo" in modes:
            for s_space in ex.keys():
                if not explo.has_key(s_space):
                    explo[s_space] = {}
                explo[s_space][config] = ex[s_space][config]
    

    except:
        pass
            




if "events" in modes:
    # EVENTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
    for event in events:
        fig1, ax = plt.subplots()
        fig1.canvas.set_window_title(event)
        for config in events[event].keys():
            #color_cycle = ax.get_
            #next_color = next(color_cycle)
            mean,std = mean_std_dic(events[event][config], add_0=True)
            ax.plot(x_events[:-(sw-1)], runningMeanFast(mean, sw), label=config)
            #ax.fill_between(x_events[:-(sw-1)], runningMeanFast(mean-std, sw), runningMeanFast(mean+std, sw), alpha=0.2, label = config, color=next_color)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
                  
        plt.savefig(log_dir + xp_name + '-event-' + event + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/" + xp_name + '-event-' + event + '.eps', format='eps', dpi=1000, bbox_inches='tight')
        #plt.close(fig1)
        
        
        
#     analysis_nbc = {}
#     for config in config_list[xp_name]:
#         analysis_nbc[config] = []
#         for trial in range(1,101):
#             analysis_nbc[config].append(nbc["hand"][config][trial] + nbc["stick"][config][trial] + nbc["object"][config][trial])

#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     fig.canvas.set_window_title("Number of abrupt behavioral changes")    
#     
#     bp = ax.boxplot([analysis_nbc[config] for config in config_list[xp_name]], notch=0, sym='', vert=1, whis=0, 
#                  positions=None, widths=0.6)
# 
#     
#     for i in range(len(bp['boxes'])):
#         box = bp['boxes'][i]
#         box.set_linewidth(0)
#         boxX = []
#         boxY = []
#         for j in range(5):
#             boxX.append(box.get_xdata()[j])
#             boxY.append(box.get_ydata()[j])
#             boxCoords = zip(boxX,boxY)
#             boxPolygon = plt.Polygon(boxCoords, facecolor = colors[i % len(colors)], linewidth=0)
#             ax.add_patch(boxPolygon)
#     
#     for i in range(0, len(bp['boxes'])):
#         bp['boxes'][i].set_color(colors[i % len(colors)])
#         # we have two whiskers!
#         bp['whiskers'][i*2].set_color(colors[i % len(colors)])
#         bp['whiskers'][i*2 + 1].set_color(colors[i % len(colors)])
#         bp['whiskers'][i*2].set_linewidth(2)
#         bp['whiskers'][i*2 + 1].set_linewidth(2)
#         # top and bottom fliers
#         bp['fliers'][i * 2].set(markerfacecolor=colors[i % len(colors)],
#                         marker='o', alpha=0.75, markersize=6,
#                         markeredgecolor='none')
#         bp['fliers'][i * 2 + 1].set(markerfacecolor=colors[i % len(colors)],
#                         marker='o', alpha=0.75, markersize=6,
#                         markeredgecolor='none')
#         bp['medians'][i].set_color('black')
#         bp['medians'][i].set_linewidth(3)
#         # and 4 caps to remove
#         for c in bp['caps']:
#             c.set_linewidth(0)
#     
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.tick_params(axis='x', direction='out')
#     ax.tick_params(axis='y', length=0)
#     
#     ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
#     ax.set_axisbelow(True)
#     
#     #ax.set_xticklabels(explo[s_space].keys())
#     plt.yticks(fontsize = 16) # work on current fig
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     
#     
#     # draw temporary red and blue lines and use them to create a legend
#     h0, = ax.plot([1,1], color=colors[0], lw=3)
#     h1, = ax.plot([1,1], color=colors[1], lw=3)
#     h2, = ax.plot([1,1], color=colors[2], lw=3)
#     h3, = ax.plot([1,1], color=colors[3], lw=3)
#     h4, = ax.plot([1,1], color=colors[4], lw=3)
#     h5, = ax.plot([1,1], color=colors[5], lw=3)
#     l = ax.legend((h0, h1, h2, h3, h4, h5),tuple(config_list[xp_name]), loc=2, fontsize=18)
#     h0.set_visible(False)
#     h1.set_visible(False)
#     h2.set_visible(False)
#     h3.set_visible(False)
#     h4.set_visible(False)
#     h5.set_visible(False)
#     
#     l.get_frame().set_linewidth(2)
#     
#     #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/nbc.pdf", format='pdf', dpi=1000, bbox_inches='tight')


 
if "legend" in modes:
    

    config_names = {"F-RmB":"F-RmB",
                    "F-RGB":"F-RGB",
                    "M-RMB":"M-RMB",
                    "M-P-AMB":"M-P-AMB",
                    "F-RGB-LWR":"F-RGB-LWR",
                    "M-RMB-LWR":"M-RMB-LWR",
                    "M-P-AMB-LWR":"M-P-AMB-LWR",
                    }
    
    # EXPLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    for s_space in explo.keys():
        for config in explo[s_space].keys():
            for trial in explo[s_space][config].keys():
                explo[s_space][config][trial] = explo[s_space][config][trial][-1]
            print explo[s_space][config]
            explo[s_space][config] = np.array(explo[s_space][config].values())
    print explo
    
    
    
    params = {
        'axes.labelsize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [2.5, 4.5]
    }
    plt.rcParams.update(params)
    
    
    # LEGEND
    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim([0, 6.])
    ax.set_ylim([0, 0.15])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    for i in range(len(config_list[xp_name])):
        ax.add_patch(plt.Rectangle((i, 0), 0.3, 0.15, facecolor = colors[i % len(colors)]))
        ax.text(i + 0.35, 0.05, "" + config_names[config_list[xp_name][i]], fontsize=17) 
        
    plt.savefig(log_dir + xp_name + '-explo-legend.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/" + xp_name + '-explo-legend.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    
    
if "explo" in modes:
    
    
    
    stats_pairs = {
                   'hand':[(1,2)],
                   'stick_1':[(1,2)],
                   'stick_2':[(1,2)],
                   'obj1':[(1,2)],
                   'obj2':[(1,2)],
                   }
    
#     config_sorted = [
#                         "RmB",
#                         "F-NN",
#                         "F-LWLR",
#                         "M-NN-RMB",
#                         "M-NN-LP-AMB",
#                         "M-LWLR-RMB",
#                         "M-LWLR-LP-AMB"
#                           ]
#     
#     stats = {}
#     
#     for config in config_sorted:
    
    stats = {}
    
    for config in config_list[xp_name]:
        
        if config[-8:] == "ENVNOISE":
            continue
        
        if not explo["obj1"].has_key(config):
            continue
        
        list1 = [explo["obj1"][config][trial][-1] for trial in explo["obj1"][config].keys()]
        list2 = [explo["obj2"][config][trial][-1] for trial in explo["obj2"][config].keys()]
        
        merged = list1 + list2
        #print merged
        q0 = min(merged)
        q100 = max(merged)
        median = np.median(merged)
        q1 = np.percentile(merged, 25)
        q3 = np.percentile(merged, 75)
        
        stats[config] = merged
        
        if explo["obj1"].has_key(config+"-ENVNOISE"):
            print "        \multirow{2}{*}{\scriptsize " + config + "} & ", "No", " & $", int(q0), "$ & $", int(q1), "$ & $", int(median), "$ & $", int(q3), "$ & $", int(q100), "$&\\\\[2pt]\\cline{2-7}"
        else:
            print "        {\scriptsize " + config + "} & ", "No", " & $", int(q0), "$ & $", int(q1), "$ & $", int(median), "$ & $", int(q3), "$ & $", int(q100), "$&\\\\[2pt]\\cline{2-7}"
        try:
            list1 = [explo["obj1"][config+"-ENVNOISE"][trial][-1] for trial in explo["obj1"][config+"-ENVNOISE"].keys()]
            list2 = [explo["obj2"][config+"-ENVNOISE"][trial][-1] for trial in explo["obj2"][config+"-ENVNOISE"].keys()]
            
            merged = list1 + list2
            #print merged
            q0 = min(merged)
            q100 = max(merged)
            median = np.median(merged)
            q1 = np.percentile(merged, 25)
            q3 = np.percentile(merged, 75)
            
            
            stats[config+"-ENVNOISE"] = merged
            
            print "        ", " & ", "Yes", " & $", int(q0), "$ & $", int(q1), "$ & $", int(median), "$ & $", int(q3), "$ & $", int(q100), "$&\\\\[2pt]"
            print "        " + "\hline"
        except:
            pass        
    with open(log_dir + 'stats_explore.pickle', 'wb') as f:
        cPickle.dump(stats, f)
    
if "stats" in modes:
    
    with open(log_dir + 'stats_explore.pickle', 'r') as f:
        stats = cPickle.load(f)
        f.close()  
    
    #print stats
    print
    print
    print "Question 1: Flat vs Modular"
    print
    
    for cdt1 in [
                "F-NN",
                "F-LWLR",
                "F-NN-ENVNOISE",
                "F-LWLR-ENVNOISE",]:
        for cdt2 in [
                    "M-NN-RMB",
                    "M-LWLR-RMB",
                    "M-NN-RMB-ENVNOISE",
                    "M-LWLR-RMB-ENVNOISE",]:
            # the stars
            z, p = scipy.stats.mannwhitneyu(stats[cdt1], stats[cdt2])
            p_value = p * 2
            print "Stat", cdt1, cdt2, z, p_value
    
    
    print
    print
    print "Question 2: Random vs Active Model Babbling"
    print
    
    for cdt1 in [
                    "M-NN-RMB",
                    "M-LWLR-RMB",
                    "M-NN-RMB-ENVNOISE",
                    "M-LWLR-RMB-ENVNOISE",]:
        for cdt2 in [
                    "M-NN-LP-AMB",
                    "M-LWLR-LP-AMB",
                    "M-NN-LP-AMB-ENVNOISE",
                    "M-LWLR-LP-AMB-ENVNOISE",]:
            # the stars
            z, p = scipy.stats.mannwhitneyu(stats[cdt1], stats[cdt2])
            p_value = p * 2
            print "Stat", cdt1, cdt2, z, p_value
    
            print "medians", np.median(stats[cdt1]), np.median(stats[cdt2])
            print
    
    
    
#        [[explo[s_space][config][trial][-1] for trial in explo[s_space][config].keys()] for config in config_list[xp_name]]
        
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         fig.canvas.set_window_title(s_space)    
#         
# #         print config_list[xp_name], explo[s_space].keys()
# #         print "explo", [explo[s_space][config] for config in config_list[xp_name]]
#         bp = ax.boxplot([[explo[s_space][config][trial][-1] for trial in explo[s_space][config].keys()] for config in config_list[xp_name]])#, notch=0, sym='', vert=1, whis=0, positions=None, widths=0.6)
#         
#         for i in range(len(bp['boxes'])):
#             box = bp['boxes'][i]
#             box.set_linewidth(0)
#             boxX = []
#             boxY = []
#             for j in range(5):
#                 boxX.append(box.get_xdata()[j])
#                 boxY.append(box.get_ydata()[j])
#                 boxCoords = zip(boxX,boxY)
#                 boxPolygon = plt.Polygon(boxCoords, facecolor = colors[i % len(colors)], linewidth=0)
#                 ax.add_patch(boxPolygon)
#         
# #         for i in range(0, len(bp['boxes'])):
# #             bp['boxes'][i].set_color(colors[i % len(colors)])
# #             # we have two whiskers!
# #             bp['whiskers'][i*2].set_color(colors[i % len(colors)])
# #             bp['whiskers'][i*2 + 1].set_color(colors[i % len(colors)])
# #             bp['whiskers'][i*2].set_linewidth(2)
# #             bp['whiskers'][i*2 + 1].set_linewidth(2)
# #             # top and bottom fliers
# #             bp['fliers'][i * 2].set(markerfacecolor=colors[i % len(colors)],
# #                             marker='o', alpha=0.75, markersize=6,
# #                             markeredgecolor='none')
# #             bp['fliers'][i * 2 + 1].set(markerfacecolor=colors[i % len(colors)],
# #                             marker='o', alpha=0.75, markersize=6,
# #                             markeredgecolor='none')
# #             bp['medians'][i].set_color('black')
# #             bp['medians'][i].set_linewidth(3)
# #             # and 4 caps to remove
# #             for c in bp['caps']:
# #                 c.set_linewidth(0)
#         
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.get_xaxis().tick_bottom()
#         ax.get_yaxis().tick_left()
#         ax.tick_params(axis='x', direction='out')
#         ax.tick_params(axis='y', length=0)
#         
#         ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
#         ax.set_axisbelow(True)
#         
#         #ax.set_xticklabels(explo[s_space].keys())
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
# #         
#         if stats_pairs.has_key(s_space):
#             for pair in stats_pairs[s_space]:
#                 # the stars
#                 print "Stat", pair
#                 data1 = [explo[s_space][config_list[xp_name][pair[0]]][trial][-1] for trial in explo[s_space][config_list[xp_name][pair[0]]].keys()]
#                 data2 = [explo[s_space][config_list[xp_name][pair[1]]][trial][-1] for trial in explo[s_space][config_list[xp_name][pair[1]]].keys()]
#                 print "data1", data1
#                 print "data2", data2
#                 z, p = scipy.stats.mannwhitneyu(data1, data2)
#                 p_value = p * 2
#                 print p
# #                 s = stars(p)
# #                 y_max = np.max((np.max(bp['boxes'][pair[0]].get_ydata()), 
# #                                np.max(bp['boxes'][pair[1]].get_ydata())))
# #                 y_min = np.min((np.min(bp['boxes'][pair[0]].get_ydata()), 
# #                                np.min(bp['boxes'][pair[1]].get_ydata())))
# #                 ax.annotate("", xy=(pair[0]+1, y_max), xycoords='data',
# #                             xytext=(pair[1]+1, y_max), textcoords='data',
# #                             arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
# #                                             connectionstyle="bar,fraction=0.2"))
# #                 ax.text((pair[0]+pair[1])/2.+1, y_max + abs(y_max - y_min)*0.05, stars(p_value),
# #                         horizontalalignment='center',
# #                         verticalalignment='center') 
# #                   
# #                 fig.subplots_adjust(left=0.2)
# #         
#         plt.show(block=False)
#         
#         plt.savefig(log_dir + xp_name + '-explo-' + s_space + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#         #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/" + xp_name + '-explo-' + s_space + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        
    
#plt.close(fig)
plt.show(block=True)

    