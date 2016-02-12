import cPickle
import matplotlib
import matplotlib.pyplot as plt
import sys

plt.switch_backend('Agg')


# ANALYSIS
def main(log_dir, config):
    
    trials = range(1, 10)
    
    for trial in trials:
        print trial
        try:
            data = {}
            
            def get_data_topic(topic):
                data[topic] = []
                print log_dir + config + "/log{}-".format(trial) + topic + "-{}.pickle".format(0)
                with open(log_dir + config + "/log{}-".format(trial) + topic + "-{}.pickle".format(0), 'r') as f:
                    log = cPickle.load(f)
                    f.close()
                data[topic] = data[topic] + log
            
            
            get_data_topic("chidren_choice_mod4")
            get_data_topic("inference_mod4")
            
            print len(data["chidren_choice_mod4"])
            
            
        
            #Object exploration
            fig, ax = plt.subplots()
            fig.canvas.set_window_title('choice_mod4')  
            
            color1=matplotlib.colors.ColorConverter().to_rgba('b', alpha=1)
            color2=matplotlib.colors.ColorConverter().to_rgba('r', alpha=1)
            
            
            list_c1x = []
            list_c1y = []
            list_c2x = []
            list_c2y = []
            
            for i,xy in zip(range(len(data["inference_mod4"])), data["inference_mod4"]):
                if data["chidren_choice_mod4"][i][1][0] == "mod3":
                    list_c1x.append(xy[0])
                    list_c1y.append(xy[1])
                else:
                    list_c2x.append(xy[0])
                    list_c2y.append(xy[1])
                
                
            ax.scatter(list_c2x, list_c2y, s=0.5, color=color2, rasterized=True)
            ax.scatter(list_c1x, list_c1y, s=0.5, color=color1, rasterized=True)
                
    #         plt.xlabel('X', fontsize = 16)
    #         plt.ylabel('Y', fontsize = 16)   
            alpha = 1
            ax.add_patch(plt.Rectangle((-1.5, -0.1), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((-1.2, 1.), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((-0.1, 1.3), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((1., 1.), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((1.3, -0.1), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((-1., -0.1), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((-0.7, 0.5), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((-0.1, 0.8), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((0.5, 0.5), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.add_patch(plt.Rectangle((0.8, -0.1), 0.2, 0.2, fc="none", alpha=alpha, lw=4))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.xlim(-1.7, 1.7)
            plt.ylim(-0.5, 1.7)
            ax.set_aspect('equal')   
            plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/test-choice-" + config + '.png', format='png', dpi=200, bbox_inches='tight')         
            #plt.savefig(log_dir + "img/" + config + '-log{}-choice_mod4.pdf'.format(trial), format='pdf', dpi=1000, bbox_inches='tight')
            #plt.show()
            plt.close(fig)
        except:
            pass    
                    
if __name__ == "__main__":
    
    log_dir = sys.argv[1]
    config = sys.argv[2]
    main(log_dir, config)