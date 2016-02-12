import os
import time
import cPickle
import datetime

from explauto.utils import rand_bounds
from explauto.experiment import Experiment
from explauto.experiment.log import ExperimentLog


class ToolsExperiment(Experiment):
    def __init__(self, config, log = None, log_dir = None, n_trials = 1):
        
        self.config = config
        
        if hasattr(config, 'env_cls') and hasattr(config, 'env_cfg'):
            self.env = config.env_cls(**config.env_cfg)
        else:
            raise NotImplementedError
            #self.env = VrepDivaEnvironment(self.config.environment, self.config.vrep, self.config.diva)
            
        #self.ag = DmpAgent(self.config, self.env)
        self.ag = self.config.supervisor_cls(self.config, self.env, **self.config.supervisor_config)
        
        Experiment.__init__(self, self.env, self.ag)
        
            
        if log is None:
            if log_dir is None:
                self.log_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             '../../logs/') 
                                + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
                                + '-' 
                                + config.tag)
                
            else:
                self.log_dir = log_dir + config.tag
            try: # muliprocess collisions
                if not os.path.exists(self.log_dir):
                    os.mkdir(self.log_dir)
            except OSError:
                pass
            config.log_dir = self.log_dir
        else:
            assert log_dir is not None
            self.log_dir = log_dir
            self.log = log 
            
        self.ag.subscribe('agentM', self)
        self.ag.subscribe('agentS', self)
        self.ag.subscribe('babbling_module', self)
        self.ag.subscribe_topics_mod(['interest', 'competence', 'chidren_choice'], self)
            
        self.n_trials = n_trials
        self.trial = 0
        
        
        
    def reset(self):
        self.ag = self.config.supervisor_cls(self.config, self.env, **self.config.supervisor_config)
        self.log = ExperimentLog(self.ag.conf, self.ag.expl_dims, self.ag.inf_dims)
        self.log.log_dir = self.log_dir
        self.evaluate_at(self.config.eval_at, self.testcases)
        self.ag.subscribe('agentM', self)
        self.ag.subscribe('agentS', self)
        self.ag.subscribe('babbling_module', self)
        self.ag.subscribe_topics_mod(['interest', 'competence', 'chidren_choice'], self)
        
    @classmethod
    def from_log(cls, config, log_dir, from_log_dir, from_log_trial, n_logs=1, forward_im=False):

        log = ExperimentLog()
        
        keys=['agentM', 'agentS', 'babbling_module']
        for mid in config.modules.keys():
            keys.append('im_update_' + mid)
        
        for key in keys:
            for n in range(n_logs):
                filename = from_log_dir + 'log{}-'.format(from_log_trial) + key + '-{}.pickle'.format(n)
                with open(filename, 'r') as f:
                    log_key_n = cPickle.load(f)
                log._logs[key] = log._logs[key] + log_key_n
                 
        experiment = cls(config=config, log=log, log_dir=log_dir)
        experiment.ag.fast_forward(log, forward_im=forward_im)
        experiment.log.purge()
        
        return experiment
        
    
    def motor_babbling(self, n, range_div = 1.):
        #print 'Motor babbling : ', n, "points..."

        for i in range(n):
            m = self.ag.motor_babbling()
            m_mov = self.ag.motor_primitive(m)
            s_mov = self.env.update(m_mov, log=False)
#             print "MOTOR BABBLING"
#             print "m", m
#             print 's_h', s_mov[0:9]
#             print 's_t1', s_mov[9:15]
#             print 's_t2', s_mov[15:21]
#             print 's_o1', s_mov[21:27]
#             print 's_o2', s_mov[27:33]
#             print 's_o3', s_mov[33:39]
#             print 's_o4', s_mov[39:45]
#             print 's_o5', s_mov[45:51]
#             print 's_o6', s_mov[51:57]
            s = self.ag.sensory_primitive(s_mov)
            #print 'Babbling iteration', i, ': m =', m, 's =', s
            self.ag.update_sensorimotor_models(m, s)

        self._update_logs()

    def rest_trial(self):
        m = self.ag.rest_params()
        m_mov = self.ag.motor_primitive(m)
        s_mov = self.env.update(m_mov, log=False)
        s = self.ag.sensory_primitive(s_mov)
        self.ag.update_sensorimotor_models(m, s)
        self._update_logs()
        print "Rest trial", "m:", m, "m_mov:", m_mov, "s:", s
        

    def start(self):
        for i in range(1,self.n_trials+1):
            self.trial = i
            self.start_trial()
            if i < self.n_trials:
                self.reset()

    def start_trial(self):

        print '[' + self.config.tag + '] ' + 'Starting trial', self.trial 

        #self.ag.subscribe('movement', self)
        # xp.evaluate_at(eval_at, tc)


        self.log.bootstrap_conf = {'n': self.config.bootstrap, 
                                   'bootstap_range_div': self.config.bootstrap_range_div}
        if self.config.init_rest_trial:
            self.rest_trial()
        if self.config.bootstrap > 0:
            self.motor_babbling(self.config.bootstrap, self.config.bootstrap_range_div)
        
        #print "Running", self.config.iter, "iterations..."
        log_each = self.config.log_each
        
        for i in range((self.config.iter) / log_each):
            t_start = time.time()
            self.run(log_each)
            print '[' + self.config.tag + '] ' + 'Run up to ' + str((i + 1) * log_each)
            print "Time for", log_each, "iterations :", time.time() - t_start
            self.save_logs()
            

    def save_logs(self):
        #print 'Log directory : ', self.log_dir
        #self.log.config = copy.copy(self.config)
        #self.log.config.env_config = None
        
        for key in self.log._logs.keys():
            filename = self.log_dir + '/log{}-'.format(self.trial) + key + '-{}.pickle'.format(self.log.n_purge)
            with open(filename, 'wb') as f:
                cPickle.dump(self.log._logs[key], f)
            f.close()
        self.log.purge()
            
