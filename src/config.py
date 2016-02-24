import numpy as np

from explauto.utils.config import make_configuration
from explauto.sensorimotor_model.non_parametric import NonParametric
from explauto.interest_model.random import MiscRandomInterest, competence_dist, competence_exp
from supervisor import Supervisor
from environment import IROS2016Environment
from explauto.interest_model.competences import competence_dist


class Config(object):
    def __init__(self, 
                 name=None, 
                 hierarchy_type=0, 
                 babbling_name="goal", 
                 supervisor_name="interest", 
                 supervisor_explo="motor", 
                 supervisor_n_explo_points = 0,
                 supervisor_ccm="competence", 
                 supervisor_ccl="local", 
                 sm_model='NN',
                 tdd=False,
                 ns=False,
                 perturbation=None,
                 allow_split_mod1=False,
                 from_log=None,
                 iterations=None):
              
        ################################### EXPERIMENT CONFIG ###################################
    
        self.name = name or 'Experiment'
        self.init_rest_trial = False
        self.bootstrap = 0
        self.bootstrap_range_div = 1.
        self.iter = iterations or 50
        self.log_each = self.iter #must be <= iter
        self.eval_at = []
        self.n_eval = 0
        self.eval_modes = []
        
        self.gui = False
        
        self.hierarchy_type = hierarchy_type
        self.babbling_name = babbling_name
        if self.babbling_name == "goal":
            self.motor_babbling_n_iter = 10
        else:
            self.motor_babbling_n_iter = self.iter
            
        self.from_log = from_log
        
        ################################### AGENT CONFIG ###################################
        
        self.n_dyn_motors = 4
        self.n_dmps = self.n_dyn_motors
        self.dmp_use_initial = False
        self.dmp_use_goal = True
        self.n_bfs = 2
        self.n_static_motor = 0
        self.rest_position = np.zeros(self.n_dmps + self.n_static_motor)
        
        self.motor_n_dims = self.n_dyn_motors * self.n_bfs + self.n_static_motor
        if self.dmp_use_initial: 
            self.motor_n_dims = self.motor_n_dims +  self.n_dmps
        if self.dmp_use_goal:
            self.motor_n_dims = self.motor_n_dims +  self.n_dmps
             
        
        self.move_steps = 50
        self.motor_dims = range(self.motor_n_dims)
        self.s_n_dims = 31 * 3
        
        self.sensori_dims = range(self.motor_n_dims, self.motor_n_dims + self.s_n_dims)
        self.used_dims = self.motor_n_dims + self.s_n_dims
        
        self.choose_children_local = (supervisor_ccl == 'local')
        
        self.ims = {'miscRandom_local': (MiscRandomInterest, {
                                  'competence_measure': competence_dist,
                                  #'competence_measure': lambda target, reached, dist_max:competence_exp(target, reached, dist_min=0.01, dist_max=dist_max, power=20.),
                                   'win_size': 100,
                                   'competence_mode': 'knn',
                                   'k': 20,
                                   'progress_mode': 'local'}),
            }
        
        self.sms = {
            'NN': (NonParametric, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.01}),
            'LWLR-BFGS': (NonParametric, {'fwd': 'LWLR', 'k':1, 'sigma':0., 'sigma_explo_ratio':0.01, 'inv': 'L-BFGS-B', 'maxfun':200, 'ftol':0, 'gtol':0}),
            'LWLR-CMAES': (NonParametric, {'fwd': 'LWLR', 'k':1, 'sigma':0.01, 'inv': 'CMAES', 'cmaes_sigma':0.05, 'sigma_explo_ratio':0.01, 'maxfevals':20}),
        }
          
        self.sm_model = sm_model
        self.im_model = 'miscRandom_local'
        self.im_name = self.im_model
        
        sm = self.sm_model
        im_mode = 'sg'
        self.std_range = [-1.,1.]
        
        
        m = self.motor_dims
        s = self.sensori_dims
        
        self.operators = ["par"]
        
        if self.hierarchy_type == 0:
            self.m_spaces = dict(m=m)
            self.s_spaces = dict(s=s)
            
            self.modules = dict(mod1 = dict(m = m,
                                          s = s,     
                                          m_list = [m],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                )
        elif self.hierarchy_type == 1:
            self.m_spaces = dict(m_arm=range(12))
            self.s_spaces = dict(s_h=range(self.motor_n_dims + 0, self.motor_n_dims + 9),
                                 s_t1=range(self.motor_n_dims + 9, self.motor_n_dims + 15),
                                 s_t2=range(self.motor_n_dims + 15, self.motor_n_dims + 21),
                                 s_o1=range(self.motor_n_dims + 21, self.motor_n_dims + 27),
                                 s_o2=range(self.motor_n_dims + 27, self.motor_n_dims + 33),
                                 s_o3=range(self.motor_n_dims + 33, self.motor_n_dims + 39),
                                 s_o4=range(self.motor_n_dims + 39, self.motor_n_dims + 45),
                                 s_o5=range(self.motor_n_dims + 45, self.motor_n_dims + 51),
                                 s_o6=range(self.motor_n_dims + 51, self.motor_n_dims + 57),
                                 s_o7=range(self.motor_n_dims + 57, self.motor_n_dims + 63),
                                 s_o8=range(self.motor_n_dims + 63, self.motor_n_dims + 69),
                                 s_o9=range(self.motor_n_dims + 69, self.motor_n_dims + 75),
                                 s_o10=range(self.motor_n_dims + 75, self.motor_n_dims + 81),
                                 s_o11=range(self.motor_n_dims + 81, self.motor_n_dims + 87),
                                 s_o12=range(self.motor_n_dims + 87, self.motor_n_dims + 93))

            self.modules = dict(mod1 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_h"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod2 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_t1"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod3 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_t2"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                
                                mod4 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o1"],     
                                          m_list = [self.m_spaces["m_arm"]],        
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod5 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o2"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod6 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o3"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod7 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o4"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod8 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o5"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod9 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o6"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod10 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o7"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod11 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o8"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod12 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o9"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod13 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o10"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod14 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o11"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                
                                mod15 = dict(m = self.m_spaces["m_arm"],
                                          s = self.s_spaces["s_o12"],     
                                          m_list = [self.m_spaces["m_arm"]],      
                                          operator = "par",                            
                                          babbling_name = "goal",
                                          sm_name = sm,
                                          im_name = self.im_name,
                                          im_mode = im_mode,
                                          from_log = None,
                                          motor_babbling_n_iter=self.motor_babbling_n_iter),
                                )
        else:
            raise NotImplementedError
        
        
        self.supervisor_name = supervisor_name
        self.supervisor_explo = supervisor_explo
        self.supervisor_n_explo_points = supervisor_n_explo_points
        self.supervisor_ccm = supervisor_ccm
        self.supervisor_ccl = supervisor_ccl
        
        if self.supervisor_name == "random":
            self.supervisor_cls = Supervisor
            self.supervisor_config = dict(choice="random",
                                          llb=False,
                                          explo=self.supervisor_explo,
                                          n_explo_points=self.supervisor_n_explo_points,
                                          choose_children_mode=self.supervisor_ccm,
                                          choose_children_local=self.supervisor_ccl,
                                          allow_split_mod1=allow_split_mod1)
        elif self.supervisor_name == "interest":
            self.supervisor_cls = Supervisor
            self.supervisor_config = dict(choice="prop",
                                          llb=False,
                                          explo=self.supervisor_explo,
                                          n_explo_points=self.supervisor_n_explo_points,
                                          choose_children_mode=self.supervisor_ccm,
                                          choose_children_local=self.supervisor_ccl,
                                          allow_split_mod1=allow_split_mod1)
        elif self.supervisor_name == "interest-pmin":
            self.supervisor_cls = Supervisor
            self.supervisor_config = dict(choice="prop-min",
                                          llb=False,
                                          explo=self.supervisor_explo,
                                          n_explo_points=self.supervisor_n_explo_points,
                                          choose_children_mode=self.supervisor_ccm,
                                          choose_children_local=self.supervisor_ccl,
                                          allow_split_mod1=allow_split_mod1)
        elif self.supervisor_name == "interest_greedy":
            self.supervisor_cls = Supervisor
            self.supervisor_config = dict(choice="greedy",
                                          llb=False,
                                          explo=self.supervisor_explo,
                                          n_explo_points=self.supervisor_n_explo_points,
                                          choose_children_mode=self.supervisor_ccm,
                                          choose_children_local=self.supervisor_ccl,
                                          allow_split_mod1=allow_split_mod1)
        elif self.supervisor_name == "interest_bias":
            self.supervisor_cls = Supervisor
            self.supervisor_config = dict(choice="prop",
                                          llb=True,
                                          explo=self.supervisor_explo,
                                          n_explo_points=self.supervisor_n_explo_points,
                                          choose_children_mode=self.supervisor_ccm,
                                          choose_children_local=self.supervisor_ccl,
                                          allow_split_mod1=allow_split_mod1)
        else:
            raise NotImplementedError
        
        
        self.eval_dims = s[-4:-2]
        self.eval_explo_dims = s[-4:-2]
        
        self.eval_range = np.array([[-1.],
                                 [1.]])
        self.eval_explo_eps = 0.02
        self.eval_explo_comp_eps = 0.02
        
        
        ################################### Env CONFIG ###################################
                
        self.max_param = 500. # max DMP weight 
        self.max_params = self.max_param * np.ones((self.n_dmps * self.n_bfs,))  

        if self.dmp_use_initial: 
            self.max_params = np.append([1]*self.n_dmps, self.max_params)
        if self.dmp_use_goal:
            self.max_params = np.append(self.max_params, [1]*self.n_dmps)

        self.env_cls = IROS2016Environment
        self.env_cfg = dict(move_steps=self.move_steps, 
                            max_params=self.max_params,
                            perturbation=perturbation,
                            gui=self.gui)
        
        self.rest_position = [0.] * self.motor_n_dims
        
        self.m_mins = [-1.] * (self.n_dyn_motors * (self.n_bfs+1))
        self.m_maxs = [1.] * (self.n_dyn_motors * (self.n_bfs+1))
        
        self.s_mins = [-1.5] * 31 * 3
        self.s_maxs = [1.5] * 31 * 3
        
        
        ################################### Process CONFIG ###################################
        
        self.agent = make_configuration(self.m_mins, 
                                        self.m_maxs, 
                                        self.s_mins, 
                                        self.s_maxs)
        self.tag = self.name
        self.log_dir = ''#determined later
    
     

configs = {}

#################### EXPERIMENT  ####################

iterations = 100000

config_list = {"xp1":["F-RmB",
                      "F-RGB",
                      "F-RGB-SPLIT",
                      "M-RMB",
                      "M-P-AMB",
#                     "M-PMIN-AMB",
#                     "M-GR-AMB",
                    "F-RGB-LWR",
                    "M-RMB-LWR",
                    "M-P-AMB-LWR",
#                     "M-GR-AMB-LWR",
                      ]}

config = Config(name="F-RmB", hierarchy_type=0, babbling_name="motor", iterations=iterations)
configs[config.name] = config

config = Config(name="F-RGB", hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="F-RGB-SPLIT", hierarchy_type=0, allow_split_mod1=True, iterations=iterations)
configs[config.name] = config

config = Config(name="M-P-AMB", hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="M-PMIN-AMB", hierarchy_type=1, supervisor_name="interest-pmin", iterations=iterations)
configs[config.name] = config

config = Config(name="M-RMB", hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-GR-AMB", hierarchy_type=1, supervisor_name="interest_greedy", iterations=iterations)
configs[config.name] = config


config = Config(name="F-RGB-LWR", sm_model='LWLR-BFGS', hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="M-P-AMB-LWR", sm_model='LWLR-BFGS', hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="M-RMB-LWR", sm_model='LWLR-BFGS', hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-GR-AMB-LWR", sm_model='LWLR-BFGS', hierarchy_type=1, supervisor_name="interest_greedy", iterations=iterations)
configs[config.name] = config
