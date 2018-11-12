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
                 im_model='miscRandom_local',
                 im_mode='sg',
                 tdd=False,
                 ns=False,
                 envnoise=0,
                 perturbation=None,
                 allow_split_mod1=False,
                 from_log=None,
                 bootstrap=0,
                 explo_noise=0.01,
                 iterations=None):
              
        ################################### EXPERIMENT CONFIG ###################################
    
        self.name = name or 'Experiment'
        self.init_rest_trial = False
        self.bootstrap = bootstrap
        self.bootstrap_range_div = 1.
        self.iter = iterations or 50
        self.log_each = self.iter #must be <= iter
        self.eval_at = []
        self.n_eval = 0
        self.eval_modes = []
        
        self.gui = True
        
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
                                   'win_size': 1000,
                                   'competence_mode': 'knn',
                                   'k': 20,
                                   'progress_mode': 'local'}),
                    'miscRandom_global': (MiscRandomInterest, {
                                  'competence_measure': competence_dist,
                                  #'competence_measure': lambda target, reached, dist_max:competence_exp(target, reached, dist_min=0.01, dist_max=dist_max, power=20.),
                                   'win_size': 1000,
                                   'competence_mode': 'knn',
                                   'k': 20,
                                   'progress_mode': 'global'}),
            }
        
        self.sms = {
            'NN': (NonParametric, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':explo_noise}),
            'LWLR-BFGS-EXPLO': (NonParametric, {'fwd': 'LWLR', 'k':10, 'sigma':0.1, 'sigma_explo_ratio':explo_noise, 'inv': 'L-BFGS-B', 'maxfun':200, 'ftol':0, 'gtol':0}),
            'LWLR-BFGS-NOEXPLO': (NonParametric, {'fwd': 'LWLR', 'k':20, 'sigma':0.1, 'sigma_explo_ratio':0., 'inv': 'L-BFGS-B', 'maxfun':200, 'ftol':0, 'gtol':0}),
            'LWLR-CMAES': (NonParametric, {'fwd': 'LWLR', 'k':10, 'sigma':0.1, 'inv': 'CMAES', 'cmaes_sigma':0.05, 'sigma_explo_ratio':explo_noise, 'maxfevals':20}),
        }
          
        self.sm_model = sm_model
        self.im_model = im_model
        self.im_name = self.im_model
        
        sm = self.sm_model
        
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
        elif self.hierarchy_type == 2:
            self.m_spaces = dict(m_arm=range(12))
            self.s_spaces = dict(
                                 s_o1=[self.motor_n_dims + 23, self.motor_n_dims + 26],#range(self.motor_n_dims + 21, self.motor_n_dims + 27),
                                 s_o4=[self.motor_n_dims + 41, self.motor_n_dims + 44],#range(self.motor_n_dims + 39, self.motor_n_dims + 45),
                                 )

            self.modules = dict(
                                
                                
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
                            noise=envnoise,
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

# config_list = {"xp1":[
#                     "F-RmB",
#                     "F-RGB",
#                     "M-RMB",
#                     "M-P-AMB",
#                     "M-P-AMB",
#                     "F-RGB-LWR",
#                     "M-RMB-LWR",
#                     "M-P-AMB-LWR",
#                       ]}
 
config_list = {"xp1":[
                    "RmB",
                    "F-NN",
                    "M-NN-RMB",
                    "M-NN-LP-AMB",
                    "F-LWLR",
                    "M-LWLR-RMB",
                    "M-LWLR-LP-AMB",
#                     "RmB-ENVNOISE",
#                     "F-NN-ENVNOISE",
#                     "M-NN-RMB-ENVNOISE",
#                     "M-NN-LP-AMB-ENVNOISE",
#                     "F-LWLR-ENVNOISE",
#                     "M-LWLR-RMB-ENVNOISE",
#                     "M-LWLR-LP-AMB-ENVNOISE",
                      ],
               "xp2":[
                    "EXPLOIT-NN",
                    "EXPLOIT-LWLR",
#                     "EXPLOIT-NN-ENVNOISE",
#                     "EXPLOIT-LWLR-ENVNOISE",
                      ],
               "xp_snoise":[
                    "RmB-SNOISE",
                    "F-NN-SNOISE",
                    "M-NN-RMB-SNOISE",
                    "M-NN-LP-AMB-SNOISE",
                    "F-LWLR-SNOISE",
                    "M-LWLR-RMB-SNOISE",
                    "M-LWLR-LP-AMB-SNOISE",],
               "xp_long":[
                    "RmB-300k",
                    "F-NN-300k",
                    "M-NN-RMB-300k",
                    "M-NN-LP-AMB-300k",
                    "F-LWLR-300k",
                    "M-LWLR-RMB-300k",
                    "M-LWLR-LP-AMB-300k",],
               "xp_bootstrap":[
                    "RmB-bootstrap",
                    "F-NN-bootstrap",
                    "M-NN-RMB-bootstrap",
                    "M-NN-LP-AMB-bootstrap",
                    "F-LWLR-bootstrap",
                    "M-LWLR-RMB-bootstrap",
                    "M-LWLR-LP-AMB-bootstrap",],
               "xp_explo_noise":[
                    "F-NN-0.03",
                    "M-NN-RMB-0.03",
                    "M-NN-LP-AMB-0.03",
                    "F-LWLR-0.03",
                    "M-LWLR-RMB-0.03",
                    "M-LWLR-LP-AMB-0.03",
                    "F-NN-0.1",
                    "M-NN-RMB-0.1",
                    "M-NN-LP-AMB-0.1",
                    "F-LWLR-0.1",
                    "M-LWLR-RMB-0.1",
                    "M-LWLR-LP-AMB-0.1",
                    "F-NN-0.3",
                    "M-NN-RMB-0.3",
                    "M-NN-LP-AMB-0.3",
                    "F-LWLR-0.3",
                    "M-LWLR-RMB-0.3",
                    "M-LWLR-LP-AMB-0.3"],
               "xp_credit":[
                    "M-NN-RMB",
                    "M-NN-LP-AMB",
                      ],}

config = Config(name="RmB", hierarchy_type=0, babbling_name="motor", iterations=iterations)
configs[config.name] = config

config = Config(name="F-NN", hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB", hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="EXPLOIT-NN", hierarchy_type=2, supervisor_name="random", iterations=iterations)
configs[config.name] = config
 
config = Config(name="M-NN-LP-AMB", hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="F-LWLR", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="EXPLOIT-LWLR", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=2, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config


config = Config(name="RmB-ENVNOISE", hierarchy_type=0, envnoise=1, babbling_name="motor", iterations=iterations)
configs[config.name] = config

config = Config(name="F-NN-ENVNOISE", hierarchy_type=0, envnoise=1, iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB-ENVNOISE", hierarchy_type=1, envnoise=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config
 
config = Config(name="EXPLOIT-NN-ENVNOISE", hierarchy_type=2, envnoise=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config
 
config = Config(name="M-NN-LP-AMB-ENVNOISE", hierarchy_type=1, envnoise=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="F-LWLR-ENVNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=1, hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB-ENVNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=1, hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="EXPLOIT-LWLR-ENVNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=1, hierarchy_type=2, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB-ENVNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=1, hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config


config = Config(name="RmB-SNOISE", hierarchy_type=0, envnoise=2, babbling_name="motor", iterations=iterations)
configs[config.name] = config

config = Config(name="F-NN-SNOISE", hierarchy_type=0, envnoise=2, iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB-SNOISE", hierarchy_type=1, envnoise=2, supervisor_name="random", iterations=iterations)
configs[config.name] = config
 
config = Config(name="M-NN-LP-AMB-SNOISE", hierarchy_type=1, envnoise=2, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="F-LWLR-SNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=2, hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB-SNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=2, hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB-SNOISE", sm_model='LWLR-BFGS-EXPLO', envnoise=2, hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config


config = Config(name="RmB-300k", hierarchy_type=0, babbling_name="motor", iterations=3*iterations)
configs[config.name] = config

config = Config(name="F-NN-300k", hierarchy_type=0, iterations=3*iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB-300k", hierarchy_type=1, supervisor_name="random", iterations=3*iterations)
configs[config.name] = config

config = Config(name="M-NN-LP-AMB-300k", hierarchy_type=1, supervisor_name="interest", iterations=3*iterations)
configs[config.name] = config

config = Config(name="F-LWLR-300k", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=0, iterations=3*iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB-300k", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, supervisor_name="random", iterations=3*iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB-300k", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, supervisor_name="interest", iterations=3*iterations)
configs[config.name] = config



config = Config(name="RmB-bootstrap", hierarchy_type=0, bootstrap=10000, babbling_name="motor", iterations=iterations)
configs[config.name] = config

config = Config(name="F-NN-bootstrap", hierarchy_type=0, bootstrap=10000, iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB-bootstrap", hierarchy_type=1, bootstrap=10000, supervisor_name="random", iterations=iterations)
configs[config.name] = config
 
config = Config(name="M-NN-LP-AMB-bootstrap", hierarchy_type=1, bootstrap=10000, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="F-LWLR-bootstrap", sm_model='LWLR-BFGS-EXPLO', bootstrap=10000, hierarchy_type=0, iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB-bootstrap", sm_model='LWLR-BFGS-EXPLO', bootstrap=10000, hierarchy_type=1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB-bootstrap", sm_model='LWLR-BFGS-EXPLO', bootstrap=10000, hierarchy_type=1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config



config = Config(name="F-NN-0.03", hierarchy_type=0, explo_noise=0.03, iterations=iterations)
configs[config.name] = config

config = Config(name="F-NN-0.1", hierarchy_type=0, explo_noise=0.1, iterations=iterations)
configs[config.name] = config

config = Config(name="F-NN-0.3", hierarchy_type=0, explo_noise=0.3, iterations=iterations)
configs[config.name] = config


config = Config(name="M-NN-RMB-0.03", hierarchy_type=1, explo_noise=0.03, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB-0.1", hierarchy_type=1, explo_noise=0.1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-RMB-0.3", hierarchy_type=1, explo_noise=0.3, supervisor_name="random", iterations=iterations)
configs[config.name] = config

 
config = Config(name="M-NN-LP-AMB-0.03", hierarchy_type=1, explo_noise=0.03, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-LP-AMB-0.1", hierarchy_type=1, explo_noise=0.1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="M-NN-LP-AMB-0.3", hierarchy_type=1, explo_noise=0.3, supervisor_name="interest", iterations=iterations)
configs[config.name] = config


config = Config(name="F-LWLR-0.03", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=0, explo_noise=0.03, iterations=iterations)
configs[config.name] = config

config = Config(name="F-LWLR-0.1", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=0, explo_noise=0.1, iterations=iterations)
configs[config.name] = config

config = Config(name="F-LWLR-0.3", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=0, explo_noise=0.3, iterations=iterations)
configs[config.name] = config


config = Config(name="M-LWLR-RMB-0.03", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, explo_noise=0.03, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB-0.1", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, explo_noise=0.1, supervisor_name="random", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-RMB-0.3", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, explo_noise=0.3, supervisor_name="random", iterations=iterations)
configs[config.name] = config


config = Config(name="M-LWLR-LP-AMB-0.03", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, explo_noise=0.03, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB-0.1", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, explo_noise=0.1, supervisor_name="interest", iterations=iterations)
configs[config.name] = config

config = Config(name="M-LWLR-LP-AMB-0.3", sm_model='LWLR-BFGS-EXPLO', hierarchy_type=1, explo_noise=0.3, supervisor_name="interest", iterations=iterations)
configs[config.name] = config


