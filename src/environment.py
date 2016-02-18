import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('QT4Agg')

from combined_env import CombinedEnvironment, HierarchicallyCombinedEnvironment
from dynamic_env import DynamicEnvironment


from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.simple_arm.simple_arm import joint_positions


class GripArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 lengths, angle_shift, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.lengths = lengths
        self.angle_shift = angle_shift
        self.rest_state = rest_state
        self.reset()
        
    def reset(self):
        self.gripper = self.rest_state[3]
        self.logs = []
        
    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        #return m

    def compute_sensori_effect(self, m):
        a = self.angle_shift + np.cumsum(np.array(m[:-1]))
        a_pi = np.pi * a 
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.lengths), np.sum(np.sin(a_pi)*self.lengths)])
        if m[-1] >= 0.:
            new_gripper = 1. 
        else:
            new_gripper = -1.
        gripper_change = (self.gripper - new_gripper) / 2.
        self.gripper = new_gripper
        angle = np.mod(a[-1] + 1, 2) - 1
        self.logs.append(m)
        return [hand_pos[0], hand_pos[1], angle, gripper_change, self.gripper]
    
    
    def plot(self, ax, i, **kwargs_plot):
        m = self.logs[i]
        angles = m[:-1]
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x, y = [np.hstack((0., a)) for a in x, y]
        ax.plot(x, y, 'grey', lw=4, **kwargs_plot)
        ax.plot(x[0], y[0], 'ok', ms=12, **kwargs_plot)
        for i in range(len(self.lengths)-1):
            ax.plot(x[i+1], y[i+1], 'ok', ms=12, **kwargs_plot)
        ax.plot(x[-1], y[-1], 'or', ms=4, **kwargs_plot)
        ax.axis([-1.6, 2.1, -1., 2.1])        

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
        self.plot_gripper(ax, x[-1], y[-1], np.cumsum(m[:-1]), m[-1] >= 0., **kwargs_plot)
        
    def plot_gripper(self, ax, x, y, angle, gripper_open, **kwargs_plot):
        if gripper_open:
            if kwargs_plot.has_key("alpha"):
                color=matplotlib.colors.ColorConverter().to_rgba('g', alpha=kwargs_plot["alpha"])
            else:
                color = 'g'
            ax.plot(x, y, 'o', markerfacecolor='none', markeredgewidth=6, markeredgecolor=color, ms=26, **kwargs_plot)
                
        else:
            ax.plot(x, y, 'og', ms=18, **kwargs_plot)
        
       
class Stick(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length, type, handle_tol, handle_noise, rest_state, perturbation=None):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length = length
        self.type = type
        self.handle_tol = handle_tol
        self.handle_tol_sq = handle_tol * handle_tol
        self.handle_noise = handle_noise
        self.rest_state = rest_state
        self.perturbation = perturbation
        
        if self.perturbation == "BrokenTool1":
            self.length_breakpoint = 0.5
            self.angle_breakpoint = np.pi * 0.5
        
        self.reset()


    def reset(self):
        self.held = False
        self.handle_pos = np.array(self.rest_state[0:2])
        self.angle = self.rest_state[2]
        self.compute_end_pos()
        self.logs = []
        
    def compute_end_pos(self):
        if self.perturbation == "BrokenTool1":
            a = np.pi * self.angle
            breakpoint = [self.handle_pos[0] + np.cos(a) * self.length * self.length_breakpoint, 
                            self.handle_pos[1] + np.sin(a) * self.length * self.length_breakpoint]
            self.end_pos = [breakpoint[0] + np.cos(a + self.angle_breakpoint) * (self.length * (1. - self.length_breakpoint)), 
                            breakpoint[1] + np.sin(a + self.angle_breakpoint) * (self.length * (1. - self.length_breakpoint))]
        else:
            a = np.pi * self.angle
            self.end_pos = [self.handle_pos[0] + np.cos(a) * self.length, 
                            self.handle_pos[1] + np.sin(a) * self.length]
                
        
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        hand_pos = m[0:2]
        hand_angle = m[2]
        gripper_change = m[3]
        
        if not self.held:
            if gripper_change == 1. and (hand_pos[0] - self.handle_pos[0]) ** 2. + (hand_pos[1] - self.handle_pos[1]) ** 2. < self.handle_tol_sq:
                self.handle_pos = hand_pos
                self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_end_pos()
                self.held = True
        else:
            if gripper_change == 0:
                self.handle_pos = hand_pos
                self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_end_pos()
            else:
                self.held = False
        
        #print "Stick log added"
        self.logs.append([self.handle_pos, 
                          self.angle, 
                          self.end_pos, 
                          self.held])
        #print "Tool hand_pos:", hand_pos, "hand_angle:", hand_angle, "gripper_change:", gripper_change, "self.handle_pos:", self.handle_pos, "self.angle:", self.angle, "self.held:", self.held 
        return list(self.end_pos) # Tool pos
    
    def plot(self, ax, i, **kwargs_plot):
        handle_pos = self.logs[i][0]
        end_pos = self.logs[i][2]
        
        
        if self.perturbation == "BrokenTool1":
            a = np.pi * self.logs[i][1]
            breakpoint = [handle_pos[0] + np.cos(a) * self.length * self.length_breakpoint, 
                            handle_pos[1] + np.sin(a) * self.length * self.length_breakpoint]
        
            ax.plot([handle_pos[0], breakpoint[0]], [handle_pos[1], breakpoint[1]], '-b', lw=6, **kwargs_plot)
            ax.plot([breakpoint[0], end_pos[0]], [breakpoint[1], end_pos[1]], '-b', lw=6, **kwargs_plot)
        else:
            ax.plot([handle_pos[0], end_pos[0]], [handle_pos[1], end_pos[1]], '-b', lw=6, **kwargs_plot)
        ax.plot(handle_pos[0], handle_pos[1], 'og', ms=12, **kwargs_plot)
        if self.type == "magnetic":
            ax.plot(end_pos[0], end_pos[1], 'om', ms=12, **kwargs_plot)
        else:
            ax.plot(end_pos[0], end_pos[1], 'or', ms=12, **kwargs_plot)
                    
    

class MagneticObject(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 object_tol1, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.object_tol1_sq = object_tol1 * object_tol1
        self.rest_state = rest_state
        self.reset()
        
        
    def reset(self):
        self.move = 0
        self.pos = self.rest_state
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        #print "SceneObject m", m
        if self.move == 1 or (m[0] - self.pos[0]) ** 2 + (m[1] - self.pos[1]) ** 2 < self.object_tol1_sq:
            self.pos = m[0:2]
            self.move = 1
            #print "OBJECT PUSHED BY TOOL 1!"
        self.logs.append([self.pos,
                          self.move])
        return list(self.pos)
    
    def plot(self, ax, i, **kwargs_plot):
        pos = self.logs[i][0]        
        rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='m', **kwargs_plot)
        ax.add_patch(rectangle) 


class ScratchObject(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 object_tol2, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.object_tol2_sq = object_tol2 * object_tol2
        self.rest_state = rest_state
        self.reset()
        
        
    def reset(self):
        self.move = 0
        self.pos = self.rest_state
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        #print "SceneObject m", m
        if self.move == 2 or (m[2] - self.pos[0]) ** 2 + (m[3] - self.pos[1]) ** 2 < self.object_tol2_sq:
            self.pos = m[2:4]
            self.move = 2
            #print "OBJECT PUSHED BY TOOL 2!"
        self.logs.append([self.pos,
                          self.move])
        return list(self.pos)
    
    def plot(self, ax, i, **kwargs_plot):
        pos = self.logs[i][0]        
        rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='r', **kwargs_plot)
        ax.add_patch(rectangle) 
        

class StaticObject(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 object_shape, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.object_shape = object_shape
        self.rest_state = rest_state
        self.reset()
        
    def reset(self):
        self.move = 0
        self.pos = self.rest_state
        self.logs = []
        
    def compute_motor_command(self, m):
        return m

    def compute_sensori_effect(self, m):
        return list(self.pos)
    
    def plot(self, ax, i, **kwargs_plot):
        pos = self.rest_state   
        if self.object_shape == "rectangle":
            rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='k', **kwargs_plot)
            ax.add_patch(rectangle)
        elif self.object_shape == "circle":
            circle = plt.Circle((pos[0], pos[1]), 0.05, fc='k', **kwargs_plot)
            ax.add_patch(circle)            
        else:
            raise NotImplementedError  
    

class Animal(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 species, noise, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.species = species
        self.noise = noise
        self.rest_state = rest_state
        self.reset()
        
        
    def reset(self):
        self.pos = np.array(self.rest_state)
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        self.pos = self.pos + np.random.randn(2) * self.noise
        self.logs.append(self.pos)
        return list(self.pos)
    
    def plot(self, ax, i, **kwargs_plot):
        pos = self.logs[i]
        if self.species == "cat":
            rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='y', **kwargs_plot)
            ax.add_patch(rectangle)
        elif self.species == "dog":
            rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='c', **kwargs_plot)
            ax.add_patch(rectangle)          
        else:
            raise NotImplementedError       
        
        
class Box(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, box_m_mins, box_m_maxs):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        
        self.box_m_mins = box_m_mins
        self.box_m_maxs = box_m_maxs
        self.center = (np.array(self.box_m_mins) + np.array(self.box_m_maxs)) / 2.
        self.reset()
        
    def reset(self):        
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        full = 0
        min_d = 2
        for i in range(6):
            #print m, i, m[0+i*2:2+i*2]
            d = np.linalg.norm(np.array(m[0+i*2:2+i*2]) - self.center)
            if d < min_d:
                min_d = d                
            if m[0+i*2] >= self.box_m_mins[0] and m[0+i*2] <= self.box_m_maxs[0] and m[1+i*2] >= self.box_m_mins[1] and m[1+i*2] <= self.box_m_maxs[1]:            
                full = 1
                
        self.logs.append([full, d])
        return [full, d] # object pos + hand pos + gripper state + tool1 end pos + tool2 end pos
    
    def plot(self, ax, i, **kwargs_plot): 
        if i >= len(self.logs):
            i = len(self.logs) - 1       
        if self.logs[i][0]:
            fc = "none"#"k"
        else:
            fc = "none"     
        rectangle = plt.Rectangle((self.box_m_mins[0], self.box_m_mins[1]), 
                                  self.box_m_maxs[0] - self.box_m_mins[0], 
                                  self.box_m_maxs[1] - self.box_m_mins[1], 
                                  fc=fc, alpha=0.5, lw=4)
        ax.add_patch(rectangle)



class IROS2016Environment(DynamicEnvironment):
    def __init__(self, move_steps=50, max_params=None, perturbation=None, gui=False):

        def motor_perturbation(m):
            if perturbation == "BrokenMotor":
                m[2] = 0                
                return m
            elif perturbation == "ShiftedMotor":
                m[0] = m[0] - 0.3
                if m[0] < -1.:
                    m[0] = -1.
                return m
            else:
                return m
            
        gripArm_cfg = dict(m_mins=[-1, -1, -1, -1],  # joints pos + gripper state
                             m_maxs=[1, 1, 1, 1], 
                             s_mins=[-1, -1, -1, -1, -1], # hand pos + hand angle + gripper_change + gripper state
                             s_maxs=[1, 1, 1, 1, 1], 
                             lengths=[0.5, 0.3, 0.2], 
                             angle_shift=0.5,
                             rest_state=[0., 0., 0., 0.])
        
        stick1_cfg = dict(m_mins=[-1, -1, -1, -1, -1], 
                         m_maxs=[1, 1, 1, 1, 1], 
                         s_mins=[-2, -2],  # Tool pos
                         s_maxs=[2, 2],
                         length=0.5, 
                         type="magnetic",
                         handle_tol=0.25, 
                         handle_noise=0., 
                         rest_state=[-0.75, 0.25, 0.75],
                         perturbation=perturbation)
        
        stick2_cfg = dict(m_mins=[-1, -1, -1, -1, -1], 
                         m_maxs=[1, 1, 1, 1, 1], 
                         s_mins=[-2, -2], 
                         s_maxs=[2, 2],
                         length=0.5, 
                         type="scratch",
                         handle_tol=0.25, 
                         handle_noise=0., 
                         rest_state=[0.75, 0.25, 0.25])
        
        sticks_cfg = dict(
                        s_mins = [-2, -2, -2, -2],
                        s_maxs = [2, 2, 2, 2],
                        envs_cls = [Stick, Stick],
                        envs_cfg = [stick1_cfg, stick2_cfg],
                        combined_s = lambda s:s  # from s:  Tool1 end pos + Tool2 end pos
                        )
        
        arm_stick_cfg = dict(m_mins=list([-1.] * 4), # 3DOF + gripper
                             m_maxs=list([1.] * 4),
                             s_mins=list([-2.] * 7),
                             s_maxs=list([2.] * 7),
                             top_env_cls=CombinedEnvironment, 
                             lower_env_cls=GripArmEnvironment, 
                             top_env_cfg=sticks_cfg, 
                             lower_env_cfg=gripArm_cfg, 
                             fun_m_lower= lambda m:motor_perturbation(m[0:4]),
                             fun_s_lower=lambda m,s:s+s,  # (hand pos + hand angle + gripper_change + gripper state) * 2 tools
                             fun_s_top=lambda m,s_lower,s:s_lower[0:2] + [s_lower[4]] + s) # from s: Tool1 end pos + Tool2 end pos  from m: hand_pos + gripper state
        
        
        
        mag_object_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol1 = 0.25, 
                          rest_state = [-0.3, 1.1])
        
        mag_object2_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol1 = 0., 
                          rest_state = [-0.5, 1.5])
        
        mag_object3_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol1 = 0., 
                          rest_state = [-0.3, 1.5])
        
        scratch_object_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol2 = 0.25, 
                          rest_state = [0.3, 1.1])
        
        scratch_object2_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol2 = 0., 
                          rest_state = [0.3, 1.5])
        
        scratch_object3_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol2 = 0., 
                          rest_state = [0.5, 1.5])
        
        cat_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          species= "cat", 
                          noise = 0.01, 
                          rest_state = [-0.1, 1.1])
        
        dog_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          species= "dog", 
                          noise = 0.01, 
                          rest_state = [0.1, 1.1])
        
        static1_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_shape = "rectangle",
                          rest_state = [-0.7, 1.1])
        
        static2_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_shape = "rectangle",
                          rest_state = [-0.5, 1.1])
        
        static3_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_shape = "rectangle",
                          rest_state = [0.5, 1.1])
        
        static4_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_shape = "rectangle",
                          rest_state = [0.7, 1.1])
        
        objects_cfg = dict(
                        s_mins = list([-2.]*24),
                        s_maxs = list([2.]*24),
                        envs_cls = [MagneticObject, MagneticObject, MagneticObject, ScratchObject, ScratchObject, ScratchObject, Animal, Animal, StaticObject, StaticObject, StaticObject, StaticObject], 
                        envs_cfg = [mag_object_cfg, mag_object2_cfg, mag_object3_cfg, scratch_object_cfg, scratch_object2_cfg, scratch_object3_cfg, cat_cfg, dog_cfg, static1_cfg, static2_cfg, static3_cfg, static4_cfg], 
                        combined_s = lambda s:s
                        )
        
        arm_sticks_objects_cfg = dict(
                                           m_mins=arm_stick_cfg['m_mins'],
                                           m_maxs=arm_stick_cfg['m_maxs'],
                                           s_mins=list([-2.] * 31),
                                           s_maxs=list([2.] * 31), # (hand pos + gripper state + tool1 end pos + tool2 end pos + last objects pos
                                           top_env_cls=CombinedEnvironment, 
                                           lower_env_cls=HierarchicallyCombinedEnvironment, 
                                           top_env_cfg=objects_cfg, 
                                           lower_env_cfg=arm_stick_cfg, 
                                           fun_m_lower= lambda m:m,
                                           fun_s_lower=lambda m,s:s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:]+s[3:],
                                           fun_s_top=lambda m,s_lower,s: s_lower + s)
        
#         box1_cfg = dict(
#                      m_mins = [-2.]*12,
#                      m_maxs = [2.]*12,
#                      s_mins = [0.],
#                      s_maxs = [1.],
#                      box_m_mins = [-1.5, -0.1], 
#                      box_m_maxs = [-1.3, 0.1]
#                      )
#         
#         box2_cfg = dict(
#                      m_mins = [-2.]*12,
#                      m_maxs = [2.]*12,
#                      s_mins = [0.],
#                      s_maxs = [1.],
#                      box_m_mins = [-1.2, 1.], 
#                      box_m_maxs = [-1., 1.2]
#                      )
#         
#         
#         boxes_cfg = dict(
#                         s_mins = list([0., 0., 0., 0.]),
#                         s_maxs = list([1., 2., 1., 2.]),
#                         envs_cls = [Box, Box], 
#                         envs_cfg = [box1_cfg, box2_cfg], 
#                         combined_s = lambda s:s
#                         )
#         
#         static_env_cfg = dict(m_mins=arm_stick_cfg['m_mins'],
#                                m_maxs=arm_stick_cfg['m_maxs'],
#                                s_mins=list([-2.] * 19 + [0., 0., 0., 0.]),
#                                s_maxs=list([2.] * 19 + [1., 2., 1., 2.]),
#                                top_env_cls=CombinedEnvironment, 
#                                lower_env_cls=HierarchicallyCombinedEnvironment, 
#                                top_env_cfg=boxes_cfg, 
#                                lower_env_cfg=arm_sticks_objects_cfg, 
#                                fun_m_lower= lambda m:m,
#                                fun_s_lower=lambda m,s:s[7:19]+s[7:19],
#                                fun_s_top=lambda m,s_lower,s: s_lower + s)
#         
        
        denv_cfg = dict(env_cfg=arm_sticks_objects_cfg,
                        env_cls=HierarchicallyCombinedEnvironment,
                        m_mins=[-1.] * 4 * 3, 
                        m_maxs=[1.] * 4 * 3, 
                        s_mins=[-1.5] * 31 * 3,
                        s_maxs=[1.5] * 31 * 3,
                        n_bfs = 2,
                        n_motor_traj_points=3, 
                        n_sensori_traj_points=3, 
                        move_steps=move_steps, 
                        n_dynamic_motor_dims=4,
                        n_dynamic_sensori_dims=31, 
                        max_params=max_params,
                        motor_traj_type="DMP", 
                        sensori_traj_type="samples",
                        optim_initial_position=False, 
                        optim_end_position=True, 
                        default_motor_initial_position=[0.]*4, 
                        default_motor_end_position=[0.]*4,
                        default_sensori_initial_position=[0., 1., 0., 0., -0.85, 0.35, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                        default_sensori_end_position=[0., 1., 0., 0., -0.85, 0.35, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        gui=gui)
            

# DEFAULT S
# s_t1 [-1.10355339 -1.10355339 -1.10355339  0.60355339  0.60355339  0.60355339]
# s_t2 [ 1.10355339  1.10355339  1.10355339  0.60355339  0.60355339  0.60355339]
# s_o1 [-0.3 -0.3 -0.3  1.1  1.1  1.1]
# s_o2 [ 0.3  0.3  0.3  1.1  1.1  1.1]
# s_o3 [-0.11926431  0.00277633  0.01955739  1.1217168   1.34855082  1.35216384]
# s_o4 [ 0.13369242  0.16894663  0.15680744  1.08700156  1.0268538   0.9980367 ]
# s_o5 [-0.5 -0.5 -0.5  1.1  1.1  1.1]
# s_o6 [ 0.5  0.5  0.5  1.1  1.1  1.1]

        
        DynamicEnvironment.__init__(self, **denv_cfg)
        
        
        
