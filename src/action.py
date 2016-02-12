import numpy as np


class Action(object):    
    def __init__(self, m_deps, m_dims=None, operator=None, actions=None):
        self.m_deps = m_deps
        self.m_dims = m_dims
        self.operator = operator
        self.actions = actions or []
        
        self.n_actions = len(self.actions)
        self.n_iterations_actions = [action.n_iterations for action in self.actions]
        self.n_iterations = self.compute_n_iterations()  
        
        assert (m_dims is not None) or (operator is not None and actions is not None), (m_dims, operator, actions)
        
    
    def compute_n_iterations(self):
        if self.operator == "seq":
            return sum(self.n_iterations_actions)
        elif self.operator == "par":
            return max(self.n_iterations_actions)
        elif self.operator == None:
            return 1
        else:
            raise NotImplementedError
                  
    def get_m_seq(self, n_dims):
        if self.operator == "seq":
            return [item for action in self.actions for item in action.get_m_seq(n_dims)]
        elif self.operator == "par":
            #Warning: "par" of "seq" not implemented
            return np.sum(np.array([action.get_m_seq(n_dims) for action in self.actions]), axis=0)
        elif self.operator == None:
            m = np.zeros((n_dims,))
            m[self.m_dims] = self.m_deps
            return [m]
        else:
            raise NotImplementedError
        
    def print_action(self, depth = 0):
        for _ in range(depth):
            print "    ",
        print "Action", self.operator, self.n_iterations, len(self.actions)
        if self.operator == "seq":
            [action.print_action(depth+1) for action in self.actions]
        elif self.operator == "par":
            [action.print_action(depth+1) for action in self.actions]            
        elif self.operator == None:
            for _ in range(depth):
                print "    ",
            print "Primitive Action", self.m_dims
        