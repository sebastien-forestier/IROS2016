from copy import deepcopy


class Hierarchy(object):
    """ 
    Hierarchy implementation
    """

    def __init__(self):
        """ 
        Build an empty hierarchy.
        """
        self.motor_spaces = {}
        self.sensori_spaces = {}
        self.modules = {}
        self.operators = {}

    def _space2str(self, space):
        """ 
        Return string from int list.
        """
        return str(sorted(space))

    def add_module(self, mid, operator=None):
        """
        Add a module.
        """
        if mid not in self.modules:
            self.modules[mid] = set()
            self.operators[mid] = operator
        
    def add_motor_space(self, space):
        """
        Add a motor space.
        """
        space_name = self._space2str(space)
        if space_name not in self.motor_spaces:
            self.motor_spaces[space_name] = set()
            
    def add_sensori_space(self, space):
        """
        Add a sensori space.
        """
        space_name = self._space2str(space)
        if space_name not in self.sensori_spaces:
            self.sensori_spaces[space_name] = set()
            
    def is_motor_space(self, space):
        """
        Test if space is a motor space in the hierarchy.
        """
        return self._space2str(space) in self.motor_spaces
            
    def is_sensori_space(self, space):
        """
        Test if space is a sensori space in the hierarchy.
        """
        return self._space2str(space) in self.sensori_spaces
    
    def is_space(self, space): 
        """
        Test if space is a space in the hierarchy.
        """
        return self.is_motor_space(space) or self.is_sensori_space(space)
    
    def is_module(self, mid):
        """
        Test if module is a module in the hierarchy.
        """
        return isinstance(mid, basestring) and mid in self.modules 
        
    def add_edge_space_module(self, mid, space):
        """ 
        Add an edge (dependency) between a module and one of its motor space. 
        """
        if not self.is_module(mid):
            raise KeyError('module do not exist in graph')
        if not self.is_space(space):
            raise KeyError('space do not exist in graph')            
        self.modules[mid].add(self._space2str(space))
        
    def add_edge_module_space(self, mid, space):
        """ 
        Add an edge (dependency) between a sensory space and one module. 
        """
        if not self.is_module(mid):
            raise KeyError('module do not exist in graph')
        if not self.is_space(space):
            raise KeyError('space do not exist in graph')  
        if self.is_motor_space(space):
            self.motor_spaces[self._space2str(space)].add(mid)
        else:
            self.sensori_spaces[self._space2str(space)].add(mid)
            
    def module_children(self, mid):
        """ 
        Returns a list of all space dependencies of a module. 
        """
        return list(self.modules[mid])
    
    def space_children(self, space):
        """ 
        Returns a list of all module dependencies of a sensori space. 
        """
        return list(self.sensori_spaces[self._space2str(space)])
    
    def module_ancestors(self, mid, levels=None):
        """
        Returns a list of all parents' and ancestors' modules of given module (Not efficient).
        """
        levels = levels or self.module_levels()
        higher_mids = [parent_mid for parent_mid in self.modules if levels[parent_mid] > levels[mid]]
        close_parents = [parent_mid for parent_mid in higher_mids if mid in 
                            [children_mid for space_name in self.module_children(parent_mid) for children_mid in self.sensori_spaces[space_name]]]
        return list(set(close_parents + [parent_mid for close_parent in close_parents for parent_mid in self.module_ancestors(close_parent, levels)]))
        
    def parents_with_operator(self, mid, operators):
        """
        Returns a list of all parents of given module that have one of the given operators.
        """
        mid_parents = self.module_ancestors(mid)        
        return [mid_parent for mid_parent in mid_parents if self.operators[mid_parent] in operators]

    def _ind_nodes(self, graph):
        """ 
        Returns a list of all nodes in the graph with no dependencies. 
        """
        if graph is None:
            raise Exception("Graph given is None")
        all_nodes, dependent_nodes = set(graph.keys()), set()
        for downstream_nodes in graph.itervalues():
            [dependent_nodes.add(node) for node in downstream_nodes]
        return list(all_nodes - dependent_nodes)
    
    def _dependencies(self, target_node, graph):
        """ 
        Returns a list of all nodes from incoming edges. 
        """
        if graph is None:
            raise Exception("Graph given is None")
        result = set()
        for node, outgoing_nodes in graph.iteritems():
            if target_node in outgoing_nodes:
                result.add(node)
        return list(result)
    
    def topological_sort(self):
        """ 
        Returns a topological ordering of the DAG.
        Raises an error if this is not possible (graph is not valid).
        """
        graph = self.modules.copy()
        graph.update(self.motor_spaces)
        graph.update(self.sensori_spaces)
        graph = deepcopy(graph)
        l = []
        q = deepcopy(self._ind_nodes(graph))
        while len(q) != 0:
            n = q.pop(0)
            l.append(n)
            iter_nodes = deepcopy(graph[n])
            for m in iter_nodes:
                graph[n].remove(m)
                if len(self._dependencies(m, graph)) == 0:
                    q.append(m)
        if len(l) != len(graph.keys()):
            raise ValueError('graph is not acyclic')
        list.reverse(l)
        return l

    def module_levels(self):
        """
        Returns a dictionary containing the level of each module in the hierarchy.
        """
        ts = [mid for mid in self.topological_sort() if self.is_module(mid)]
        levels = {}
        for mid in ts:
            if len([m_space for m_space in self.module_children(mid) if m_space in self.sensori_spaces]) == 0:
                levels[mid] = 0
            else:
                levels[mid] = 1 + max([levels[l] for space_name in self.modules[mid] for l in self.sensori_spaces[space_name]])
        return levels
    
    

if __name__ == "__main__":
    
    hierarchy = Hierarchy()
    
    hierarchy.add_module("mod1", None)
    hierarchy.add_module("mod2", None)
    hierarchy.add_module("mod3", "seq")
    hierarchy.add_module("mod4", None)
    
    hierarchy.add_motor_space([0])
    hierarchy.add_motor_space([1])
    
    hierarchy.add_sensori_space([2])
    hierarchy.add_sensori_space([3])    
    hierarchy.add_sensori_space([4])
    
    hierarchy.add_edge_space_module("mod1", [0])
    hierarchy.add_edge_space_module("mod2", [1])
    hierarchy.add_edge_space_module("mod3", [2])
    hierarchy.add_edge_space_module("mod3", [3])
    hierarchy.add_edge_space_module("mod4", [2])
    
    hierarchy.add_edge_module_space("mod1", [2])
    hierarchy.add_edge_module_space("mod2", [3])
    hierarchy.add_edge_module_space("mod3", [4])
    hierarchy.add_edge_module_space("mod4", [4])
    
    print hierarchy.modules
    print hierarchy.motor_spaces
    print hierarchy.sensori_spaces
    print hierarchy.topological_sort()
    print hierarchy.module_levels()
    print hierarchy.module_ancestors("mod3")
    print hierarchy.parents_with_operator("mod1", ["par", None])
    