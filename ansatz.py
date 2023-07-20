from mpo import *
       
def rz_gate(t, angle): 
    return np.array([[np.exp(-1.j*angle*t/2), 0], [0, np.exp(1.j*angle*t/2)]])
    
def one_qubit_gate(theta, lamb, phi): 
    return np.array([[np.cos(theta/2), -np.exp(1.j*lamb)*np.sin(theta/2)], 
                 [np.exp(1.j*phi)*np.sin(theta/2), np.exp(1.j*(lamb+phi))*np.cos(theta/2)]])

class Rotation_MPO(MPO): 
    ''' 
    single layer of Rz gates turned into nodes with bond_dim=2
    angles must be np.array of shape (num_qubits,)
    '''
    def __init__(self, t, angles): 
        nodes = [Node(rz_gate(t, angle)[np.newaxis,np.newaxis,:,:]) for angle in angles]
        super().__init__(nodes)
        
class Single_Qubit_MPO(MPO): 
    '''
    single layer of general one qubit gates turned into nodes with bond_dim=2
    params must be np.array of shape (3*num_qubits,)
    '''
    def __init__(self, params): 
        nodes = [Node(one_qubit_gate(*params[i*3:(i+1)*3])[np.newaxis,np.newaxis,:,:]) for i in range(len(params)//3)]
        super().__init__(nodes)

class Entangling_MPO(MPO): 
    ''' two layers of entangling gates (CNOT for now) in brickwall pattern, contracted into an MPO of bond_dim=4 '''
    def __init__(self, num_qubits, parity): 
        node0 = Node(np.eye(2,2)[np.newaxis,np.newaxis,:,:])
        node1,node2 = gate_to_nodes(np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]))
        if parity == 0: 
            super().__init__([node0] + [node1,node2] * (num_qubits//2 -1) + [node0])
        else: 
            super().__init__([node1,node2]*(num_qubits//2))
    
class Ansatz():
    ''' 
    contains a collection of MPOs
    angles is a np.array of shape (num_qubits,) 
    params is a np.array of shape (2*num_w_layers+1, 3*num_qubits)
    '''
    def __init__(self, t, angles, params):
        self.t = t
        self.angles = angles
        self.params = params 
        
        self.even_mpo = Entangling_MPO(len(angles), 0)
        self.odd_mpo = Entangling_MPO(len(angles), 1)
        
        self.param_mpo_stacks = [Rotation_MPO(t, angles)] + [Single_Qubit_MPO(params[i]) for i in range(params.shape[0])]
        self.num_stacks = 1 + params.shape[0]  # total stacks of parametrized gates = 2*num_w_layers + 2
        self.num_qubits = len(angles)
        
    def mpo(self, min_sv_ratio=None, max_dim=None): 
        mpo = self.param_mpo_stacks[0]
        for i in range(1,self.num_stacks-1): 
            param_mpo = self.param_mpo_stacks[i]
            if i%2 == 1: 
                mpo = (self.odd_mpo @ mpo @ self.odd_mpo).compress(min_sv_ratio, max_dim)
                mpo = (param_mpo @ mpo @ param_mpo.conj())
                mpo = (self.odd_mpo @ mpo @ self.odd_mpo).compress(min_sv_ratio, max_dim)
                
            elif i%2 == 0:
                mpo = (self.even_mpo @ mpo @ self.even_mpo).compress(min_sv_ratio, max_dim)
                mpo = (param_mpo @ mpo @ param_mpo.conj())
                mpo = (self.even_mpo @ mpo @ self.even_mpo).compress(min_sv_ratio, max_dim)
            
        param_mpo = self.param_mpo_stacks[-1]
        return param_mpo @ mpo @ param_mpo.conj()
        
    def update_node(self, stack_idx, node_idx, new_params): 
        if stack_idx == 0: 
            self.angles[node_idx] = new_params
            self.param_mpo_stacks[stack_idx].nodes[node_idx] = Node(rz_gate(self.t, new_params)[np.newaxis,np.newaxis,:,:])
        else: 
            self.params[stack_idx-1][node_idx*3:(node_idx+1)*3] = new_params
            self.param_mpo_stacks[stack_idx].nodes[node_idx] = Node(one_qubit_gate(*new_params)[np.newaxis,np.newaxis,:,:])