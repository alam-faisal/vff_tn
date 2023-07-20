import copy
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm

default_min_sv_ratio = 1e-12

pauli = np.array([np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1.j],[1.j,0]]), np.array([[1,0],[0,-1]])])
pauli_tensor = np.array([[np.kron(pauli[i], pauli[j]) for i in range(4)] for j in range(4)])

########################
######### Node ########
########################

def group_ind(data, recipient_index, donor_index): 
    """ combines recipient_index and donor_index into a single index """
    data = np.moveaxis(data, donor_index, recipient_index+1)
    shape  = list(data.shape[:recipient_index]) + [data.shape[recipient_index] * data.shape[recipient_index+1]] \
            + list(data.shape[recipient_index+2:])
    return np.reshape(data, shape, order='F')

def ungroup_ind(data, index_to_ungroup, new_index_dim, new_index_location):
    """
    ungroups index_to_ungroup with dimension reduced_dim * new_index_dim into axes with dimensions reduced_dim and new_index_dim
    index with new_index_dim goes to new_index_location
    index with reduced_dim takes place of index_to_ungroup 
    """
    reduced_dim = data.shape[index_to_ungroup]//new_index_dim
    shape = list(data.shape[:index_to_ungroup]) + [reduced_dim, new_index_dim] + list(data.shape[index_to_ungroup+1:])
    ungrouped = np.reshape(data, shape, order='F')
    return np.moveaxis(ungrouped, index_to_ungroup+1, new_index_location)

class Node():  
    """ Tensor of order 4, shape format is (left_bond_dim, right_bond_dim, top_spin_dim, bottom_spin_dim) """
    def __init__(self, data): 
        self.data = data 
        self.shape = data.shape
    
    def svd(self, which='left', min_sv_ratio=None, max_dim=None):
        """
        does an asymmetric SVD, keeping both spin indices on 'which' side, truncates depending on either max_dim or min_sv_ratio
        """
        if which == 'left': 
            data = group_ind(group_ind(self.data, 0,2), 0,2)
        else: 
            data = group_ind(group_ind(self.data, 1,2), 1,2)
        
        # now we truncate
        u,s,vh = svd(data)        
        if min_sv_ratio is not None: 
            s = s[s>min_sv_ratio*s[0]]
        elif max_dim is not None:
            dim = min(max_dim, len(s[s>default_min_sv_ratio*s[0]]))
            s = s[:dim]
        else: 
            s = s[s>default_min_sv_ratio*s[0]]
        u = u[:,:len(s)] @ np.diag(np.sqrt(s))
        vh = np.diag(np.sqrt(s)) @ vh[:len(s),:]
        
        if which == 'left': 
            left_tensor = Node(ungroup_ind(ungroup_ind(u, 0,2,2), 0,2,2))
            right_tensor = vh
        else: 
            left_tensor = u
            right_tensor = Node(ungroup_ind(ungroup_ind(vh, 1,2,2), 1,2,2))
        
        return left_tensor, right_tensor
    
    def conj(self): 
        """ swap spin indices """
        return Node(np.swapaxes(self.data.conj(), 2,3))  

def gate_to_nodes(gate): 
    """
    turns a two qubit gate matrix constructed from tensor product to two nodes connected by a bond
    the matrix has the form A_{ij}^{i'j'} where ij correspond to incoming wires and i'j' the outgoing wires of the two qubits 
    we have to first reshape this into A_{ii'}{jj'} before we can SVD
    we also delete any singular values below default_min_sv_ratio
    """
    gate = gate.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    u, s, vh = svd(gate)
    s = s[s>default_min_sv_ratio*s[0]]     
    u = u[:,:len(s)]
    vh = vh[:len(s),:]
    return Node(u.T.reshape(len(s),2,2)[np.newaxis,:,:,:]), Node((np.diag(s) @ vh).reshape(len(s),2,2)[:,np.newaxis,:,:])
    
######################
######## MPO #########
######################
    
class MPO: 
    def __init__(self, nodes): 
        self.nodes = nodes 
        self.num_nodes = len(nodes)
        self.skeleton = [node.shape for node in nodes]
        self.weights = [np.linalg.norm(np.ravel(node.data)) for node in nodes]
        self.max_dim = max(list(sum(self.skeleton, ())))
        
    def to_matrix(self):
        matrix = self.nodes[0].data
        for i in range(1, self.num_nodes): 
            matrix = np.einsum('ijkl,jmno->imknlo', matrix, self.nodes[i].data)
            matrix = group_ind(group_ind(matrix, 4,5), 2,3) 
        return matrix[0][0] 
    
    def __matmul__(self, other):
        return MPO([Node(group_ind(group_ind(np.einsum('ijkl,mnlo->imjnko', self.nodes[i].data, other.nodes[i].data), 0,1), 1,2))
                    for i in range(self.num_nodes)])
    
    def compress(self, min_sv_ratio=None, max_dim=None): 
        nodes = copy.deepcopy(self.nodes)
        for i in range(int(self.num_nodes//2)):
            left_tensor, right_tensor = nodes[i].svd('left', min_sv_ratio, max_dim)
            nodes[i] = left_tensor
            nodes[i+1] = Node(np.einsum('ij,jmno->imno', right_tensor, nodes[i+1].data))
            
        for i in range(self.num_nodes-1, self.num_nodes//2 - 1, -1):
            left_tensor, right_tensor = nodes[i].svd('right', min_sv_ratio, max_dim)
            nodes[i] = right_tensor
            nodes[i-1] = Node(np.einsum('ijkl,jm->imkl', nodes[i-1].data, left_tensor))
        
        return MPO(nodes) 
    
    def trace(self): 
        """ this returns Tr(MPO)/2^n, which matches the cost function and prevents blow up of numbers """
        trace = np.einsum('ijkk', self.nodes[0].data)/2
        for i in range(1, self.num_nodes): 
            trace = np.einsum('ij,jmno->imno', trace, self.nodes[i].data)
            trace = np.einsum('ijkk', trace)/2
        return trace[0][0]
    
    def mult_and_trace(self, other, start='top'):
        """ this returns Tr(self @ other)/2^n """ 
        num_nodes = self.num_nodes
        if start == 'top':
            trace = np.einsum('ijkl,ablk->iajb', self.nodes[0].data, other.nodes[0].data)/2
            for i in range(1,num_nodes): 
                trace = np.einsum('iajb,jklm->iakblm', trace, self.nodes[i].data)
                trace = np.einsum('iakblm,bcml->iakc', trace, other.nodes[i].data)/2
            return trace
        else: 
            trace = np.einsum('ijkl,ablk->iajb', self.nodes[num_nodes-1].data, other.nodes[num_nodes-1].data)/2
            for i in range(num_nodes-2,-1,-1):
                trace = np.einsum('iajb,hikl->jbhakl', trace, self.nodes[i].data)
                trace = np.einsum('jbhakl,calk->hcjb', trace, other.nodes[i].data)/2
            return trace
        
    def conj(self): 
        nodes = [copy.deepcopy(node).conj() for node in self.nodes]
        return MPO(nodes)
    

def random_mpo(num_qubits, bond_dim):
    return MPO([Node(np.random.rand(1,bond_dim,2,2))] + [Node(np.random.rand(bond_dim,bond_dim,2,2)) 
                                for i in range(num_qubits-2)] + [Node(np.random.rand(bond_dim,1,2,2))])
