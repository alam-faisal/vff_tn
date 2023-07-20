from mpo import *

def kronecker_pad(matrix, num_qubits, starting_site): 
    ''' takes a local gate described as a matrix and pads it with identity matrices to create a global operator '''
    kron_list = [np.eye(2) for i in range(num_qubits)]    
    kron_list[starting_site] = matrix
    if matrix.shape[0] == 4: 
        del kron_list[starting_site+1]
    
    padded_matrix = kron_list[0]
    for i in range(1, len(kron_list)):
        padded_matrix = np.kron(kron_list[i], padded_matrix)    
    return padded_matrix

def xy_ham(num_qubits): 
    terms = []        
    for i in range(num_qubits-1): 
        y_hop = kronecker_pad(pauli_tensor[2,2], num_qubits, i)
        terms.append(y_hop)
        x_hop = kronecker_pad(pauli_tensor[1,1], num_qubits, i)
        terms.append(x_hop)
    return sum(terms) 

def xy_gate(t):
    return expm(1.j * t * (pauli_tensor[1,1] + pauli_tensor[2,2]))    

def xy_even_layer_mpo(num_qubits, t):
    node_list = []
    for i in range(0, num_qubits, 2): 
        top_node, bottom_node = gate_to_nodes(xy_gate(t))
        node_list.append(top_node)       
        node_list.append(bottom_node)
    return MPO(node_list)

def xy_odd_layer_mpo(num_qubits, t):
    node_list = [Node(np.eye(2)[np.newaxis,np.newaxis,:,:])]
    for i in range(1, num_qubits-1, 2): 
        top_node, bottom_node = gate_to_nodes(xy_gate(t))
        node_list.append(top_node)
        node_list.append(bottom_node)
    node_list.append(Node(np.eye(2)[np.newaxis,np.newaxis,:,:]))
    return MPO(node_list)

def xy_mpo(num_qubits, t, num_trotter_layers=1, min_sv_ratio=None, max_dim=None): 
    ''' breaks down t into num_trotter_layers, creates MPO for each step, contracts, truncates '''
    single_step = (xy_odd_layer_mpo(num_qubits, t/num_trotter_layers) @\
                   xy_even_layer_mpo(num_qubits,t/num_trotter_layers)).compress(min_sv_ratio,max_dim)
    mpo = copy.deepcopy(single_step)
    for step in range(1,num_trotter_layers): 
        mpo = (single_step @ mpo).compress(min_sv_ratio, max_dim)
    
    return mpo