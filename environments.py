from ansatz import *

########################
#### Stack envs ########
########################

def stack_env(ansatz, target_mpo, stack_idx, min_sv_ratio=None, max_dim=None):
    """
    let T be target unitary and we have T^dWDW^d, 
    for stack_idx = 0, this function returns W^dT^dW
    for stack_idx > 0, this function returns A,B, where Tr(T^dWDW^d) = Tr(AMBM^d) and M,M^d are the MPOs at stack_idx
    contraction order and compression ensures that returned MPOs don't have large bond dim 
    example: 
    ignoring entangling layers, suppose we have Tr(T L3 L2 L1 D L1^d L2^d L3^d)
    env1 = right environment of L2 is (L3^d T L3)
    env2 = left environment of L2 is (L1 D L1^d)
    Tr(env1 L2 env2 L2^d)
    """
    num_nodes = ansatz.num_qubits
    num_stacks = ansatz.num_stacks
    param_mpo_stacks = ansatz.param_mpo_stacks
    entanglers = [ansatz.even_mpo, ansatz.odd_mpo]
    
    def collapse(env, param_mpo, i):
        env = (entanglers[i%2] @ env @ entanglers[i%2]).compress(min_sv_ratio, max_dim)
        env = (param_mpo.conj() @ env @ param_mpo)
        env = (entanglers[i%2] @ env @ entanglers[i%2]).compress(min_sv_ratio, max_dim)
        return env

    if stack_idx == 0:
        env = param_mpo_stacks[-1].conj() @ target_mpo.conj() @ param_mpo_stacks[-1] 
        for i in range(num_stacks-2,0,-1): 
            param_mpo = param_mpo_stacks[i]
            env = collapse(env, param_mpo, i)
        return env, None

    elif stack_idx == num_stacks - 1:
        env1 = target_mpo.conj()
        env2 = param_mpo_stacks[0]
        for i in range(1,num_stacks-1): 
            param_mpo = param_mpo_stacks[i].conj()
            env2 = collapse(env2, param_mpo, i)
        return env1, env2 

    else: 
        env1 = param_mpo_stacks[-1].conj() @ target_mpo.conj() @ param_mpo_stacks[-1] 
        for i in range(num_stacks-2,stack_idx,-1): 
            param_mpo = param_mpo_stacks[i]
            env1 = collapse(env1, param_mpo, i)

        env2 = param_mpo_stacks[0]
        for i in range(1,stack_idx):
            param_mpo = param_mpo_stacks[i].conj()
            env2 = collapse(env2, param_mpo, i)
        
        env1 = (entanglers[stack_idx%2] @ env1 @ entanglers[stack_idx%2]).compress(min_sv_ratio, max_dim)
        env2 = (entanglers[stack_idx%2] @ env2 @ entanglers[stack_idx%2]).compress(min_sv_ratio, max_dim)
        return env1, env2 
    
def build_right_env(cur_right_env, new_stack, stack_idx, num_stacks, min_sv_ratio=None, max_dim=None): 
    """
    Given the right environment at stack_idx + 1, this produces the right environment of stack_idx
    stack_idx can be 0,1,...,num_stacks-2
    example: 
    For stack_idx = 1, cur_right_env = L3^d T L3, new_stack = L2, 
    returns right environment of L1 = (L2^d L3^d T L3 L2)
    """
    num_qubits = new_stack.num_nodes
    entanglers = [Entangling_MPO(num_qubits, 0), Entangling_MPO(num_qubits, 1)]
    new_env = new_stack.conj() @ cur_right_env @ new_stack 
    if stack_idx == 0:
        return (entanglers[1] @ new_env @ entanglers[1]).compress(min_sv_ratio, max_dim)
    elif stack_idx == num_stacks-2: 
        return (entanglers[0] @ new_env @ entanglers[0]).compress(min_sv_ratio, max_dim)
    elif stack_idx % 2 == 0: 
        return (entanglers[0] @ entanglers[1] @ new_env @ entanglers[1] @ entanglers[0]).compress(min_sv_ratio, max_dim)
    else: 
        return (entanglers[1] @ entanglers[0] @ new_env @ entanglers[0] @ entanglers[1]).compress(min_sv_ratio, max_dim)
    
def build_left_env(cur_left_env, new_stack, stack_idx, num_stacks, min_sv_ratio=None, max_dim=None): 
    """
    Given the left environment at stack_idx - 1, this produces the left environment of stack_idx
    stack_idx can be 1,2,...,num_stacks-1
    example: 
    For stack_idx = 3, cur_left_env = (L1 D L1^d), new_stack = L2, 
    returns left environment of L3 = (L2 L1 D L1^d L2^d) 
    """
    num_qubits = new_stack.num_nodes
    entanglers = [Entangling_MPO(num_qubits, 0), Entangling_MPO(num_qubits, 1)]
    if stack_idx == 1: 
        return (entanglers[1] @ new_stack @ entanglers[1]).compress(min_sv_ratio, max_dim)
    else:
        new_env = new_stack @ cur_left_env @ new_stack.conj()
        if stack_idx == num_stacks-1:
            return (entanglers[0] @ new_env @ entanglers[0]).compress(min_sv_ratio, max_dim)
        elif stack_idx%2==0: 
            return (entanglers[0] @ entanglers[1] @ new_env @ entanglers[1] @ entanglers[0]).compress(min_sv_ratio, max_dim)
        else:
            return (entanglers[1] @ entanglers[0] @ new_env @ entanglers[0] @ entanglers[1]).compress(min_sv_ratio, max_dim)
        
def all_stack_envs(ansatz, target_mpo, min_sv_ratio=None, max_dim=None): 
    all_envs = [stack_env(ansatz, target_mpo, i, min_sv_ratio, max_dim) for i in range(ansatz.num_stacks)]
    left_envs = [envs[1] for envs in all_envs]
    right_envs = [envs[0] for envs in all_envs]
    return left_envs, right_envs
        

######################
###### Node envs #####
######################

        
def build_bottom_env(cur_bottom_env, param_data, link_data1, link_data2=None):
    """ link_data are the nodes coming from stack_envs, param_data is the node coming from current stack """
    if link_data2 is None: 
        ''' rz stack '''
        if cur_bottom_env is not None: 
            return np.einsum('ij,hiab,kjba->hk', cur_bottom_env, link_data1, param_data)/2
        else: 
            return np.einsum('ijkl,ajlk->ia', link_data1, param_data)/2
    else: 
        ''' a general stack '''
        link_data1 = group_ind(group_ind(np.einsum('ijkl,mnlo->imjnko', link_data1, param_data), 0,1), 1,2)
        link_data2 = group_ind(group_ind(np.einsum('ijkl,mnlo->imjnko', link_data2, param_data.conj().transpose(0,1,3,2)), 0,1), 1,2)
        if cur_bottom_env is not None: 
            return np.einsum('ijxy,hiab,gjba->hgxy', cur_bottom_env, link_data1, link_data2)/2
        else: 
            return np.einsum('hiab,gjba->hgij', link_data1, link_data2)/2

def build_top_env(cur_top_env, param_data, link_data1, link_data2=None): 
    """ link_data are the nodes coming from stack_envs, param_data is the node coming from current stack """
    if link_data2 is None: 
        ''' rz stack '''
        if cur_top_env is not None: 
            return np.einsum('ij,jkab,xxba->ik', cur_top_env, link_data1, param_data)/2
        else: 
            return np.einsum('ijkl,ialk->aj', link_data1, param_data)/2
    else: 
        ''' a general stack '''
        link_data1 = group_ind(group_ind(np.einsum('ijkl,mnlo->imjnko', link_data1, param_data), 0,1), 1,2)
        link_data2 = group_ind(group_ind(np.einsum('ijkl,mnlo->imjnko', link_data2, param_data.conj().transpose(0,1,3,2)), 0,1), 1,2)
        if cur_top_env is not None: 
            return np.einsum('xyij,ikab,jlba->xykl', cur_top_env, link_data1, link_data2)/2
        else: 
            return np.einsum('ikab,jlba->ijkl', link_data1, link_data2)/2
        return None    
                                         
def node_semi_env(param_mpo, node_idx, stack_env1, stack_env2=None):
    """ Given stack_envs of param_mpo, this computes the top and bottom semi_envs of a particular node """ 
    num_nodes = param_mpo.num_nodes
    if stack_env2 is None: 
        ''' rz stack '''
        cur_bottom_env = None
        for i in range(num_nodes-1, node_idx, -1):
            cur_bottom_env = build_bottom_env(cur_bottom_env, param_mpo.nodes[i].data, stack_env1.nodes[i].data)
        cur_top_env = None 
        for i in range(node_idx): 
            cur_top_env = build_top_env(cur_top_env, param_mpo.nodes[i].data, stack_env1.nodes[i].data)
        return cur_top_env, cur_bottom_env
    
    else: 
        ''' general stack ''' 
        cur_bottom_env = None
        for i in range(num_nodes-1, node_idx, -1):
            cur_bottom_env = build_bottom_env(cur_bottom_env, param_mpo.nodes[i].data, stack_env1.nodes[i].data, stack_env2.nodes[i].data)
        cur_top_env = None 
        for i in range(node_idx): 
            cur_top_env = build_top_env(cur_top_env, param_mpo.nodes[i].data, stack_env1.nodes[i].data, stack_env2.nodes[i].data)
        return cur_top_env, cur_bottom_env                                                             
        
def full_env(link_data1, link_data2=None, top_env=None, bottom_env=None): 
    """ link_data are the nodes of stack_envs at the current node_idx """ 
    if link_data2 is not None: 
        ''' we are in a general stack ''' 
        if top_env is not None and bottom_env is not None: 
            ''' we are in the bulk of the stack ''' 
            return np.einsum('xyij,ikab,jlcd,klmn->bcda', top_env, link_data1, link_data2, bottom_env)/2
        elif top_env is None: 
            ''' we are at the top boundary of the stack '''
            return np.einsum('ijxx,hiab,hjcd->bcda', bottom_env, link_data1, link_data2)/2
        else: 
            ''' we are at the bottom boundary of the stack '''
            return np.einsum('xyij,ikab,jlcd->bcda', top_env, link_data1, link_data2)/2
        
    else: 
        ''' we are in the rz stack '''
        if top_env is not None and bottom_env is not None: 
            ''' we are in the bulk of the stack '''
            return np.einsum('ij,jklm,ki->lm', top_env, link_data1, bottom_env)/2
        elif top_env is None: 
            ''' we are at the top boundary of the stack '''
            return np.einsum('jklm,kj->lm', link_data1, bottom_env)/2
        else: 
            ''' we are at the bottom boundary of the stack '''
            return np.einsum('kj,jklm->lm', top_env, link_data1)/2

def node_env(param_mpo, node_idx, stack_env1, stack_env2=None):
    """ Given stack_envs of param_mpo, this computes the environment of a particular node inside param_mpo """ 
    top_env, bottom_env = node_semi_env(param_mpo, node_idx, stack_env1, stack_env2)
    if stack_env2 is None: 
        return full_env(stack_env1.nodes[node_idx].data, None, top_env, bottom_env)
    else: 
        return full_env(stack_env1.nodes[node_idx].data, stack_env2.nodes[node_idx].data, top_env, bottom_env)
    

def all_node_envs(param_mpo, stack_env1, stack_env2=None): 
    bottom_envs = [] 
    top_envs = []
    for node_idx in range(param_mpo.num_nodes): 
        top_env, bottom_env = node_semi_env(param_mpo, node_idx, stack_env1, stack_env2)
        bottom_envs.append(bottom_env)
        top_envs.append(top_env)
    return bottom_envs, top_envs

