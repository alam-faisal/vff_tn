from tqdm import tqdm
from models import *
from environments import *
import pickle
import time

def node_cost(params, n_env, t):
    if isinstance(params, (float, np.float64)):
        angle = params
        return 1 - (n_env[0, 0] * np.exp(-1.j * angle * t / 2) + n_env[1, 1] * np.exp(1.j * angle * t / 2)).real
    else:
        param_mat = one_qubit_gate(*params)
        return 1 - np.einsum('ijkl,ij,kl->', n_env, param_mat, param_mat.conj().T).real

def rz_node_deriv(n_env, angle, t): 
    return ((1.j*t/2)*(n_env[0,0]*np.exp(-1.j*t*angle/2) - n_env[1,1]*np.exp(1.j*t*angle/2))).real

from jax import jit
import jax.numpy as jnp
@jit
def general_node_deriv(n_env, params):
    theta, lamb, phi = params
    gate = jnp.array([[jnp.cos(theta/2), -jnp.exp(1.j*lamb)*jnp.sin(theta/2)], 
                 [jnp.exp(1.j*phi)*jnp.sin(theta/2), jnp.exp(1.j*(lamb+phi))*jnp.cos(theta/2)]])

    exp_lamb_phi = jnp.exp(1.j * (lamb + phi))
    sin_theta_2 = jnp.sin(theta / 2)
    cos_theta_2 = jnp.cos(theta / 2)

    dual_theta = (1 / 2) * jnp.array([
        [sin_theta_2, jnp.exp(1.j * lamb) * cos_theta_2],
        [-jnp.exp(1.j * phi) * cos_theta_2, exp_lamb_phi * sin_theta_2]
    ])

    dual_phi = (-1.j) * jnp.array([
        [0, 0],
        [jnp.exp(1.j * phi) * sin_theta_2, exp_lamb_phi * cos_theta_2]
    ])

    dual_lamb = (-1.j) * jnp.array([
        [0, -jnp.exp(1.j * lamb) * sin_theta_2],
        [0, exp_lamb_phi * cos_theta_2]
    ])

    # Compute the derivatives using broadcasting
    result = jnp.stack((
        jnp.einsum('ijkl,ij,kl', n_env, gate, dual_theta.conj().T) +
                 jnp.einsum('ijkl,ij,kl', n_env, dual_theta, gate.conj().T),
        jnp.einsum('ijkl,ij,kl', n_env, gate, dual_lamb.conj().T) +
                 jnp.einsum('ijkl,ij,kl', n_env, dual_lamb, gate.conj().T),
        jnp.einsum('ijkl,ij,kl', n_env, gate, dual_phi.conj().T) +
                 jnp.einsum('ijkl,ij,kl', n_env, dual_phi, gate.conj().T)
    )).real

    return result
 
def node_optimizer(ansatz, stack_idx, node_idx, n_env, t, max_iter=200, eta=20, min_update=1e-3):
    """ optimizes the node at (stack_idx, node_idx) of ansatz """ 
    params = ansatz.angles[node_idx] if stack_idx == 0 else ansatz.params[stack_idx-1][node_idx*3:(node_idx+1)*3]
    
    cost_list = [node_cost(params, n_env, t)]
    for i in range(max_iter):
        derivs = rz_node_deriv(n_env, params, t) if stack_idx == 0 else np.asarray(general_node_deriv(n_env, params))
        new_params = params - eta * np.asarray(derivs)
        new_cost = node_cost(new_params, n_env, t)
        if new_cost < cost_list[-1]: 
            params = new_params 
            cost_list.append(new_cost)
        else: 
            eta = eta*(3/4)
            
        # see if I need to break
        if len(cost_list) > 10:
            if (np.abs(cost_list[-1] - cost_list[-2]))/(cost_list[0]) < min_update: 
                break
    
    ansatz.update_node(stack_idx, node_idx, new_params)
    new_node = ansatz.param_mpo_stacks[stack_idx].nodes[node_idx].data
    return cost_list[-1], new_node
   
def sweep_up_down(ansatz, stack_idx, bottom_envs, top_envs, link_datas1, link_datas2, t, max_iter=200, eta=20, min_update=1e-3):
    """ sweeps up and down the nodes in a given stack """ 
    cost_list = []
    for node_idx in range(ansatz.num_qubits-1,0,-1):
        n_env = full_env(link_datas1[node_idx], link_datas2[node_idx], top_envs[node_idx], bottom_envs[node_idx])
        new_cost, new_node = node_optimizer(ansatz, stack_idx, node_idx, n_env, t, max_iter, eta, min_update)
        cost_list.append(new_cost)
        bottom_envs[node_idx-1] = build_bottom_env(bottom_envs[node_idx], new_node, link_datas1[node_idx], link_datas2[node_idx])          
        
    for node_idx in range(ansatz.num_qubits-1):
        n_env = full_env(link_datas1[node_idx], link_datas2[node_idx], top_envs[node_idx], bottom_envs[node_idx])
        new_cost, new_node = node_optimizer(ansatz, stack_idx, node_idx, n_env, t, max_iter, eta, min_update)
        cost_list.append(new_cost)
        top_envs[node_idx+1] = build_top_env(top_envs[node_idx], new_node, link_datas1[node_idx], link_datas2[node_idx])
        
    return cost_list            
        
def nodewise_train(num_qubits, t, num_trotter_steps, num_w_layers, angles=None, params=None, min_sv_ratio=None, max_dim=None, 
        num_stack_sweeps=200, num_node_sweeps=200, max_iter=200, eta=20, min_update=1e-3, testing=True):
    
    angles = angles if angles is not None else np.zeros(num_qubits)
    params = params if params is not None else np.random.rand(2*num_w_layers+1,3*num_qubits)*2*np.pi
    ansatz = Ansatz(t, angles, params)
    target_mpo = xy_mpo(num_qubits, t, num_trotter_steps, min_sv_ratio, max_dim)
    
    noisy_cost_data = []
    time_data = []
    if testing:
        exact_target_mpo = xy_mpo(num_qubits, t, num_trotter_steps, max_dim=2**num_qubits)
        exact_cost_data = [1-exact_target_mpo.conj().mult_and_trace(ansatz.mpo())[0,0,0,0].real]

    left_envs, right_envs = all_stack_envs(ansatz, target_mpo, min_sv_ratio, max_dim)
    
    begin_time = time.time()
    for stack_sweep in range(num_stack_sweeps): 
        # left to right sweep 
        for stack_idx in tqdm(range(ansatz.num_stacks-1)): 
            stack_env1, stack_env2 = right_envs[stack_idx], left_envs[stack_idx]
            param_mpo = ansatz.param_mpo_stacks[stack_idx]
            bottom_envs, top_envs = all_node_envs(param_mpo, stack_env1, stack_env2)
            link_datas1 = [stack_env1.nodes[i].data for i in range(num_qubits)]
            link_datas2 = [None]*num_qubits if stack_env2 is None else [stack_env2.nodes[i].data for i in range(num_qubits)]
            
            starting_params = ansatz.angles[0] if stack_idx == 0 else ansatz.params[stack_idx-1][0:3]
            starting_cost = node_cost(starting_params, full_env(link_datas1[0], link_datas2[0], top_envs[0], bottom_envs[0]), t)
            for node_sweep in range(num_node_sweeps):
                cost_list = sweep_up_down(ansatz, stack_idx, bottom_envs, top_envs, link_datas1, link_datas2, t)
                if node_sweep > 5: 
                    if np.abs(cost_list[0]-cost_list[-1])/starting_cost < min_update:
                        break 
                        
            left_envs[stack_idx+1] = build_left_env(left_envs[stack_idx], ansatz.param_mpo_stacks[stack_idx], stack_idx+1, 
                                                    ansatz.num_stacks, min_sv_ratio, max_dim)
            noisy_cost_data.append(cost_list[-1])
            new_time = time.time()
            time_data.append(new_time - begin_time)
            
        # right to left sweep
        for stack_idx in range(ansatz.num_stacks-1,0,-1): 
            stack_env1, stack_env2 = right_envs[stack_idx], left_envs[stack_idx]
            param_mpo = ansatz.param_mpo_stacks[stack_idx]
            bottom_envs, top_envs = all_node_envs(param_mpo, stack_env1, stack_env2)
            link_datas1 = [stack_env1.nodes[i].data for i in range(num_qubits)]
            link_datas2 = [None]*num_qubits if stack_env2 is None else [stack_env2.nodes[i].data for i in range(num_qubits)]
            
            starting_params = ansatz.angles[0] if stack_idx == 0 else ansatz.params[stack_idx-1][0:3]
            starting_cost = node_cost(starting_params, full_env(link_datas1[0], link_datas2[0], top_envs[0], bottom_envs[0]), t)
            for node_sweep in range(num_node_sweeps):
                cost_list = sweep_up_down(ansatz, stack_idx, bottom_envs, top_envs, link_datas1, link_datas2, t)
                if node_sweep > 5: 
                    if np.abs(cost_list[0]-cost_list[-1])/starting_cost < min_update:
                        break      
                        
            right_envs[stack_idx-1] = build_right_env(right_envs[stack_idx], ansatz.param_mpo_stacks[stack_idx], stack_idx-1, 
                                                      ansatz.num_stacks, min_sv_ratio, max_dim)
            noisy_cost_data.append(cost_list[-1])
            new_time = time.time()
            time_data.append(new_time - begin_time)
            
        if testing:
            exact_cost = 1-exact_target_mpo.conj().mult_and_trace(ansatz.mpo())[0,0,0,0].real
            exact_cost_data.append(exact_cost)
            #print(exact_cost)
        
    if testing: 
        return noisy_cost_data, exact_cost_data, time_data, ansatz.angles, ansatz.params
    else:
        return noisy_cost_data, time_data, ansatz.angles, ansatz.params


def run_training(output_file, num_qubits, t, num_trotter_steps, num_w_layers, angles=None, params=None, min_sv_ratio=None, max_dim=None, 
        num_stack_sweeps=200, num_node_sweeps=200, max_iter=200, eta=20, min_update=1e-3, testing=True):
    
    result = nodewise_train(num_qubits, t, num_trotter_steps, num_w_layers, angles, params, min_sv_ratio, max_dim, num_stack_sweeps,
                            num_node_sweeps, max_iter, eta, min_update, testing)
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
        
        

###############################################################################
############################ Debugging methods ################################
###############################################################################
delta = 0.0001
def delta_matrix(shape, i, j=None): 
    ''' j represents which gate for params_array '''
    a = np.zeros(shape)
    if j is not None: 
        a[i][j] = delta
    else: 
        a[i] = delta
    return a
    
def node_fd(params, i, n_env, t=None): 
    for_params = params + delta_matrix(params.shape, i) 
    back_params = params - delta_matrix(params.shape, i) 
    if len(params) == 1: 
        for_params = for_params[0]
        back_params = back_params[0]
    return (node_cost(for_params, n_env, t) - node_cost(back_params, n_env, t)) / (2*delta) 

def node_grad_fd(params, n_env, t=None):
    if type(params) == float or type(params) == np.float64: 
        params = np.array([params])
    return np.array([node_fd(params, i, n_env, t) for i in range(len(params))])

##################################################################################
        
        
'''
def node_cost(params, n_env, t=None):
    if type(params) == float or type(params) == np.float64:
        return 1-(n_env[0,0]*np.exp(-1.j*angle*t/2) + n_env[1,1]*np.exp(1.j*angle*t/2)).real 
    else:
        param_mat = one_qubit_gate(*params)
        return 1-np.einsum('ijkl,ij,kl', n_env, param_mat, param_mat.conj().T).real
        
def theta_dual(theta, lamb, phi): 
    return (1/2) * np.array([[np.sin(theta/2), np.exp(1.j*lamb)*np.cos(theta/2)], 
                 [-np.exp(1.j*phi)*np.cos(theta/2), np.exp(1.j*(lamb+phi))*np.sin(theta/2)]])

def phi_dual(theta, lamb, phi): 
    return (-1.j) * np.array([[0, 0], 
                 [np.exp(1.j*phi)*np.sin(theta/2), np.exp(1.j*(lamb+phi))*np.cos(theta/2)]])

def lamb_dual(theta, lamb, phi): 
    return (-1.j) * np.array([[0, -np.exp(1.j*lamb)*np.sin(theta/2)], 
                 [0, np.exp(1.j*(lamb+phi))*np.cos(theta/2)]])
    
def general_node_deriv(n_env, params): 
    gate = one_qubit_gate(*params)
    duals = [theta_dual(*params), lamb_dual(*params), phi_dual(*params)]
    return np.array([np.einsum('ijkl,ij,kl', n_env, gate, dual.conj().T) + np.einsum('ijkl,ij,kl', n_env, dual, gate.conj().T) 
                     for dual in duals]).real
'''

