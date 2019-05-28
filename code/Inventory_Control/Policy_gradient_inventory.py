'''
Set-up:
1) Threshold policy 
2) Continuous state space and finite horizon 
3) Monte Carlo approximation
4) Approximate DP for optimal policy
'''

import numpy as np
import math
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

'''
Parameters: norizon length, demand distribution, start state distribution
'''
np.random.seed(10)
horizon = 25
demand_min = 0
demand_max = 1
start_mean = 1
start_var = 0.5
'''
Costs
'''
c = 0.5
p = 2#0.6
h = 0.55
'''
Monte Carlo approximation and golden search
'''
sims_for_MC = 500
y_min = -1
y_max = 1

'''
Common for approximate DP and gradient approximation.
'''
demand_mat = np.random.uniform(demand_min,demand_max,size=(sims_for_MC,horizon))

def get_cost(state,action):
    return c*action + h*max(0,state) + p*max(0,-state)


## Monte Carlo Approximation for Qfunction
'''
horizon = length if the decision horizon. Example, horizon = 3 menas decision in 0,1,2
curr_time = current_period, so can go from 0 to horizon-1
Inefficient code in terms of memory but easy to understand indexing.
Efficient code will create vectors of size horizon - curr_time + 1 but requires more complex indexing
'''

def q_function(sims_for_MC,y_init,theta_star,curr_time,horizon):
    
    total_cost = 0
    for k in range(sims_for_MC):
    
        '''
        The +1's in length is to accomodate the cost in the state after the final decison is taken
        '''
        variable = np.zeros(horizon+1)
        cost = np.zeros(horizon+1)
        variable[curr_time] = y_init
        cost[curr_time] = c*variable[curr_time] 
        action = 0

        '''
        Now computing the continuation costs, J_{h+1} by forward simulation
        '''
        for i in range(curr_time,horizon-1): 
            variable[i+1] = variable[i] + action - demand_mat[k,i]
            action = max(0,theta_star[i+1]-variable[i+1])
            cost[i+1] = get_cost(variable[i+1],action) 

        variable[-1] = variable[horizon-1] + action - demand_mat[k,-1]
        cost[-1] = get_cost(variable[-1],0)
        total_cost += np.sum(cost)
    
    return total_cost/sims_for_MC

## Golden Search
'''
Wikipedia Implementation
'''
gr = (math.sqrt(5) + 1) / 2
def gss(a, b, curr_time, tol=1e-2):
    '''
    golden section search to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]
    '''
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    
    while abs(c - d) > tol:
        f_c = q_function(sims_for_MC,c,theta_star,curr_time,horizon)
        f_d = q_function(sims_for_MC,d,theta_star,curr_time,horizon)
        if f_c < f_d:
            b = d
        else:
            a = c

        # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        
    return (b + a) / 2

## Approximate DP
theta_star = np.zeros(horizon)
for i in range(horizon-1,-1,-1):
    theta_star[i] = gss(y_min,y_max,i,tol=0.001)


## Approximate Optimal Cost
def ell_approx(theta):
    
    total_cost = 0
    for k in range(sims_for_MC):
        '''
        Simulation part
        '''
        state = np.zeros(horizon+1)
        cost = np.zeros(horizon+1)
        action = np.zeros(horizon)

        state[0] = np.random.normal(start_mean,start_var)
        for i in range(0,horizon):
            action[i] = max(0,theta[i] - state[i])
            cost[i] = c*action[i] + h*max(0,state[i]) + p*max(0,-state[i])
            state[i+1] = state[i] + action[i] - demand_mat[k,i]

        cost[-1] = h*max(0,state[-1]) + p*max(0,-state[-1])
        total_cost += np.sum(cost)
    
    return total_cost/sims_for_MC

ell_star = ell_approx(theta_star)
print('Approximate Optimal cost',ell_star)

## Policy Gradient Approximation
def policy_grad_simulate(theta,horizon):
    '''
    Simulation part
    '''
    state = np.zeros(horizon+1)
    cost = np.zeros(horizon+1)
    action = np.zeros(horizon)
    demand = np.random.uniform(demand_min,demand_max,size=(horizon))
    du_dtheta = np.zeros(horizon+1)
    tau = np.zeros(horizon)
    
    state[0] = np.random.normal(start_mean,start_var)
    for i in range(0,horizon):
        action[i] = max(0,theta[i] - state[i])
        cost[i] = c*action[i] + h*max(0,state[i]) + p*max(0,-state[i])
        state[i+1] = state[i] + action[i] - demand[i]
    
    cost[-1] = h*max(0,state[-1]) + p*max(0,-state[-1])
    
    '''
    Gradient computation
    '''
        
    du_dtheta = np.sign(action)
    dr = h*(state>0) - p*(state<0)
    
    grad = np.zeros(horizon)
    idx = (np.argwhere(du_dtheta > 0)).flatten()
    for i in range(len(idx)-1):
        grad[idx[i]] = np.sum(dr[idx[i]+1:idx[i+1]+1])
    
    if len(idx) > 0:
        grad[idx[-1]] = c + np.sum(dr[idx[-1]+1:])
    return np.sum(cost), grad

## Policy Gradient in Action
'''
Gradient decent
'''
N = 1000
num_sims = 500
stepsize = 0.005
theta = np.random.uniform(0,1,size=(horizon))
gap = []
div_number = 50
for k in range(N):
    
    ell = 0
    gradient = np.zeros(horizon)
    for _ in range(sims_for_MC):
        cost, grad = policy_grad_simulate(theta,horizon)
        gradient += grad
        ell += cost
        
#     print(gradient)
    theta -= stepsize*gradient/sims_for_MC
    
    if k % div_number == 0:
        print('Average cost', ell/sims_for_MC)
        optimality_gap = np.abs(ell_approx(theta) - ell_star)
        print('Optimality gap',optimality_gap)
        gap.append(optimality_gap)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
np.save('Inv_control.npy',gap)

f = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.plot(np.array(gap))
plt.yticks(fontsize=20)
plt.yticks(np.linspace(0, round(max(gap),1), 4, endpoint=True))
plt.xticks(fontsize=20)
plt.xticks(np.linspace(0, len(gap), 5, endpoint=True))
plt.ylabel('Gap',fontsize=24)
plt.xlabel('Iterations*'+ str(div_number),fontsize=24)
f.savefig("Fig_Inventory_Control.jpg",bbox_inches='tight')
f.savefig("Fig_Inventory_Control.pdf",bbox_inches='tight')
