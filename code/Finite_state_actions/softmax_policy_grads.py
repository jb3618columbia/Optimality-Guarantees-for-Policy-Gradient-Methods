import numpy as np
import math
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

## Random Seed
np.random.seed(10) 
## Problem Setup
gamma = 0.9
n, m = 50, 10  
'''
Randomly generated probability transition matrix P((s,a) -> s') in R^{|S||A| x |S|}
Each row sums up to one
'''
raw_transition = np.random.uniform(0,1,size=(n*m,n))
prob_transition = raw_transition/raw_transition.sum(axis=1,keepdims=1)
'''
Random positive rewards
'''
reward = np.random.uniform(0,1,size=(n*m))
'''
Start state distribution
'''
rho = np.ones(n)/n

'''
Input: theta as an array and 
Ouput: array of probabilites corresponding to each state: [\pi_{s_1}(.), ...., \pi_{s_n}(.)]
'''
def theta_to_policy(theta,n,m):
    prob = []
    for i in range(n):
        norm = np.sum(np.exp(theta[m*i:m*(i+1)]))
        for j in range(m*i,m*(i+1)):
            prob.append(np.exp(theta[j])/norm)
            
    return np.asarray(prob)


'''
Get \Pi_{\pi}((s) -> (s,a)) in R^{|S| x |S||A|} matrix corresponding to the policy \pi using the prob vector
'''
def get_Pi(prob,n,m):
    Pi = np.zeros((n,n*m))
    for i in range(n):
        Pi[i,i*m:(i+1)*m] = prob[i*m:(i+1)*m]
    
    return Pi

'''
Input: probability vector, state, action
Output: \nabla_{\theta} \pi_{\theta}(s,a)

States go from 0 to n-1 and actons from 0 to m-1
'''
def grad_state_action(prob,state,action):
    grad = np.zeros(n*m)
    for j in range(0,m):
        if j == action:
            grad[m*state + j] = prob[m*state + j]*(1-prob[m*state + j])
        else:
            grad[m*state + j] = -prob[m*state + action]*prob[m*state + j]
            
    return grad

def grad_state(qvals,prob,state):
    grad = np.sum([qvals[state*m + i]*grad_state_action(prob,state,i) for i in range(0,m)],axis=0)
    return grad

def grad(qvals,prob,d_pi):
    grad = np.sum([d_pi[i]*grad_state(qvals,prob,i) for i in range(0,n)],axis=0)
    return grad

'''
The overall reward function \ell(\theta)
'''
def ell(qvals,prob,rho):
    V = np.zeros(n)
    for i in range(n):
        V[i] = np.sum([qvals[i*m + j]*prob[i*m + j] for j in range(m)])
    
    ell = np.dot(V,rho)
    return ell

'''
Policy Iteration to get the optimal policy
'''

raw_vec = np.random.uniform(0,1,size=(n,m))
prob_vec = raw_vec/raw_vec.sum(axis=1,keepdims=1)
init_policy = prob_vec.flatten()

'''
Policy iteration function
'''
def policy_iter(q_vals,n,m):
    new_policy = np.zeros(n*m)
    for i in range(n):
        idx = np.argmax(q_vals[i*m:(i+1)*m])
        new_policy[i*m + idx] = 1
    
    return new_policy

curr_policy = np.random.uniform(0,1,size=(n*m))
new_policy = init_policy
# print('Starting policy',init_policy)

while np.count_nonzero(curr_policy - new_policy) > 0:
    curr_policy = new_policy
    Pi = get_Pi(curr_policy,n,m)
    mat = np.identity(n*m) - gamma*np.matmul(prob_transition,Pi)
    q_vals = np.dot(np.linalg.inv(mat),reward)
    new_policy = policy_iter(q_vals,n,m)
    
# print('Final policy',new_policy)

ell_star = ell(q_vals,new_policy,rho)
print('Optimal Reward',ell_star)


'''
Backtracking line search
'''

def ell_theta(theta,rho):
    prob = theta_to_policy(theta,n,m)
    Pi = get_Pi(prob,n,m)
    mat = np.identity(n*m) - gamma*np.matmul(prob_transition,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)
    return ell(qvals,prob,rho)
       
def find_step(theta,gradient,alpha,beta):
    #step = alpha
    step = 10/(np.linalg.norm(gradient))
    print('gradient norm', np.linalg.norm(gradient)**2)
    while ell_theta(theta - step*gradient,rho) > ell_theta(theta,rho) - (step/2)*(np.linalg.norm(gradient)**2):
        step = beta*step
    print('step', step)
    return step

'''
Policy gradient in action
'''
N = 25
stepsize = 0.01
# Parameters for line search
alpha = 100000
beta = 0.5
theta = np.random.uniform(0,1,size=n*m)
gap = []
div_number = 1#000
for k in range(N):
    prob = theta_to_policy(theta,n,m)

    Pi = get_Pi(prob,n,m)
    mat = np.identity(n*m) - gamma*np.matmul(prob_transition,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)

    P_theta = np.matmul(Pi,prob_transition)
    d_pi = (1-gamma)*np.dot(np.transpose((np.linalg.inv(np.identity(n) - gamma*P_theta))),rho)

    gradient = grad(qvals,prob,d_pi)
    #     theta += stepsize*gradient

    step = find_step(theta,gradient,alpha,beta)
    theta += step*gradient
    
    
    if k % div_number == 0:
        avg_reward = ell(qvals,prob,rho)
        print('Optimality gap',ell_star - avg_reward)
        gap.append(ell_star - avg_reward)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
np.save('Softmax.npy',gap)

f = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.plot(np.array(gap))
plt.yticks(fontsize=20)
plt.yticks(np.linspace(0, round(max(gap),1), 4, endpoint=True))
plt.xticks(fontsize=20)
plt.xticks(np.linspace(0, len(gap), 5, endpoint=True))
plt.ylabel('Gap',fontsize=24)
plt.xlabel('Iterations*'+ str(div_number),fontsize=24)
f.savefig("Fig_Softmax.jpg",bbox_inches='tight')
f.savefig("Fig_Softmax.pdf",bbox_inches='tight')










