'''
Set-up:
1) Threshold policy 
2) Bounded rewards
3) Many contextx and finite offers 
4) Classic policy gradient with soft threshold as defined in the paper
'''

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
n, m = 10, 50
'''
Q matrix: Emission probabilities Q(y|x) in R^{|X| x |Y|}
'''
raw_em = np.random.uniform(0,1,size=(n,m))
prob_em = raw_em/raw_em.sum(axis=1,keepdims=1)
'''
Latent transitions in R^{|X| x |X|}
'''
raw_latent = np.random.uniform(0,1,size=(n,n))
prob_latent = raw_latent/raw_latent.sum(axis=1,keepdims=1)

'''
Random positive offers in R^{|Y|}
'''
offer = np.random.uniform(0,1,size=(m))
'''
Reward
'''
reward = np.zeros(2*n*m)
for j in range(m):
    for i in range(n):
        reward[2*(i*m + j)+1] = offer[j]
'''
Start state distribution
'''
rho = prob_em/n

'''
Probability transition matrix P((s,a) -> s') in R^{|X||Y|*|A| x |X||Y|} with an absorbing state
Each row sums up to one
Note, the transitons (x,y) -> (x',y') do not depend on y, so there is a block structure
'''
def get_prob_trans(prob_em,prob_latent,n,m):
    prob_trans = np.zeros((2*n*m,n*m))
    
    ## This is for action 0 which transitions between states; no transition to the absorbing state
    for i in range(n):
        prob_0 = np.zeros(n*m)
        for j in range(n):
            for k in range(m):
                prob_0[j*m + k] = prob_latent[i,j]*prob_em[j,k]
                
        for J in range(i*m,(i+1)*m):
            prob_trans[2*J,:] = prob_0
    
    return prob_trans

prob_trans = get_prob_trans(prob_em,prob_latent,n,m)

## Theta to probability vector
'''
Input: theta as an array
Ouput: array of probabilites corresponding to each (state,action): [\pi_{(x,y),a}] in R^{|X||Y|*|A|}
'''
def theta_to_policy(theta,n,m):
    prob = np.zeros(2*n*m)
    for i in range(n):
        for j in range(m):
            prob_acc = 1/(1 + np.exp(-(theta[2*i] + theta[2*i + 1]*offer[j])))
            prob[2*(i*m + j) + 1] = prob_acc  ## Action = 1
            prob[2*(i*m + j)] = 1 - prob_acc  ## Action = 0
            
    return prob

## Forming the Pi_{\pi} matrix
'''
Get \Pi_{\pi}((s) -> (s,a)) in R^{|X||Y| x |X||Y|*|A|} matrix corresponding to the policy \pi using the prob vector
'''
def get_Pi(prob,n,m):
    Pi = np.zeros((n*m,2*n*m))
    for i in range(n*m):
        Pi[i,2*i:2*(i+1)] = prob[2*i:2*(i+1)]
    
    return Pi

## Getting gradients
'''
Input: probability vector, state, action
Output: \nabla_{\theta} \pi_{\theta}(s,a)

States go from 0 to n-1 and actons from 0 to m-1
'''
def grad_state_xy(qvals,prob,state_x,state_y):
    grad = np.zeros(2*n)
    Q_s_0 = qvals[2*(state_x*m + state_y) + 1]
    Q_s_1 = qvals[2*(state_x*m + state_y)]
    pi_s_1 = prob[2*(state_x*m + state_y) + 1]
    grad[2*state_x] = (Q_s_0 - Q_s_1)*pi_s_1*(1-pi_s_1)
    grad[2*state_x + 1] = (Q_s_0 - Q_s_1)*pi_s_1*(1-pi_s_1)*offer[state_y]
    return grad

def grad_state(qvals,prob,d_pi,state_x):
    grad = np.sum([d_pi[state_x*m + j]*grad_state_xy(qvals,prob,state_x,j) for j in range(m)],axis=0)        
    return grad

def grad(qvals,prob,d_pi):
    grad = np.sum([grad_state(qvals,prob,d_pi,i) for i in range(n)],axis=0)
    return grad

## Overall reward for any parameter value
'''
The overall reward function \ell(\theta)
'''
def ell(qvals,prob,rho):
    V = np.zeros(n*m)
    for i in range(n*m):
        V[i] = np.sum([qvals[2*i + j]*prob[2*i + j] for j in range(2)])
    
    ell = np.dot(V,rho.flatten())
    return ell


## Backtracking line search
def ell_theta(theta,rho):
    prob = theta_to_policy(theta,n,m)
    Pi = get_Pi(prob,n,m)
    mat = np.identity(2*n*m) - gamma*np.matmul(prob_trans,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)
    return ell(qvals,prob,rho)
       
def find_step(theta,gradient,alpha,beta):
    step = 1500
    print('initial step', step)
    #print('gradient norm', (np.linalg.norm(gradient)**2))
    while ell_theta(theta - step*gradient,rho) > ell_theta(theta,rho) - (step/2)*(np.linalg.norm(gradient)**2):
        step = beta*step
    print('end step', step)
    return step

## Policy Iteration to find the optimal policy
raw_vec = np.random.uniform(0,1,size=(n*m,2))
prob_vec = raw_vec/raw_vec.sum(axis=1,keepdims=1)
init_policy = prob_vec.flatten()
'''
Policy iteration function
'''
def policy_iter(q_vals,n,m):
    new_policy = np.zeros(2*n*m)
    for i in range(n*m):
        idx = np.argmax(q_vals[2*i:2*(i+1)])
        new_policy[2*i + idx] = 1
    
    return new_policy

curr_policy = np.random.uniform(0,1,size=(2*n*m))
new_policy = init_policy

while np.count_nonzero(curr_policy - new_policy) > 0:
    curr_policy = new_policy
    Pi = get_Pi(curr_policy,n,m)
    mat = np.identity(2*n*m) - gamma*np.matmul(prob_trans,Pi)
    q_vals = np.dot(np.linalg.inv(mat),reward)
    new_policy = policy_iter(q_vals,n,m)

    print(np.count_nonzero(curr_policy - new_policy))

ell_star = ell(q_vals,new_policy,rho)
print('Optimal Reward',ell_star)


## Policy gradients
'''
Gradient decent
'''
N = 1000#00
stepsize = 0.01
# Parameters for line search
alpha = 1
beta = 0.5
theta = np.random.uniform(0,1,size=2*n)
gap = []
div_number = 1#000
for k in range(N):
    prob = theta_to_policy(theta,n,m)

    Pi = get_Pi(prob,n,m)
    mat = np.identity(2*n*m) - gamma*np.matmul(prob_trans,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)

    P_theta = np.matmul(Pi,prob_trans)
    d_pi = (1-gamma)*np.dot(np.transpose((np.linalg.inv(np.identity(n*m) - gamma*P_theta))),rho.flatten())

    gradient = grad(qvals,prob,d_pi)
    #     theta += stepsize*gradient
    
    step = find_step(theta,gradient,alpha,beta)
    theta += step*gradient
        
    if k % div_number == 0:
        avg_reward = ell(qvals,prob,rho)
        print('Optimality gap',ell_star - avg_reward)
        gap.append(ell_star - avg_reward)

## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
np.save('Asset_Selling.npy',gap)

f = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.plot(np.array(gap))
plt.yticks(fontsize=20)
plt.yticks(np.linspace(0, round(max(gap),1), 4, endpoint=True))
plt.xticks(fontsize=20)
plt.xticks(np.linspace(0, len(gap), 5, endpoint=True))
plt.ylabel('Gap',fontsize=24)
plt.xlabel('Iterations*'+ str(div_number),fontsize=24)
f.savefig("Fig_Asset_Selling.jpg",bbox_inches='tight')
f.savefig("Fig_Asset_Selling.pdf",bbox_inches='tight')

