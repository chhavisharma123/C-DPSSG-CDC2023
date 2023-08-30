#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random as rd
import networkx as nx
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.datasets import load_svmlight_file
from random import choice
from scipy.stats import bernoulli
from numpy.linalg import eig
import scipy


# # Loading data set

# In[ ]:


data_features, data_labels = load_svmlight_file("a4a.txt")
"""
converting sparse matrix myfeature (which is in tuple form) to a dense matrix
"""
A_dense = data_features.todense()

A_dense = np.array(A_dense)

print (type(A_dense))
print (type(data_labels))

num_samples = A_dense.shape[0]
num_features = A_dense.shape[1]
print (data_labels)
print('num samples: %d num features : %d ' %(num_samples, num_features))


# # Shuffling the indices

# In[ ]:


np.random.seed(1234)
indices = np.arange(num_samples)
np.random.shuffle(indices)

A = A_dense[indices]
b = data_labels[indices]


print ('shape of A',A.shape)
print ('shape of b',b.shape)

print ('Minimum feature value =',np.min(A))
print ('Maximum feature value =',np.max(A))


# # Distribute data points among nodes

# In[ ]:


def data_blocks(A,b,m):
    "A = feature matrix, b = target values, m = number of nodes"
    features = [[] for i in range(m)]
    values = [[] for i in range(m)]
    N = A.shape[0] ## N = number of samples
    indices = np.arange(N)
    size = int(N/m)  ## size of each block
    for i in range(m):
        start_idx = i*size
        end_idx = min(N,(i+1)*size)
        features[i] = A[indices[start_idx:end_idx]]
        values[i] = b[indices[start_idx:end_idx]]
    
    samples_after_eq_dist = m*int(N/m)  ## total samples after equally distributed samples to every node
    remaining_samples = N - m*int(N/m) ## This quantity will always be less than m
    if ( remaining_samples >= 1):
        for j in range(remaining_samples):
            features[j] = np.vstack((features[j],A[samples_after_eq_dist + j]))
            values[j] = np.hstack((values[j],b[samples_after_eq_dist + j]))

    return features,values


# In[ ]:


## creating n mini-batches of local data points
def create_mini_batches(local_features,local_values,num_batches):
    "num_batches = number of mini-batches"
    batch_features = [[] for i in range(num_batches)]
    batch_values = [[] for i in range(num_batches)]
    samples = len(local_features)
    indices = np.arange(samples)
    batch_size = int(samples/num_batches)
    for batch in range(num_batches):
        start_idx = batch*batch_size
        end_idx = min(samples,(batch+1)*batch_size)
        batch_features[batch] = local_features[start_idx:end_idx]
        batch_values[batch] = local_values[start_idx:end_idx]
    samples_after_eq_dist = num_batches*int(samples/num_batches)
    remaining_samples = samples - samples_after_eq_dist ## This quantity will always be less than m
    if ( remaining_samples >= 1):
        for j in range(remaining_samples):
            batch_features[j] = np.vstack((batch_features[j],local_features[samples_after_eq_dist+j])) ## remove indices
            batch_values[j] = np.hstack((batch_values[j],local_values[samples_after_eq_dist+j]))
     
    return batch_features,batch_values


# In[ ]:


def nodes_mini_batches(features,values,nodes,num_batches):
    "features , values = local data points of all nodes "
    mini_batch_features = [[] for i in range(nodes)]
    mini_batch_values = [[] for i in range(nodes)]
    shuffled_features = [[] for i in range(nodes)]
    shuffled_values = [[] for i in range(nodes)]
    for i in range(nodes):
        "shuffling local samples befor creating mini-batches"
        local_num_samples = features[i].shape[0]
        indices = np.arange(local_num_samples)
        np.random.seed(i+10) ## For reproducimg same minibatches for comparative algorithms
        np.random.shuffle(indices)
        shuffled_features[i] = features[i][indices]
        shuffled_values[i] = values[i][indices]
        mini_batch_features[i],mini_batch_values[i] = create_mini_batches(shuffled_features[i]
                                                                          ,shuffled_values[i],
                                                                        num_batches)
        
    return mini_batch_features, mini_batch_values


# In[ ]:


## Finding Lipschitz parameters for f_ij(x,y)
def local_lipschitz_constants(batch_features,batch_values,radius_x,radius_y,regcoef_x,regcoef_y
                                 ,scaling_factor):
    
    "This function return the Lipschitz parameter of f_ij(x,y)"
    
    number_samples = len(batch_features) ## batch size
    A_norm = LA.norm(batch_features ,2)
    Lxx_tilde = 0.25*(2*A_norm**2 + 2*number_samples*radius_y**2)
    Lxx = (num_batches/scaling_factor)*Lxx_tilde + regcoef_x/nodes
    Lyy_tilde = 0.25*number_samples*radius_x**2
    Lyy =  (num_batches/scaling_factor)*Lyy_tilde + regcoef_y/nodes
    feature_norm_sum = sum(LA.norm(batch_features, axis=1))
    Lxy_tilde = (1+ 0.25*radius_x*radius_y)*number_samples + 0.25*radius_x*feature_norm_sum
    Lxy = (num_batches/scaling_factor)*Lxy_tilde
    Lyx = Lxy
    
    return Lxx, Lyy, Lxy, Lyx



    


# In[ ]:


def global_lipschitz(mini_batch_features,mini_batch_values,radius_x,radius_y,nodes
                    ,num_batches,regcoef_x,regcoef_y,scaling_factor):
    Lxx_batch = np.zeros(num_batches)
    Lyy_batch = np.zeros(num_batches)
    Lxy_batch = np.zeros(num_batches)
    Lyx_batch = np.zeros(num_batches)
    Lxx_nodes = np.zeros(nodes)
    Lyy_nodes = np.zeros(nodes)
    Lxy_nodes = np.zeros(nodes)
    Lyx_nodes = np.zeros(nodes)
    for i in range(nodes):
        for j in range(num_batches):
            Lxx_batch[j],Lyy_batch[j],Lxy_batch[j],Lyx_batch[j] = local_lipschitz_constants(
                                            mini_batch_features[i][j],mini_batch_values[i][j]
                                            ,radius_x, radius_y,regcoef_x,regcoef_y,
                                                scaling_factor)
            
        Lxx_nodes[i] = np.max(Lxx_batch)
        Lyy_nodes[i] = np.max(Lyy_batch)
        Lxy_nodes[i] = np.max(Lxy_batch)
        Lyx_nodes[i] = np.max(Lyx_batch)
        
    return np.max(Lxx_nodes), np.max(Lyy_nodes),np.max(Lxy_nodes), np.max(Lyx_nodes)   
    


# In[ ]:


"""
logistic function value computed at (x,y)
"""

def logistic_func(A,b,x,y,nodes,num_batches,regcoef_x,regcoef_y,scaling_factor): 
    func = 0
    N = A.shape[0] ## number of samples
    for i in range(N):
        perturbed_feature = np.add(A[i],y)
        product = np.dot(x,perturbed_feature)
        domain = b[i]*product
        if (domain > 0):
            num = 1+ math.exp((-1)*domain)
            log_num = math.log(num)
            func = func + log_num 
        else:
            num = 1+ math.exp(domain)
            log_num = (-1)*domain + math.log(num)
            func = func + log_num  
            
         
    function = func/scaling_factor + 0.5*regcoef_x*(LA.norm(x))**2 - 0.5*regcoef_y*(LA.norm(y))**2

    return function


# In[ ]:


## This function returns the gradient of f_i(x,y) with respect to x and y.

def full_batch_grad(A,b,regcoef_x,regcoef_y,nodes,num_batches,x,y,scaling_factor):
    grad_x = np.zeros(A.shape[1])
    grad_y = np.zeros(A.shape[1])
    local_samples = A.shape[0]
    for i in range(local_samples):
        perturbed_feature = np.add(A[i],y) ## a_i + y
        product = np.dot(x,perturbed_feature) ## xT(a_i +y)
        domain = b[i]*product ## bi*xT(a_i +y)
        if (domain > 0): ## To avoid math flow error
            rec_exp = 1 - 1/(1 + math.exp((-1)*domain)) ## 1/(1+exp^(bi*xT(a_i +y)))

        else:
            rec_exp = 1/(1 + math.exp(domain))

            
        scalar = (-1)*b[i]*rec_exp
        ratio_x = scalar*perturbed_feature
        grad_x = np.add(grad_x,ratio_x)
        
        "gradient w.r.t y"
        ratio_y = scalar*x
        grad_y = np.add(grad_y,ratio_y)
     
    
    loss_der_x = (1/scaling_factor)*grad_x
    
    gradient_x = np.add(loss_der_x,(regcoef_x/nodes)*x)
    
    loss_der_y = (1/scaling_factor)*grad_y
    
    gradient_y = np.subtract(loss_der_y,(regcoef_y/nodes)*y)
    
    
    return gradient_x , gradient_y
    
    


# In[ ]:


## This function returns the gradient of f_{ij}(xy) with respect to x and y.

def mini_batch_grad(A,b,regcoef_x,regcoef_y,nodes,x,y,scaling_factor): 
    
    "A = mini-batch features"
    "b = mini-batch labels"
    batch_size = A.shape[0]
    grad_x = np.zeros(A.shape[1])
    grad_y = np.zeros(A.shape[1])
    for i in range(batch_size):
        perturbed_feature = np.add(A[i],y) ## a_i + y
        product = np.dot(x,perturbed_feature) ## xT(a_i +y)
        domain = b[i]*product ## bi*xT(a_i +y)
        if (domain > 0): ## To avoid math flow error
            rec_exp = 1 - 1/(1 + math.exp((-1)*domain)) ## 1/(1+exp^(bi*xT(a_i +y)))
            #print (num1)
        else:
            rec_exp = 1/(1 + math.exp(domain))
            
        scalar = (-1)*b[i]*rec_exp
        ratio_x = scalar*perturbed_feature
        grad_x = np.add(grad_x,ratio_x)
        
        "gradient w.r.t y"
        ratio_y = scalar*x
        grad_y = np.add(grad_y,ratio_y)
        
    loss_der_x = (num_batches/scaling_factor)*grad_x
    gradient_x = np.add(loss_der_x,(regcoef_x/nodes)*x)
    
    loss_der_y = (num_batches/scaling_factor)*grad_y
    gradient_y = np.subtract(loss_der_y,(regcoef_y/nodes)*y)
    
    
    return gradient_x, gradient_y


# In[ ]:


## Defining SGD oracle
def SGD_oracle(batches_features,batches_values,sample_prob ,nodes,num_batches,x,y,
               regcoef_x,regcoef_y,scaling_factor):
    
    "sample_prob = probability of sampling a mini-batch"
    "batches_features , batches_values = batches of a particular node"
    indices = np.arange(num_batches)
    sampled_batch_list = np.random.choice(indices,1,p = sample_prob) ## sampling a minibatch using sample_prob distribution
    sampled_batch = sampled_batch_list[0]

    mini_batch_f = batches_features[sampled_batch] 
    mini_batch_val = batches_values[sampled_batch] 
 
    stoch_grad_x, stoch_grad_y = mini_batch_grad(mini_batch_f,mini_batch_val,
                                     regcoef_x,regcoef_y,nodes,x,y,scaling_factor)
        
    return stoch_grad_x, stoch_grad_y 


# In[ ]:


"This function returns the stochastic gradient estimator and latest reference point for a particular node"

def svrg_oracle(features,values,batches_features,batches_values,sample_prob
                   ,omega,num_batches,ref_point_x,ref_point_y,FB_grad_x, FB_grad_y,x,y
                    ,scaling_factor):
    
    "sample_prob = probability of sampling a mini-batch"
    "ref_prob = probability with which refrence point is updated"
    "features, values = full batch features and values of a given node"
    "batches_features , batches_values = batches of a particular node"
    indices = np.arange(num_batches)
    sampled_batch_list = np.random.choice(indices,1,p = sample_prob) ## sampling a minibatch using sample_prob distribution
    sampled_batch = sampled_batch_list[0]
#     print ('sampled_batch',sampled_batch)
    mini_batch_f = batches_features[sampled_batch] 
    mini_batch_val = batches_values[sampled_batch] 
    ## computes gradient of fij at (x,y)
    grad_x, grad_y = mini_batch_grad(mini_batch_f,mini_batch_val,
                                     regcoef_x,regcoef_y,nodes,x,y,scaling_factor)
    
    ## computes gradient at (ref_point_x,ref_point_y)
    grad_ref_x, grad_ref_y = mini_batch_grad(mini_batch_f,mini_batch_val,regcoef_x,
                                             regcoef_y,nodes,ref_point_x,ref_point_y,scaling_factor)
    
    # grad_x f_il(x,y) - grad_x f_il(ref_x,ref_y)
    grad_x_diff = np.subtract(grad_x, grad_ref_x)
    ## computing 1/np_il * (grad_x f_il(x,y) - grad_x f_il(ref_x,ref_y))
    scale_grad_x_diff = 1/(num_batches*sample_prob[sampled_batch])*grad_x_diff
    stoch_grad_x = np.add(scale_grad_x_diff, FB_grad_x)
    
    ## computing stochastic gradient with respect to y
    
    # grad_y f_il(x,y) - grad_y f_il(ref_x,ref_y)
    grad_y_diff = np.subtract(grad_y, grad_ref_y)
    ## computing 1/np_il * (grad_y f_il(x,y) - grad_y f_il(ref_x,ref_y))
    scale_grad_y_diff = 1/(num_batches*sample_prob[sampled_batch])*grad_y_diff
    stoch_grad_y = np.add(scale_grad_y_diff, FB_grad_y)
    
#     "updating reference point"
    if (omega == 1):
        ref_point_x = np.copy(x)
        ref_point_y = np.copy(y)
        ## computes full batch gradient at new (ref_point_x,ref_point_y)
        FB_grad_x, FB_grad_y = full_batch_grad(features,values,regcoef_x,
                                           regcoef_y,nodes,num_batches,ref_point_x,ref_point_y
                                              , scaling_factor)
        
    return stoch_grad_x, stoch_grad_y,FB_grad_x, FB_grad_y, ref_point_x,ref_point_y 
    


# In[ ]:


"""
This function returns the weighted sum of x_1, x_2, ...,x_m where x = [x_1,...,x_m]
"""

def oneConsensus(W,nodes,x):
    v = np.zeros((nodes,len(x[0])))
    for i in range(nodes):
        u = [W[i][j]*np.array(x[j]) for j in range(nodes)]
        u = np.array(u)
        v[i] = u.sum(axis = 0)
        
    return v


# In[ ]:


## faster AccGossip. Fast for higher dimensions
## x = [x1,x2,....,xm], m-dimensional vector

def acce_consensus(W,m,eta,tau,x):
    
    v = np.zeros(nodes)

    x_new1 = np.copy(x)
    x_old1 = np.copy(x)

    for t in range(int(tau)):
        x_old2 = np.copy(x_old1)   ## z_k,t-1
        x_old1 = np.copy(x_new1)   ## z_k,t
        for i in range(m):
            v[i] = np.dot(W[i],x_old1)
        first_term = (1+eta)*v
        sec_term = eta*x_old2
        
        x_new1 = np.subtract(first_term, sec_term)  ## z_{k,t+1}
    return x_new1


# In[ ]:


"""
Projection of point v onto l2 ball = (radius*v)/max(radius,||v||)
"""

def projection_L2ball(v,radius):
    norm = LA.norm(v)
    if (norm <= radius):
        return v
    else:
        scaling = radius/norm
        projection = scaling*v
        return projection


# In[ ]:


"This function returns the sum of difference between xt ,xstar and yt, ystar"
def distance_from_saddle(x,y,xstar,ystar,nodes):
    dist_xi_xstar = 0
    dist_yi_ystar = 0
    for i in range(nodes):
        diff_xi = np.subtract(x[i],xstar)
        dist_xi_xstar += (LA.norm(diff_xi))**2
        diff_yi = np.subtract(y[i],ystar)
        dist_yi_ystar += (LA.norm(diff_yi))**2
    total_distance = dist_xi_xstar +  dist_yi_ystar   
    return total_distance


# In[ ]:


def local_distances(x,y,x0,y0):
#     distances = np.zeros(nodes)
    diff_x = np.subtract(x,x0)
    norm_x = LA.norm(diff_x,axis = 1) ## computes norm of each x[i] - x0[i]
    square_norm_x = np.square(norm_x) ## takes elementwise square of ||x[i] - x0[i]||
    
    diff_y = np.subtract(y,y0)
    norm_y = LA.norm(diff_y, axis = 1) ## computes norm of each y[i] - y0[i]
    square_norm_y = np.square(norm_y) ## takes elementwise square of ||y[i] - y0[i]||
    
    distances = np.add(square_norm_x, square_norm_y) ## ith entry: ||x[i] - x0[i]||^2 + ||y[i] - y0[i]||^2
    
    return distances
    


# In[ ]:


## compression operator

def qsgd_quantize(x, num_bits): 
    bits = 2**(num_bits - 1)
    norm = LA.norm(x, np.inf)
    if (norm <= 10**(-15)): ## if x is zero vector
        return x
    else:
        level_float = bits * np.abs(x) / norm
#         print ('level_float',level_float)
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
#         print ('level_float - previous_level',level_float - previous_level)
#         print ('is_next_level',is_next_level)
        new_level = previous_level + is_next_level
#         print ('previous_level',previous_level)
#         print ('new_level',new_level)
        return np.sign(x) * norm * new_level / bits


# In[ ]:


def delta_qsgd(num_bits,dimension):
    
    bit_rep = 2**(num_bits - 1)
    
    ratio = 1 + min(dimension/(bit_rep**2), math.sqrt(dimension)/bit_rep)
    
    return 1 - 1/ratio


# In[ ]:


# This function compresses x_1,..x_m and y_1,...,y_m.
# Then compressed vectors are communicated among nodes to get WQ(x).

def COMM(num_bits,nu_x,H_x,Hw_x,alpha,W,nodes):
    ## initializing Q_x, Q_y
    Q_x = np.copy(nu_x) 
    for i in range(nodes):
        diff_x = np.subtract(nu_x[i],H_x[i])
        Q_x[i]= qsgd_quantize(diff_x, num_bits)
 
    nu_hat_x = np.add(H_x,Q_x) 
    
    new_H_x = np.add((1-alpha)*H_x, alpha*nu_hat_x)
    
    WQ_x = oneConsensus(W,nodes,Q_x)
   
    nuW_hat_x = np.add(Hw_x,WQ_x)
  
    new_Hw_x = np.add((1-alpha)*Hw_x,alpha*nuW_hat_x)
    
    return nu_hat_x, nuW_hat_x, new_H_x, new_Hw_x


# In[ ]:


def compressed_SGD(A,b,x0,y0,D_x,D_y,H_x,H_y,Hw_x,Hw_y,sgd_stepsize,Kmax): ## Kmax = maximumm number of iterations in IPDHG with sgd oracle
    
    "Initializations"
    x_t = np.copy(x0) ## current iterate x_t
    y_t = np.copy(y0) ## current iterate y_t
    xt_hat = np.copy(x0)
    yt_hat = np.copy(y0)
    nu_xt = np.copy(x0)
    nu_yt = np.copy(y0)
    Dx_t = np.copy(D_x)
    Dy_t = np.copy(D_y)
    Hx_t = np.copy(H_x)
    Hy_t = np.copy(H_y)
    Hw_xt = np.copy(Hw_x)
    Hw_yt = np.copy(Hw_y)
    
    dist_from_saddle = np.zeros(Kmax+1)
    func_avg = np.zeros(Kmax+1)
    consensus_error_x = np.zeros(Kmax)
    consensus_error_y = np.zeros(Kmax)
    compression_error_x = np.zeros(Kmax)
    compression_error_y = np.zeros(Kmax)
    
    "step size s and other constants alpha, gamma setup"
    
    
    bx, by = sgd_param_bx_by(mux,muy,Lxx,Lyy,Lxy,Lyx,sgd_stepsize)

    alpha_x, alpha_y, gamma_x, gamma_y, Mx,My = sgd_parameters_alpha_gammaMxMy(bx,by,mux,muy,Lxx,Lyy,
                                                            Lxy,Lyx,delta,nodes,lambda_max)
    
    for t in range(Kmax):
        for i in range(nodes):
            scaled_stoch_grad_x, scaled_stoch_grad_y = SGD_oracle(mini_batch_features[i],mini_batch_values[i],
                                                        sample_prob,nodes,num_batches,x_t[i],y_t[i],
                                                            regcoef_x,regcoef_y, scaling_factor)

            direction_x = np.add(scaled_stoch_grad_x, Dx_t[i])
            nu_xt[i] = np.subtract(x_t[i],sgd_stepsize*direction_x)

            ## finding nu_yt 
            direction_y = np.subtract(scaled_stoch_grad_y, Dy_t[i])
            nu_yt[i] = np.add(y_t[i],sgd_stepsize*direction_y)

        "compressing nu_xt and nu_yt and communicate the resulting compressed vectors"   
        nu_hat_xt, nuW_hat_xt, Hx_t,Hw_xt = COMM(num_bits,nu_xt,Hx_t,Hw_xt,alpha_x,W,nodes)
        nu_hat_yt, nuW_hat_yt, Hy_t,Hw_yt = COMM(num_bits,nu_yt,Hy_t,Hw_yt,alpha_y,W,nodes)


        diff_hat_nux_nuxW = np.subtract(nu_hat_xt,nuW_hat_xt)
        diff_hat_nuy_nuyW = np.subtract(nu_hat_yt,nuW_hat_yt)

        ## updating Dxt and Dyt

        scale_diff_nux_nuxW = (gamma_x/(2*sgd_stepsize))*diff_hat_nux_nuxW
        scale_diff_nuy_nuyW = (gamma_y/(2*sgd_stepsize))*diff_hat_nuy_nuyW

        Dx_t = np.add(Dx_t,scale_diff_nux_nuxW)
        Dy_t = np.add(Dy_t,scale_diff_nuy_nuyW)

        ## upating xt_hat and yt_hat

        gamma_diff_nux_nuxW = (gamma_x/2)*diff_hat_nux_nuxW
        gamma_diff_nuy_nuyW = (gamma_y/2)*diff_hat_nuy_nuyW

        xt_hat = np.subtract(nu_xt,gamma_diff_nux_nuxW)
        yt_hat = np.subtract(nu_yt,gamma_diff_nuy_nuyW)
        for i in range(nodes):
            x_t[i] = projection_L2ball(xt_hat[i],radius_x)
            y_t[i] = projection_L2ball(yt_hat[i],radius_y)

        "To compute zT0 - zT0-1, copy xT0-1, yT0-1"
        
        if (t == Kmax-2):
            prev_xt = np.copy(x_t)
            prev_yt = np.copy(y_t)
            
        "Saving important quantities"

        x_ti_avg = x_t.mean(axis = 0) ## cumputes mean of local iterates x_t[i] over i

        y_ti_avg = y_t.mean(axis = 0) ## cumputes mean of local iterates y_t[i] over i

        func_avg[t+1] = logistic_func(A,b,x_ti_avg,y_ti_avg,nodes,num_batches,regcoef_x,regcoef_y
                                     , scaling_factor)

        function_values.write(str(func_avg[t+1])+'\n')
        function_values.flush()  

        ## saving distance from saddle point solution
        dist_from_saddle[t+1] = distance_from_saddle(x_t,y_t,xstar,ystar,nodes)
        total_distance_from_saddle.write(str(dist_from_saddle[t+1])+'\n')
        total_distance_from_saddle.flush()

        ## saving consensus error
        cons_error_x = 0
        cons_error_y = 0
        for i in range(nodes):
            consensus_x = np.subtract(x_t[i],x_ti_avg)
            cons_error_x += (LA.norm(consensus_x))**2
            consensus_y = np.subtract(y_t[i],y_ti_avg)
            cons_error_y += (LA.norm(consensus_y))**2

        consensus_error_x[t] = cons_error_x
        consensus_error_y[t] = cons_error_y
        sum_consensus_error_x.write(str(consensus_error_x[t])+'\n')
        sum_consensus_error_x.flush()
        sum_consensus_error_y.write(str(consensus_error_y[t])+'\n')
        sum_consensus_error_y.flush()

         ## saving compression error nuxhat - nux  and nuyhat - nuy

        comp_error_x = np.subtract(nu_hat_xt,nu_xt)
        compression_error_x[t] = sum([(LA.norm(comp_error_x[i]))**2 for i in range(nodes)])
        compression_error_nux.write(str(compression_error_x[t])+'\n')
        compression_error_nux.flush()

        comp_error_y = np.subtract(nu_hat_yt,nu_yt)
        compression_error_y[t] = sum([(LA.norm(comp_error_y[i]))**2 for i in range(nodes)])
        compression_error_nuy.write(str(compression_error_y[t])+'\n')
        compression_error_nuy.flush()

        "Saving completed!"   
        
    dist_local_zt_z0 = local_distances(x_t,y_t,prev_xt,prev_yt)
    avg_sgd_dist = acce_consensus(W,nodes,eta,tau,dist_local_zt_z0)
    full_sgd_distances = {'full_sgd_loc_dist':dist_local_zt_z0,'full_sgd_avg_dist':avg_sgd_dist}
        
    return x_t, y_t, Dx_t, Dy_t, Hx_t,Hy_t,Hw_xt,Hw_yt,full_sgd_distances    
        


# In[ ]:


def compressed_SVRG(A,b,x0,y0,Dx,Dy,Hx,Hy,Hwx,Hwy,svrg_stepsize,T):
    "features, values = local samples of all nodes"
    "mini_batch_features, mini_batch_values = mini-batches of all nodes "
    dist_from_saddle = np.zeros(T+1)
    func_avg = np.zeros(T+1)
    omega = np.zeros(T) ## keeps track of full batch gradient computations
    consensus_error_x = np.zeros(T)
    consensus_error_y = np.zeros(T)
    
    compression_error_x = np.zeros(T)
    compression_error_y = np.zeros(T)
    x_t = np.copy(x0) ## current iterate x_t
    y_t = np.copy(y0) ## current iterate y_t
    xt_hat = np.copy(x0)
    yt_hat = np.copy(y0)
    nu_xt = np.copy(x0)
    nu_yt = np.copy(y0)
    Dx_t = np.copy(Dx)
    Dy_t = np.copy(Dy)
    Hx_t = np.copy(Hx)
    Hy_t = np.copy(Hy)
    Hw_xt = np.copy(Hwx)
    Hw_yt = np.copy(Hwy)
    

    "reference points are initialized to the last primal and dual iterates of IPDHG + SGD oracle"
    ref_point_x = np.copy(x_t)
    ref_point_y = np.copy(y_t)
    
    ## Initial full batch gradient
    FB_grad_x = [[] for i in range(nodes)]
    FB_grad_y = [[] for i in range(nodes)]
    for i in range(nodes):
        FB_grad_x[i],FB_grad_y[i] = full_batch_grad(features[i],values[i],regcoef_x,regcoef_y,nodes,num_batches,
                                                     ref_point_x[i],ref_point_y[i],scaling_factor)
    
    "parameters setup for svrg"
    
    tilde_cx, tilde_cy, bx, by = parameters(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize)


    alpha_x, alpha_y, gamma_x, gamma_y,lambda_second_small,lambda_max = parameters_alpha_gamma(
                                                    mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,delta,nodes,W)

    Mx, My = MxMy(alpha_x, alpha_y, gamma_x, gamma_y,lambda_max,delta)
    
    for t in range(T):
        ## generate a Bernoulli rv with prob ref_prob
        omega[t] = bernoulli.rvs(ref_prob)
        ## saving omega values
        full_batch_grad_counts.write(str(omega[t])+'\n')
        full_batch_grad_counts.flush()
        ## computing gradient steps
        for i in range(nodes):
            ## compute stochastic gradient estimator using svrg oracle
            svrg_grad_x,svrg_grad_y,FB_grad_x[i],FB_grad_y[i],ref_point_x[i],ref_point_y[i] = svrg_oracle(
                                                            features[i],values[i],mini_batch_features[i],
                                                          mini_batch_values[i],sample_prob, omega[t],num_batches
                                                                 ,ref_point_x[i], ref_point_y[i],FB_grad_x[i], 
                                                                   FB_grad_y[i],x_t[i],y_t[i],scaling_factor)
            
            
            direction_x = np.add(svrg_grad_x, Dx_t[i])
            nu_xt[i] = np.subtract(x_t[i],svrg_stepsize*direction_x)

            ## finding nu_yt 
            direction_y = np.subtract(svrg_grad_y, Dy_t[i])
            nu_yt[i] = np.add(y_t[i],svrg_stepsize*direction_y)
            
        "compressing nu_xt and nu_yt and communicate the resulting compressed vectors"   
        nu_hat_xt, nuW_hat_xt, Hx_t,Hw_xt = COMM(num_bits,nu_xt,Hx_t,Hw_xt,alpha_x,W,nodes)
        nu_hat_yt, nuW_hat_yt, Hy_t,Hw_yt = COMM(num_bits,nu_yt,Hy_t,Hw_yt,alpha_y,W,nodes)
        
        
        diff_hat_nux_nuxW = np.subtract(nu_hat_xt,nuW_hat_xt)
        diff_hat_nuy_nuyW = np.subtract(nu_hat_yt,nuW_hat_yt)
        
        ## updating Dxt and Dyt
        
        scale_diff_nux_nuxW = (gamma_x/(2*svrg_stepsize))*diff_hat_nux_nuxW
        scale_diff_nuy_nuyW = (gamma_y/(2*svrg_stepsize))*diff_hat_nuy_nuyW
        
        Dx_t = np.add(Dx_t,scale_diff_nux_nuxW)
        Dy_t = np.add(Dy_t,scale_diff_nuy_nuyW)
        
        ## upating xt_hat and yt_hat
        
        gamma_diff_nux_nuxW = (gamma_x/2)*diff_hat_nux_nuxW
        gamma_diff_nuy_nuyW = (gamma_y/2)*diff_hat_nuy_nuyW
        
        xt_hat = np.subtract(nu_xt,gamma_diff_nux_nuxW)
        yt_hat = np.subtract(nu_yt,gamma_diff_nuy_nuyW)
        for i in range(nodes):
            x_t[i] = projection_L2ball(xt_hat[i],radius_x)
            y_t[i] = projection_L2ball(yt_hat[i],radius_y)
            
        "Saving required quantities"
        
        x_ti_avg = x_t.mean(axis = 0) ## cumputes mean of local iterates x_t[i] over i

        y_ti_avg = y_t.mean(axis = 0) ## cumputes mean of local iterates y_t[i] over i

        func_avg[t+1] = logistic_func(A,b,x_ti_avg,y_ti_avg,nodes,num_batches,regcoef_x,regcoef_y
                                     , scaling_factor)
        
        function_values.write(str(func_avg[t+1])+'\n')
        function_values.flush()  
        
        ## saving distance from saddle point solution
        dist_from_saddle[t+1] = distance_from_saddle(x_t,y_t,xstar,ystar,nodes)
        total_distance_from_saddle.write(str(dist_from_saddle[t+1])+'\n')
        total_distance_from_saddle.flush()
        
        ## saving consensus error
        cons_error_x = 0
        cons_error_y = 0
        for i in range(nodes):
            consensus_x = np.subtract(x_t[i],x_ti_avg)
            cons_error_x += (LA.norm(consensus_x))**2
            consensus_y = np.subtract(y_t[i],y_ti_avg)
            cons_error_y += (LA.norm(consensus_y))**2
            
        consensus_error_x[t] = cons_error_x
        consensus_error_y[t] = cons_error_y
        sum_consensus_error_x.write(str(consensus_error_x[t])+'\n')
        sum_consensus_error_x.flush()
        sum_consensus_error_y.write(str(consensus_error_y[t])+'\n')
        sum_consensus_error_y.flush()
        
         ## saving compression error nuxhat - nux  and nuyhat - nuy
        
        comp_error_x = np.subtract(nu_hat_xt,nu_xt)
        compression_error_x[t] = sum([(LA.norm(comp_error_x[i]))**2 for i in range(nodes)])
        compression_error_nux.write(str(compression_error_x[t])+'\n')
        compression_error_nux.flush()
        
        comp_error_y = np.subtract(nu_hat_yt,nu_yt)
        compression_error_y[t] = sum([(LA.norm(comp_error_y[i]))**2 for i in range(nodes)])
        compression_error_nuy.write(str(compression_error_y[t])+'\n')
        compression_error_nuy.flush()
        
        "Saving completed!"
        
    return x_t,y_t
    


# In[ ]:


def heuristic_switch(A,b,x0,y0,D_x,D_y,H_x,H_y,Hw_x,Hw_y,sgd_stepsize,svrg_stepsize,initial_Kmax,
                      sgd_epsilon,Tsvrg,threshold):
    
    "T0 = number of iterations using SG oracle to approximate Phi_0 "
    "We replace xstar, ystar in Phi0 by xT0, yT0"
    
    old_Kmax = initial_Kmax ## setting up T0
    
    
    "calling IPDHG with SG oracle for T0 = initial_Kmax iterations"

    x_t, y_t, Dx_t, Dy_t, Hx_t,Hy_t,Hw_xt,Hw_yt,full_sgd_distances = compressed_SGD(A,b,x0,y0,D_x,D_y,H_x,H_y,
                                                                       Hw_x,Hw_y,sgd_stepsize,
                                                                           initial_Kmax)
    
    "accessing local distances between last two consecutive iterations and their average"
    
    full_sgd_dist_local_zt_z0 = full_sgd_distances['full_sgd_loc_dist']
    avg_full_sgd_dist = full_sgd_distances['full_sgd_avg_dist']
    

    print ('avg_full_sgd_dist',avg_full_sgd_dist)
    
    "Continue with SG oracle if difference between last two iterates xT0, xT0-1 is not very small"
    
    print ('Gap between two last iterates is sufficient?',np.any(avg_full_sgd_dist > threshold))

#     print (np.any(avg_full_sgd_dist > threshold) == True)
    if (np.any(avg_full_sgd_dist > threshold) == True): ## 1 means all entries in avg_sgd_dist are less than equal to atleast one entry in avg_svrg_dist

        new_epsilon = best_epsilon0(target_acc,delta,initial_x0,initial_y0,x_t,y_t,sgd_stepsize,
                                        features,values,W)
        
        print ('new epsilon0',new_epsilon)

        "Computing T_{j+1}"

        new_Kmax = sgd_iteration(mux,muy,Lxx,Lyy,Lxy,Lyx,delta,nodes,lambda_max,lambda_second_small,sgd_stepsize,new_epsilon)

        print ('new Kmax',new_Kmax)
        print ('old Kmax',old_Kmax)

        "Computing T{j+1}- T_{j} "
        Kmax = new_Kmax - old_Kmax ## T{j+1}- T_{j}
        print ('Iterations Tj+1,Tj,Tj+1 - Tj', [new_Kmax,old_Kmax,Kmax])

        old_Kmax = new_Kmax
#         Kmax = 100 ## to check subsequent steps
        
        "Perform Kmax = T{j+1}- T_{j} iterations using SG oracle"
        x_t, y_t, Dx_t, Dy_t, Hx_t,Hy_t, Hw_xt, Hw_yt,sgd_distances = compressed_SGD(A,b,
                                                                                x_t,y_t,Dx_t,Dy_t,
                                                                                Hx_t,Hy_t,Hw_xt,
                                                                                Hw_yt,sgd_stepsize,Kmax)


        
            
    "switch to IPDHG with svrg oracle"
    
    print ('Switch to SVRG oracle')
    x_t, y_t = compressed_SVRG(A,b,x_t, y_t, Dx_t, Dy_t, Hx_t,Hy_t,Hw_xt, Hw_yt,svrg_stepsize,Tsvrg)
    
    
    return x_t,y_t


# In[ ]:


def computing_phi0(Mx,My,gamma_x,gamma_y,delta,x0,y0,xT0,yT0,sgd_stepsize,features,values,W):
    
    "Mx, My, gamma_x, gamma_y, alpha_x, alpha_y are the parameters associated with sgd oracle"
    "zT0 = (xT0,yT0) are the last iterates of IPDHG with SG oracle"
    
    I = np.identity(nodes)
    I_W = np.subtract(I,W)

#     local_phi0 = np.zeros(nodes)

    "store full batch gradients at f_i(zT0)"
    
    FB_grad_x = np.zeros((nodes,dimension_x))
    FB_grad_y = np.zeros((nodes,dimension_y))
    
    "stores difference of grad f_i and average (over nodes) gradients f_i to obtain (i-J)grad F "
    
    grad_diff_x = np.zeros((nodes,dimension_x))
    grad_diff_y = np.zeros((nodes,dimension_y))
    
    
    "computing ||x0-xT0||^2"
    
    diff_x0_xT0 = np.subtract(x0,xT0)
    norm_x0_xT0 = (LA.norm(diff_x0_xT0))**2
#     Mx_xstar = Mx*nodes*norm_x0_xstar
    
    "computing ||y0-yT0||^2"
    
    diff_y0_yT0 = np.subtract(y0,yT0)
    norm_y0_yT0 = (LA.norm(diff_y0_yT0))**2
#     My_ystar = My*nodes*norm_y0_ystar
    
    for i in range(nodes):
        "computing local full batch gradients"
        FB_grad_x[i], FB_grad_y[i] = full_batch_grad(features[i],values[i],regcoef_x,regcoef_y,nodes,num_batches,
                                                     xT0[i],yT0[i],scaling_factor) 
       
    avg_grad_x = FB_grad_x.mean(axis = 0) ## average of grad_x f_i's
    avg_grad_y = FB_grad_y.mean(axis = 0) ## average of grad_y f_i 's'
    
    for i in range(nodes):
        grad_diff_x[i] = np.subtract(FB_grad_x[i],avg_grad_x) 
        grad_diff_y[i] = np.subtract(FB_grad_y[i],avg_grad_y)
        
    pseudo_inv_I_W = np.linalg.pinv(I_W) ## pseudoinverse of I-W
    sqrt_pinv_I_W = scipy.linalg.sqrtm(pseudo_inv_I_W, disp=True, blocksize=64) ## square root of I-W pseudoinverse
    
    pinv_grad_diff_x = np.matmul(sqrt_pinv_I_W,grad_diff_x) 
    pinv_grad_diff_y = np.matmul(sqrt_pinv_I_W,grad_diff_y) 
        
    norm_pinv_grad_diff_x = (LA.norm(pinv_grad_diff_x))**2 ## ||(I-J)grad_x F(1zstar)||^2_{pinv_I-W}
    norm_pinv_grad_diff_y = (LA.norm(pinv_grad_diff_y))**2 ## ||(I-J)grad_y F(1zstar)||^2_{pinv_I-W}
    
    "computing square norm of H0x - Hstar_x and H0y - Hstar_y"
    
    tile_avg_grad_x = np.tile(avg_grad_x,(nodes,1))
    tile_avg_grad_y = np.tile(avg_grad_y,(nodes,1))

    diff_H0x_Hstar_x =  np.add(diff_x0_xT0, sgd_stepsize*tile_avg_grad_x) 
    diff_H0y_Hstar_y =  np.subtract(diff_y0_yT0, sgd_stepsize*tile_avg_grad_y)
    
    
    norm_diff_H0x_Hstar_x = (LA.norm(diff_H0x_Hstar_x))**2
    norm_diff_H0y_Hstar_y = (LA.norm(diff_H0y_Hstar_y))**2
    
    phi_0 = Mx*norm_x0_xT0 + My*norm_y0_yT0 + (2*sgd_stepsize**2)/gamma_x*norm_pinv_grad_diff_x + (2*sgd_stepsize**2)/gamma_y*norm_pinv_grad_diff_y + math.sqrt(delta)*(norm_diff_H0x_Hstar_x + norm_diff_H0y_Hstar_y)
    
    
    return phi_0,(LA.norm(avg_grad_x))**2, (LA.norm(avg_grad_y))**2
    


# In[ ]:


"best epsilon0 using approximation of phi0"

def best_epsilon0(target_acc,delta,x0,y0,xT0,yT0,sgd_stepsize,features,values,W):
    
    "parameters for svrg"
    
    tilde_cx, tilde_cy, svrg_bx, svrg_by = parameters(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize)


    svrg_alpha_x, svrg_alpha_y, svrg_gamma_x, svrg_gamma_y,lambda_second_small,lambda_max = parameters_alpha_gamma(
                                                            mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,delta,nodes,W)

    svrg_Mx, svrg_My = MxMy(svrg_alpha_x, svrg_alpha_y, svrg_gamma_x, svrg_gamma_y,lambda_max,delta)
    
    "Computes Phi0"
    
    
    sgd_bx, sgd_by = sgd_param_bx_by(mux,muy,Lxx,Lyy,Lxy,Lyx,sgd_stepsize) ## sgd param
    sgd_alpha_x, sgd_alpha_y, sgd_gamma_x, sgd_gamma_y, sgd_Mx, sgd_My = sgd_parameters_alpha_gammaMxMy(sgd_bx,sgd_by,mux,muy,Lxx,Lyy,Lxy,Lyx,delta,nodes,lambda_max)

#     computing_phi0(Mx,My,gamma_x,gamma_y,delta,x0,y0,xT0,yT0,sgd_stepsize,features,values,W)
    phi_0,norm_avg_grad_x,norm_avg_grad_y = computing_phi0(sgd_Mx,sgd_My,sgd_gamma_x,sgd_gamma_y,delta,x0,y0,
                                                 xT0,yT0,sgd_stepsize,features,values,W)
    
    print ('phi0',phi_0)
    svrg_rho = rho_svrg(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize,W)

    "Computes Cmax"
    
    first_term  = (svrg_Mx+tilde_cx)/sgd_Mx
    sec_term = (svrg_My+tilde_cy)/sgd_My
    third_term = (sgd_gamma_x*svrg_stepsize**2)/(svrg_gamma_x*sgd_stepsize**2)
    fourth_term = (sgd_gamma_y*svrg_stepsize**2)/(svrg_gamma_y*sgd_stepsize**2)

    
    Cmax = max( first_term,sec_term,third_term,fourth_term, 2 )
    
    epsilon0 = target_acc/(2*Cmax*phi_0)

    
    print ('best epsilon0',epsilon0)
    
    return epsilon0
    
    


# In[ ]:


def empty_graph(n=0,create_using=None):
    """Return the empty graph with n nodes and zero edges.

    Node labels are the integers 0 to n-1
    
    """
    
    if create_using is None:
        # default empty graph is a simple graph
        G=nx.Graph()
    else:
        G=create_using
        G.clear()

    G.add_nodes_from(range(n))
    G.name="empty_graph(%d)"%n
    return G


def grid_2d(m,n,periodic=False,create_using=None): ## m,n be the number of rows and number of 
    # columns in torus topolgy
    
    """ Return the 2d grid graph of mxn nodes,
        each connected to its nearest neighbors.
        Optional argument periodic=True will connect
        boundary nodes via periodic boundary conditions.
    """
    
    G=empty_graph(0,create_using)
    G.name="grid_2d_graph"
    rows=range(m)
    columns=range(n)
    G.add_nodes_from( (i,j) for i in rows for j in columns )
    G.add_edges_from( ((i,j),(i-1,j)) for i in rows for j in columns if i>0 )
    G.add_edges_from( ((i,j),(i,j-1)) for i in rows for j in columns if j>0 )
    if G.is_directed():
        G.add_edges_from( ((i,j),(i+1,j)) for i in rows for j in columns if i<m-1 )
        G.add_edges_from( ((i,j),(i,j+1)) for i in rows for j in columns if j<n-1 )
    if periodic:
        if n>2:
            G.add_edges_from( ((i,0),(i,n-1)) for i in rows )
            if G.is_directed():
                G.add_edges_from( ((i,n-1),(i,0)) for i in rows )
        if m>2:
            G.add_edges_from( ((0,j),(m-1,j)) for j in columns )
            if G.is_directed():
                G.add_edges_from( ((m-1,j),(0,j)) for j in columns )
        G.name="periodic_grid_2d_graph(%d,%d)"%(m,n)
    return G



# In[ ]:


def gen_graph(row,column,m):
    W = [[0 for j in range(m)] for i in range(m)] # weight matrix
    
    "Generating 2D Grid"
    G = grid_2d(row,column,periodic=False,create_using=None)
    
#     "Adding extra edges to get 2D Torus"
#     edges_rows = [((0,0),(0,4)),((1,0),(1,4)),((2,0),(2,4)),((3,0),(3,4))] 
#     column_rows = [((0,0),(3,0)),((0,1),(3,1)),((0,2),(3,2)),((0,3),(3,3)),((0,4),(3,4))] 
#     G.add_edges_from(edges_rows)
#     G.add_edges_from(column_rows)
    nx.draw_networkx(G)
    plt.savefig('2D-Grid')
    plt.axis('off')
    plt.show()
    
    "Changing edges format to 1D so that it becomes easy to access the indices"
    
    edges = []
    for i in range(row):
        for j in range(column-1):
            edges.append([j+i*column,j+1+i*column])

    for i in range(row-1):
        for j in range(column):
            edges.append([j+i*column,j+(i+1)*column])      

    "Adding extra edges to get 2D Torus from 2D grid"

    for i in range(column):
        edges.append([i , i + (row-1)*column])

    for i in range(row):
        edges.append([i*column , row + i*column]) 

    print ('edges:',edges)   
    print ('total edges in 2D Torus:',len(edges))
    
    for (u, v) in edges:
        W[u][v] = 1/5
        W[v][u] = 1/5

    
    for i in range(m):
        W[i][i] = 1/5

    
    B = np.matrix(W)
    print ((B.transpose() == W).all()) ## check symmetric property of B. It will print True
    print ([sum(W[i]) for i in range(m)])
    return W


# In[ ]:


# Computing parameters like step size, tilde_cx, tilde_cy, bx and by for svrg oracle
def parameters(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize):
    mu = min(mux,muy)
    L = max(Lxx,Lyy,Lxy,Lyx)

    num_tilde_cx = 8*svrg_stepsize**2*(Lxx**2 + Lyx**2)
    tilde_cx = num_tilde_cx/ref_prob ## Taking uniform sample distribution
    num_tilde_cy = 8*svrg_stepsize**2*(Lyy**2 + Lxy**2)
    tilde_cy = num_tilde_cy/ref_prob 
    bx = svrg_stepsize*mux - 4*svrg_stepsize**2*Lyx**2 - tilde_cx*ref_prob
    by = svrg_stepsize*muy - 4*svrg_stepsize**2*Lxy**2 - tilde_cy*ref_prob
    
    return tilde_cx, tilde_cy, bx, by


# In[ ]:


# Computing parameters for svrg oracle

def parameters_alpha_gamma(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,delta,nodes,W):
    tilde_cx, tilde_cy, bx, by = parameters(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize)
    alpha_x = bx/(1+delta)
    alpha_y = by/(1+delta)
    I = np.identity(nodes)
    I_W = np.subtract(I,W)
    eigvalues , eigvectors = eig(I_W)
    eigvalues = np.sort(eigvalues) ## sort eigenvalues in increasing order
    lambda_max = eigvalues[-1]
    lambda_second_small = eigvalues[1]
    "gamma_x"
    gamma_x_second = 1/(4*(1+delta)*lambda_max)
    gamma_x_first = gamma_x_second*(bx/math.sqrt(delta))
    gamma_x = min(gamma_x_first, gamma_x_second)
    
    "gamma_y"
    gamma_y_first = gamma_x_second*(by/math.sqrt(delta))
    gamma_y = min(gamma_y_first,gamma_x_second)
    
    return alpha_x, alpha_y, gamma_x, gamma_y,lambda_second_small,lambda_max
 
    


# In[ ]:


# Computing parameters for svrg oracle

def MxMy(alpha_x, alpha_y, gamma_x, gamma_y,lambda_max,delta):
    "Mx and My"
    Mx = 1 - (math.sqrt(delta)*alpha_x)/(1-0.5*gamma_x*lambda_max)
    My = 1 - (math.sqrt(delta)*alpha_y)/(1-0.5*gamma_y*lambda_max)
    return Mx,My
    
    

def rho_svrg(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize,W):
    
    tilde_cx, tilde_cy, bx, by = parameters(mux,muy,Lxx,Lyy,Lxy,Lyx,ref_prob,svrg_stepsize)
    
    alpha_x, alpha_y, gamma_x, gamma_y,lambda_second_small,lambda_max = parameters_alpha_gamma(mux,muy,
                                                                        Lxx,Lyy,Lxy,Lyx,ref_prob,
                                                                            delta,nodes,W)
    Mx,My = MxMy(alpha_x, alpha_y, gamma_x, gamma_y,lambda_max,delta)
    
    T1 = (1-bx)/Mx
    T2 = (1-by)/My
    T3 = 1 - 0.5*gamma_x*lambda_second_small
    T4 = 1 - 0.5*gamma_y*lambda_second_small
    T5 = 1-alpha_x
    T6 = 1-alpha_y
    rho = max(T1,T2,T3,T4,T5,T6)
    
    return rho        


# In[ ]:


# Computing parameters for sgd oracle


def sgd_param_bx_by(mux,muy,Lxx,Lyy,Lxy,Lyx,sgd_stepsize):

    bx = sgd_stepsize*mux - 4*sgd_stepsize**2*Lyx**2 
    by = sgd_stepsize*muy - 4*sgd_stepsize**2*Lxy**2 
 
    return bx, by

def sgd_parameters_alpha_gammaMxMy(bx,by,mux,muy,Lxx,Lyy,Lxy,Lyx,delta,nodes,lambda_max):
    
    alpha_x = bx/(1+delta)
    alpha_y = by/(1+delta)
    
    "gamma_x"
    gamma_x_second = 1/(2*((1+delta)**2)*lambda_max)
    gamma_x = bx*gamma_x_second
   
    "gamma_y"
    gamma_y = by*gamma_x_second
    
    "Mx,My"
    
    Mx = 1 - (math.sqrt(delta)*alpha_x)/(1-0.5*gamma_x*lambda_max)
    My = 1 - (math.sqrt(delta)*alpha_y)/(1-0.5*gamma_y*lambda_max)

    return alpha_x, alpha_y, gamma_x, gamma_y, Mx, My
    


# In[ ]:


def sgd_iteration(mux,muy,Lxx,Lyy,Lxy,Lyx,delta,nodes,lambda_max,lambda_second_small,sgd_stepsize,sgd_epsilon):

    bx,by = sgd_param_bx_by(mux,muy,Lxx,Lyy,Lxy,Lyx,sgd_stepsize)
    alpha_x, alpha_y, gamma_x, gamma_y, Mx, My = sgd_parameters_alpha_gammaMxMy(bx,by,mux,muy,
                                                            Lxx,Lyy,Lxy,Lyx,delta,nodes,lambda_max)
    
    T1 = (1-bx)/Mx
    T2 = (1-by)/My
    T3 = 1 - 0.5*gamma_x*lambda_second_small
    T4 = 1 - 0.5*gamma_y*lambda_second_small
    T5 = 1-alpha_x
    T6 = 1-alpha_y
    rho = max(T1,T2,T3,T4,T5,T6)
    denominator = (-1)*math.log(rho)
    T = -1*math.log(sgd_epsilon)/denominator

    return int(T)


# In[ ]:


## Initialization
sgd_epsilon = 0.5
row = 4 ### total number of rows in 2D Torus
column = 5 ### total number of columns in 2D Torus. Note that column = row + 1
nodes = row*column ### total number of nodes

num_batches = 20 ## This must be fixed for comparison
ref_prob = 1/num_batches

print('ref prob',ref_prob)

regcoef_x = 10
regcoef_y = 10

mux = regcoef_x
muy = regcoef_y

radius_x = 100
radius_y = 1


scaling_factor = A.shape[0]

dimension_x = A.shape[1]
dimension_y = dimension_x


## finding delta

num_bits = 4
delta = delta_qsgd(num_bits,dimension_x)


sample_prob = (1/num_batches)*np.ones(num_batches)
print ('sample prob',sample_prob)


# In[ ]:


## Generating weight matrix W

W = gen_graph(row,column,nodes)

eigenvalues_W, eigenvectors_W = eig(W)

sorted_eigenvalues_W = np.sort(eigenvalues_W)
print ('eigenvalues of W', sorted_eigenvalues_W)

print ('delta',delta)


I = np.identity(nodes)
I_W = np.subtract(I,W)
eigvalues , eigvectors = eig(I_W)
eigvalues = np.sort(eigvalues) ## sort eigenvalues in increasing order
lambda_max = eigvalues[-1]
lambda_second_small = eigvalues[1]


"constants eta and tau used in accelerated consensus method "

muW = sorted_eigenvalues_W[nodes-2] ## second largest eigen value of W
print ('muW',muW)
eta = (1 - math.sqrt(1 - muW**2))/(1 + math.sqrt(1 - muW**2))
tau = 20 ## iterations in accelerated consensus


# In[ ]:


## distribute data point among nodes
features, values = data_blocks(A,b,nodes)

## creating mnibatches for all nodes

mini_batch_features,mini_batch_values = nodes_mini_batches(features,values,nodes,num_batches)


# In[ ]:


Lxx,Lyy,Lxy,Lyx = global_lipschitz(mini_batch_features,mini_batch_values,radius_x,radius_y,nodes
                    ,num_batches,regcoef_x,regcoef_y,scaling_factor)

print ('global Lipschitz parameters',(Lxx,Lyy,Lxy,Lyx))


# In[ ]:


"Initialization"

L = max(Lxx,Lyy,Lxy,Lyx)
mu = min(mux,muy)

"stepsize used in SG oracle"

sgd_stepsize = mu/(4*L**2) ## step size for sgd
print ('sgd_stepsize',sgd_stepsize)

"step size used in svrg oracle"

svrg_stepsize = mu/(21*L**2)
print ('svrg step size',svrg_stepsize)

"Initial Kmax"

# sgd_epsilon = 0.5

initial_Kmax = sgd_iteration(mux,muy,Lxx,Lyy,Lxy,Lyx,delta,nodes,lambda_max,lambda_second_small,sgd_stepsize,sgd_epsilon)

# initial_Kmax = 10
print ('sgd_iterations =',initial_Kmax)


Tsvrg = 2000 ## number of iterations of IPDHG + SVRG

threshold = 1e-08
target_acc = 1e-08


# In[ ]:


## Creating files to save soutput

sum_consensus_error_x = open(r"sgd_switch_tosvrg_consensus_error_x_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")
sum_consensus_error_y = open(r"sgd_switch_tosvrg_consensus_error_y_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")
total_distance_from_saddle = open(r"sgd_switch_tosvrg_total_distance_from_saddle_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")
function_values = open(r"sgd_switch_tosvrg_function_values_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")
full_batch_grad_counts = open(r"sgd_switch_tosvrg_full_batch_grad_counts_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")
compression_error_nux = open(r"sgd_switch_tosvrg_compression_error_nux_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")
compression_error_nuy = open(r"sgd_switch_tosvrg_compression_error_nuy_num_bits_4_eps_"+str(sgd_epsilon)+".txt","w")


## generate initial points randomly

np.random.seed(1234)
x0_i = np.random.random(dimension_x)
x0 = np.tile(x0_i,(nodes,1))

np.random.seed(100)
y0_i = np.random.random(dimension_y)
y0 = np.tile(y0_i,(nodes,1))

print ('shape of x0',x0.shape)
print ('shape of y0',y0.shape)

H_x = np.copy(x0)
H_y = np.copy(y0)

D_x = np.zeros((nodes,dimension_x))
D_y = np.zeros((nodes,dimension_y))

Hw_x = oneConsensus(W,nodes,H_x)
Hw_y = oneConsensus(W,nodes,H_y)
    
initial_x0 = np.copy(x0)
initial_y0 = np.copy(y0)

## Loading saddle point solution zstar = (xstar,ystar) to plot the ||zt - zstar||
## User has to change file name accordingly

xstar = np.loadtxt('xstar_radiusx_100.txt')
ystar = np.loadtxt('ystar_radiusx_100.txt')


# In[ ]:


xT, yT = heuristic_switch(A,b,x0,y0,D_x,D_y,H_x,H_y,Hw_x,Hw_y,sgd_stepsize,svrg_stepsize,initial_Kmax,
                      sgd_epsilon,Tsvrg,threshold)


# In[ ]:





# In[ ]:




