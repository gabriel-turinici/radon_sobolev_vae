# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:11:34 2020

@author: Gabriel Turinici

Goal: utility files for the computation of the Radon-Sobolev distance

    Reference: Radon-Sobolev distance is that in paper https://arxiv.org/abs/1911.13135

"""

import numpy as np
#from scipy.special import gamma
from scipy.special import loggamma
from scipy.stats import poisson
from scipy.optimize import check_grad


N=8#dimension of the Gaussian

K=3#for some routines, the number of Dirac masses in the measure

##############Radon-Sobolev routines#########################


exact_xi=True # if set to false use an approximation as in the paper: sqrt(x**2+c1)+c0
#we obtain qualitatively same solutions, although the distance is not said to be the same
#exact_xi=False

def mean_distance_to_normal_poisson(x,D=8):
    ''' 
    compute mean distance to a normal variable using energy metric
    and the poisson distribution using formula from https://arxiv.org/abs/1911.13135
    '''
#    if(np.abs(x) <=1.e-15):
#        nr_terms=1#for x=0 one term is enough

    #only take relevant indices i.e. those that are ok for the poisson distrib
    i1,i2=poisson.interval(0.999999999999999,mu=x*x/2)
    indices=np.array(range(np.int(i1),np.int(i2)+1))
    proba_poisson=poisson.pmf(k=indices,mu=x*x/2)

    values = np.exp(loggamma(D/2+0.5+indices)- loggamma(D/2+indices))
    
    return np.sqrt(2.)*np.sum(values*proba_poisson)

#testing
print('mean distance squared among K-samples',
      mean_distance_to_normal_poisson(0,N)/K/np.sqrt(2))

def radon_sobolev_distance_to_normal_sq_poisson(x,D=8):
    if(exact_xi):
        return mean_distance_to_normal_poisson(x,D)-\
               mean_distance_to_normal_poisson(0,D)/np.sqrt(2)
    else:
        xi0pp=np.exp(loggamma(D/2+0.5)- loggamma(1.+D/2.))/np.sqrt(2.)
        xi0=mean_distance_to_normal_poisson(0,D)    
        c0,c1=xi0-1/xi0pp,1/(xi0pp**2)
        return np.sqrt(x*x+c1)+c0

#Next function is ony used to check the gradient
def grad_mean_distance_to_normal_poisson(x,D=8):
    ''' 
    gradient of the distance to a normal variable using energy metric
    '''
    return x*(mean_distance_to_normal_poisson(x,D+2)
              -mean_distance_to_normal_poisson(x,D)
              )



#performs a check of the gradient
check_gradient_xiN=False
#check_gradient_xiN=True
if(check_gradient_xiN):
    grad_err=[]
    for ii in range(1000):
        x0=np.random.uniform(0,20.)
        check_grad_result=check_grad(
            lambda x: radon_sobolev_distance_to_normal_sq_poisson(x,N),
            lambda x: grad_mean_distance_to_normal_poisson(x,N),x0
            )
#        print('check grad result: x0=',x0,' error=',check_grad_result)    
        grad_err.append(check_grad_result)
    print('max err=',np.max(grad_err))

def grad_rs_dist_to_normal_over_x(x,D=8):
    ''' 
    with notations from paper, this function computes : xi'(x)/x
    If approximation is to be used: use as xi = sqrt(x**2+c1)+c0 and return
    1/sqrt(x**2+c1)
    '''
    if(exact_xi):
        return (mean_distance_to_normal_poisson(x,D+2)
                -mean_distance_to_normal_poisson(x,D))
    else:
        xi0pp=np.exp(loggamma(D/2+0.5)- loggamma(1.+D/2.))/np.sqrt(2.)
        c1=1/(xi0pp**2)
        return 1./np.sqrt(x*x+c1)

#testing: uncomment the next lines
#for testval in [0,10,38,50,150,250.]:
#   print(testval,mean_distance_to_normal_poisson(testval),
#         grad_mean_distance_to_normal_poisson(testval))


def distance_sq_to_normal_from_discrete_distrib(points,alphas=None):
    '''
    Parameters
    ----------
    points : N x K matrix
        each of the K columns is a point X_k in R^N
    alphas : 1D array of weights of shape [K,1]
        weights of the distribution; default value = uniform = 1/K

    Returns
    -------
    the distance from the distribution sum_k alpha_k X_k to a N-dim normal
    
    The distance squared is computed as 
    \sum_k alpha_k xi(|X_k|) - (1/2)\sum_{k,l} |Xk-Xl|

    '''
    dim,cK=points.shape
    if alphas is None:
        alphas = np.ones((cK))/cK
    nK,=alphas.shape
    assert(nK==cK)
    
    norms= np.linalg.norm(points,ord=2,axis=0)#norm of each X_k
    xi_of_Xk = np.array([radon_sobolev_distance_to_normal_sq_poisson(xkn,dim) 
                         for xkn in norms ])

    #matrix of distances
#    distXX = np.linalg.norm(np.expand_dims(points, 2)
#                            -np.expand_dims(points, 1),ord=2,axis=0)    
    distXX = np.linalg.norm(points[:,:,None]-points[:,None,:],ord=2,axis=0)
    #implementation note: with notation deltaxkl=points[:,:,None]-points[:,None,:]
    #then deltaxkl[:,k,l] = points[:,k]-points[:,l]
    
    return alphas@xi_of_Xk - 0.5*alphas@distXX@alphas

#test  distance_sq_to_normal_from_discrete_distrib(np.random.randn(N,K))


def grad_dist_sq_to_normal_from_discrete_distrib(points,alphas=None,output_parts=False):
    '''
    Parameters
    ----------
    points, alphas :same as in distance_sq_to_normal_from_discrete_distrib
    output_parts : if set to True then return a triplet of gradient, term1 and term2
    Returns
    -------
    the matrix of gradients
    
    Compuation details: 
        gradient with respect to X_k is alpha_k X_k \cdot xi'(|X_k|)'/|X_k|  - 
       \frac{X_k-X_l}{|X_k-X_l|} sum_{l \neq k} alpha_k alpha_l

    '''
    dim,cK=points.shape
    if alphas is None:
        alphas = np.ones((cK))/cK
    nK,=alphas.shape
    assert(nK==cK)


    norms= np.linalg.norm(points,ord=2,axis=0)#norm of each X_k
    #matrix of distances
    deltaxkl=points[:,:,None]-points[:,None,:]#dimension N,K,K
    distXX = np.linalg.norm(deltaxkl,ord=2,axis=0)

    #smoothing part : put small term everywhere to be able to divide by it    
    # small_eps=1.e-16
    #smooth_distXX = np.maximum(distXX,small_eps)
    #put 1.0 on the diagonal
    smooth_distXX = distXX
    np.fill_diagonal(smooth_distXX, 1.0)
    #compute Xk_Xl/smoothed_norm times alpha_k alpha_l
    Xk_minus_Xl_over_sm_norm = deltaxkl/smooth_distXX[None,:,:]
    Xk_minus_Xl_over_sm_norm *= (alphas[None,:,None]*alphas[None,None,:])
    # note: the 0.5 term dissapears in the gradient because there are 
    # two terms dist(Xk,Xl) and dist(Xl,Xk)

    alphaxikoverx = np.array([alphak*grad_rs_dist_to_normal_over_x(xkn,dim) 
                     for alphak,xkn in zip(alphas,norms)])

    grad_term_xi=alphaxikoverx[None,:]*points#part concerning xi(X_k)
    grad_term_dist=np.sum(Xk_minus_Xl_over_sm_norm,axis=2)
    if(output_parts):
        return grad_term_xi- grad_term_dist,grad_term_xi,grad_term_dist
    else:
        return grad_term_xi- grad_term_dist

grad_code_check=False
#grad_code_check=True
if(grad_code_check): 
    checkdim=217
    Xt=np.random.randn(checkdim,3)#each point of a column of the NxK matrix
#    Xt[:,2]=Xt[:,0]+1.e-18*np.random.randn(217)#each point of a column of the NxK matrix

    full_grad,term_xi_grad,term_dist_grad = \
    grad_dist_sq_to_normal_from_discrete_distrib(Xt,output_parts=True)
    Vx=Xt[:,0]
    Vy=Xt[:,1]
    Vz=Xt[:,2]
    
    print(np.linalg.norm(term_dist_grad[:,0]- (Vx-Vy)/np.sqrt(np.sum((Vx-Vy)**2))/(K*K)
        -(Vx-Vz)/np.sqrt(np.sum((Vx-Vz)**2))/(K*K)))
    print(np.linalg.norm(term_dist_grad[:,1]- (Vy-Vz)/np.sqrt(np.sum((Vz-Vy)**2))/(K*K)
        -(Vy-Vx)/np.sqrt(np.sum((Vx-Vy)**2))/(K*K)))
    print(np.linalg.norm(term_dist_grad[:,2]- (Vz-Vy)/np.sqrt(np.sum((Vz-Vy)**2))/(K*K)
        -(Vz-Vx)/np.sqrt(np.sum((Vx-Vz)**2))/(K*K)))
    normVx = np.sqrt(np.sum(Vx**2))
    grad_dn_Vx  = grad_mean_distance_to_normal_poisson(normVx,checkdim)
    grad_dn_Vx2  = grad_rs_dist_to_normal_over_x(normVx,checkdim)
    
    print(np.linalg.norm(term_xi_grad[:,0]- Vx*grad_dn_Vx/normVx/K))
    print(np.linalg.norm(term_xi_grad[:,0]- Vx*grad_dn_Vx2/K))

    #print(normVx*grad_rs_dist_to_normal_over_x(normVx,checkdim) - grad_mean_distance_to_normal_poisson(normVx,checkdim))

#%%
######################################
#
#         check Radon-Sobolev gradient
#
######################################
check_gradient=False
#check_gradient=True
if(check_gradient):
    X0=np.random.randn(N,K)
    check_grad_result=check_grad(
        lambda x: distance_sq_to_normal_from_discrete_distrib(x.reshape(N,K)),
        lambda x: grad_dist_sq_to_normal_from_discrete_distrib(x.reshape(N,K)).flatten(),
        X0.flatten()
        )
    print('check grad result=',check_grad_result)    
