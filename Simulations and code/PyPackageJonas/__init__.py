
import numpy as np
from matplotlib.colors import ListedColormap, Colormap, LinearSegmentedColormap


import pandas as pd
import scipy
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

import sys
import random

import scipy.integrate
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from matplotlib import cm

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from networkx.drawing.nx_agraph import graphviz_layout


colormap_corr = LinearSegmentedColormap.from_list("", ["#8B0000","white","#00008B"])

def ND(G_obs, alpha = 0, beta = 0.99):
    # alpha : how many (percent) edges should be set to 0. I.e. how many entries should be set to 0 in G_obs

    G = G_obs.copy()   
    n = G.shape[0]
    
    # Filter out small values
    q = np.quantile(G[np.triu_indices(n)], alpha)
    G[G < q] = 0
    
    eg = np.linalg.eig(G)

    D = np.real(eg.eigenvalues)
    V = np.real(eg.eigenvectors)

    lam_p=np.abs( np.max( [D.max(), 0]))
    lam_n=np.abs( np.min( [D.min(), 0]))
    
    m1=lam_p*(1-beta)/beta
    m2=lam_n*(1+beta)/beta

    m = np.max( [m1,m2] )

    D /= (m + D)

    mat_new1 = V @ np.diag(D) @ V.T

    return mat_new1

def GenLabels(variableName, N):
    return [ "$" + f"{variableName}" + "_{" + f"{n+1}" + "}" + "$" for n in range(N)]

def VariableMatrixSetLabels(ax, N, variableName: str):
    ax.set_xticks([x for x in range(N)], GenLabels(variableName, N), rotation='vertical');
    ax.set_yticks([x for x in range(N)], GenLabels(variableName, N));


def ComputeA0(x_i, h):
    """
    compute the Jones (3.4) boundary correted coefficient of order 0 on support [0,1]
    """
    lower_bounds = -x_i/h[:,None]
    upper_bounds = (1-x_i)/h[:,None]
    cdf = scipy.stats.norm().cdf
    m, n = x_i.shape
    # res = np.full(x_i.shape, -np.inf)
    # for i in tqdm(range(m)):
    #     for j in range(n):
    #         res[i,j] = cdf(upper_bounds[i,j]) - cdf(lower_bounds[i,j])
    
    res = cdf(upper_bounds) - cdf(lower_bounds)
    return res

def ComputeA1(x_i, h):
    """
    compute the Jones (3.4) boundary correted coefficient of order 1 on support [0,1]
    """
    lower_bounds = -x_i/h[:,None]
    upper_bounds = (1-x_i)/h[:,None]
    pdf = scipy.stats.norm().pdf
    m, n = x_i.shape
    res = np.full(x_i.shape, -np.inf)
    for i in tqdm(range(m)):
        for j in range(n):
            res[i,j] = scipy.integrate.quad(lambda x: pdf(x) * x, lower_bounds[i,j],upper_bounds[i,j])[0]
    
    return res


def ComputeA1Approx(x_i, h, n_int):
    """
    compute the Jones (3.4) boundary correted coefficient of order 1 on support [0,1]
    """
    lower_bounds = -x_i/h[:,None]
    upper_bounds = (1-x_i)/h[:,None]
    pdf = scipy.stats.norm().pdf

    # x_int = np.linspace(0,1,n_int)[:,None,None]

    x_int = lower_bounds + (upper_bounds - lower_bounds) * np.linspace(0,1,n_int)[..., np.newaxis, np.newaxis]

    return (pdf(x_int) * x_int).sum(axis=0) / (n_int * h[..., np.newaxis])


    # m, n = x_i.shape
    # res = np.full(x_i.shape, -np.inf)
    # for i in tqdm(range(m)):
    #     for j in range(n):
    #         res[i,j] = scipy.integrate.quad(lambda x: pdf(x) * x, lower_bounds[i,j],upper_bounds[i,j])[0]
    
    # return res

def ComputeA2(x_i, h):
    """
    compute the Jones (3.4) boundary correted coefficient of order 2 on support [0,1]
    """
    lower_bounds = -x_i/h[:,None]
    upper_bounds = (1-x_i)/h[:,None]
    pdf = scipy.stats.norm().pdf
    m, n = x_i.shape
    res = np.full(x_i.shape, -np.inf)
    for i in tqdm(range(m)):
        for j in range(n):
            res[i,j] = scipy.integrate.quad(lambda x: pdf(x) * x**2, lower_bounds[i,j],upper_bounds[i,j])[0]
    
    return res


def ComputeA2Approx(x_i, h, n_int):
    """
    compute the Jones (3.4) boundary correted coefficient of order 1 on support [0,1]
    """
    lower_bounds = -x_i/h[:,None]
    upper_bounds = (1-x_i)/h[:,None]
    pdf = scipy.stats.norm().pdf

    # x_int = np.linspace(0,1,n_int)[:,None,None]

    x_int = lower_bounds + (upper_bounds - lower_bounds) * np.linspace(0,1,n_int)[..., np.newaxis, np.newaxis]

    return (pdf(x_int) * x_int**2).sum(axis=0) / (n_int * h[..., np.newaxis])


    # m, n = x_i.shape
    # res = np.full(x_i.shape, -np.inf)
    # for i in tqdm(range(m)):
    #     for j in range(n):
    #         res[i,j] = scipy.integrate.quad(lambda x: pdf(x) * x, lower_bounds[i,j],upper_bounds[i,j])[0]
    
    # return res



def special_func(x_eval, x_i, h, n_int = 3_000):
    a0 = ComputeA0(x_i, h)
    a1 = ComputeA1Approx(x_i, h, n_int)
    a2 = ComputeA2Approx(x_i, h, n_int)


    # using (U1)

    # In the first variable (using jnn and jones 1993)
    # Need normalization (proper)
    return (np.exp(((a2[0,:] - a1[0,:]* (x_eval[None,:] - x_i[0,:])/h[0] ) / (a2[0,:] * a0[0,:] - a1[0,:]**2) ) - 1 - 1/2 * ((x_eval[None,:] - x_i[0,:])/h[0])**2) ).sum(axis=0) / (h[0] * np.sqrt(2*np.pi))




def akde_n(X, grid = None, gam = None):
    # gam: number of (randomly drawn w.o. repetition) samples from X
    (n,d) = X.shape
    if not gam:
        gam = int(np.ceil( n**(1/3) )) + 20

    
    MAX = X.max(axis=0)
    MIN = X.min(axis=0)
    scaling = MAX - MIN
    MAX = MAX + scaling / 10
    MIN = MIN - scaling / 10
    scaling = MAX - MIN

    X = X - MIN
    X = X / scaling

    
    
    # Initialize algorithm
    del_ = 0.9 / n**(d/(d+4))
    del_ = 0.05248
    perm = list(range(n)); random.shuffle(perm)

    mu = X[perm[:gam],:]
    w = np.random.uniform(size = (gam))
    w = w/w.sum()
    Sig = del_**2 * np.random.uniform(size = (gam,d))

    ent = -np.inf

    for iter in range(1_500):
        Eold = ent
        # print("entering regEM", iter)
        [w,mu,Sig,del_,ent] = regEM_n(w,mu,Sig,del_,X)
        err = np.abs( ( ent-Eold ) / ent)
        print(f"Iter.   Tol.              Bandwith \n{iter}        {err}   {del_}")
        if err < 10**(-5):             # Super weird. why not use while loop or just loop up to 200 in first place??
            break

    

    if not grid:
        grid = np.linspace(MIN, MAX, 2**12)
    
    mesh = grid - MIN
    mesh = mesh / scaling

    pdf = probfun_n(mesh, w, mu, Sig) / np.prod(scaling)

    del_ = scaling * del_

    print(f"final band with: {del_}")
    # print(Sig[0,:])
    # Sig : bandwith for each of the observed points

    return [pdf,grid]


def regEM_n(w,mu,Sig,del_,X):
    eps = sys.float_info.epsilon


    (gam,d) = mu.shape
    (n,d) = X.shape
    log_lh = np.zeros((n,gam))
    log_sig = log_lh.copy()

    for i in range(gam):
        L = scipy.linalg.cholesky(Sig[:,:,i], lower=True)
        s = np.diag(L)

        xRinv = scipy.linalg.solve_triangular(L, X - mu[i,:])
        xSig = scipy.linalg.solve_triangular(L, xRinv).sum(axis = 1) + eps

        
        L_inv = scipy.linalg.solve_triangular(L, np.identity(d))


        log_lh[:,i] = -0.5 * xRinv.sum(axis=1) - 0.5*np.log(s).sum() + np.log(w[i]) - d*np.log(2*np.pi)/2 - 0.5*del_**2 * np.trace(L_inv.T @ L_inv)
        log_sig[:,i] = log_lh[:,i] + np.log(xSig)

    # print(log_lh.sum(axis=1).mean())

    maxll = log_lh.max(axis=1)
    maxlsig = log_sig.max(axis=1)
    p = np.exp(log_lh - maxll[:,None])
    psig = np.exp(log_sig - maxlsig[:,None])

    density = p.sum(axis=1)
    psigd = psig.sum(axis=1)

    logpdf = np.log(density) + maxll
    logpsigd = np.log(psigd) + maxlsig

    p = p/density[:,None]
    ent = logpdf.sum()
    w = p.sum(axis=0)

    for i in np.where(w > 0)[0]:
        mu[i,:] = p[:,i] @ X / w[i]
        X_centered = (X - mu[i,:]) * np.sqrt(p[:,i])
        # Sig[i,:] = p[:,i] @ X_centered**2/w[i] + del_**2
        Sig[:,:,i] = X_centered.T @ X_centered/w[i] + del_**2 * np.identity(d)

    w = w/w.sum()
    curv = np.exp(logpsigd - logpdf).mean()                     # estimate curvature
    del_ = 1/(4 * n * (4*np.pi)**(d/2)* curv)**(1/(d+2))

    return [w, mu, Sig, del_, ent]



def probfun_n(x,w,mu,Sig):
    # Higher dimensional KDE (diffusion)
    (gam,d) = mu.shape
    pdf = 0
    for k in range(gam):
        L = scipy.linalg.cholesky(Sig[:,:,k], lower=True)
        s = np.diag(L)

        logpdf = -0.5*(scipy.linalg.solve_triangular(L, x - mu[k,:], lower = True)**2).sum(axis=1) + np.log(w[k]) - np.log(s).sum() - d*np.log(2*np.pi)/2

        pdf += np.exp( logpdf )
    
    return pdf



# tests
def KolmogorovSmirnovTestUnif(U, printOut = True):
    n_samples = U.__len__()
    
    # theoretical distribution (x -> x, when x in [0,1])
    F_H = lambda x : np.min( [np.ones(x.shape), np.max( np.vstack([x , np.zeros(x.shape)]) , axis = 0)], axis = 0 )

    # emperical distribution sampled at many points deviation from theoretical dist.
    xx = np.linspace(0,1,10_000)
    D_N = np.abs((U <= xx[:,None]).mean(axis=1) - F_H(xx)).max()

    # Adjusted test statistic (according to BO slides using no estimated parameters etc.)
    D_N_adjusted = ( np.sqrt(n_samples) + 0.12 + 0.11/np.sqrt(n_samples) ) * D_N

    p_value = scipy.special.kolmogorov(D_N_adjusted)

    if printOut:
        print(f"Test-stat. :          {D_N}")
        print(f"Adjusted test-stat. : {D_N_adjusted}")
        print(f"p-value :             {p_value}")
    return D_N, D_N_adjusted, p_value


# mutual informaiton from correlation (Gaussian)
def GaussianCorrelationToMutualInformation(rho):
    # Warning, may return nan or inf for rho = 1
    return -0.5 * np.log(1 - rho**2)



def ka0(trunc_point, boundary = "lower", h = None):
    if boundary == "lower":
        return scipy.stats.norm.cdf(trunc_point)
    elif boundary == "lower-upper":
        return scipy.stats.norm.cdf(trunc_point) - scipy.stats.norm.cdf(trunc_point - 1/h)

def ka1(trunc_point, boundary = "lower", h = None):
    if boundary == "lower":
        return -scipy.stats.norm.pdf(trunc_point)
    elif boundary == "lower-upper":
        return scipy.stats.norm.pdf(trunc_point - 1/h) - scipy.stats.norm.pdf(trunc_point)

def ka2(trunc_point, boundary = "lower", h = None):
    if boundary == "lower":
        return ka0(trunc_point) + trunc_point * ka1(trunc_point)
    elif boundary == "lower-upper":
        return ka0(trunc_point) + trunc_point * ka1(trunc_point) - (ka0(trunc_point - 1/h) + (trunc_point - 1/h) * ka1(trunc_point - 1/h))

def JonesDensity(x, kern_centers, h, boundary = "lower-upper"):
    trunc_points = x / h

    a0 = ka0(trunc_points, boundary = boundary, h = h)
    a1 = ka1(trunc_points, boundary = boundary, h = h)
    a2 = ka2(trunc_points, boundary = boundary, h = h)

    denom = (a2*a0 - a1**2)
    lx = a2/denom
    mx = a1/denom

    u = (x - kern_centers[:,None]) / h


    return ((lx - mx*u) * scipy.stats.norm.pdf(u)).mean(axis = 0)/h


def KernelJones1D(kern_centers, h = None, boundary = "lower-upper", proper = True):
    n = kern_centers.__len__()
    if h == None:
        h = n**(-1/5) * kern_centers.var()**0.5
    
    def kern(x):

        trunc_points = x / h

        a0 = ka0(trunc_points, boundary = boundary, h = h)
        a1 = ka1(trunc_points, boundary = boundary, h = h)
        a2 = ka2(trunc_points, boundary = boundary, h = h)

        denom = (a2*a0 - a1**2)
        lx = a2/denom
        mx = a1/denom

        u = (x - kern_centers[:,None]) / h


        return ((lx - mx*u) * scipy.stats.norm.pdf(u)).mean(axis = 0)/h
    
    if proper:
        c = scipy.integrate.quad(kern, 0, 1)[0]
    else:
        c = 1
    
    return lambda x : kern(x) / c


def KernelJones2D(kern_centers, h = None, boundary = "lower-upper", proper = True):
    if kern_centers.shape[0] != 2:
        raise AttributeError("Error: kern_centers shoud be a 2xn array")
    
    if (h != None):
        if (h.shape != (2,)):
            raise AttributeError("Error: h should be a (2,) array")
    
    n = kern_centers.shape[1]
    if h == None:
        h = n**(-1/5) * kern_centers.var(axis = 1)**0.5

    h = h[:,None]


    # def kern_x(x):
    #     trunc_points = x / h[0]
    #     a0 = ka0(trunc_points, boundary = boundary, h = h[0])
    #     a1 = ka1(trunc_points, boundary = boundary, h = h[0])
    #     a2 = ka2(trunc_points, boundary = boundary, h = h[0])

    #     denom = (a2*a0 - a1**2)
    #     lx = a2/denom
    #     mx = a1/denom
    #     u = (x - kern_centers[0,:]) / h[0]

    #     return ((lx - mx*u) * scipy.stats.norm.pdf(u)).mean(axis = 0)/h[0]
    
    # if proper:
    #     c_x = scipy.integrate.quad(kern_x, 0, 1)[0]
    # else:
    #     c_x = 1


    # def kern_y(y):
    #     trunc_points = y / h[0]
    #     a0 = ka0(trunc_points, boundary = boundary, h = h[1])
    #     a1 = ka1(trunc_points, boundary = boundary, h = h[1])
    #     a2 = ka2(trunc_points, boundary = boundary, h = h[1])

    #     denom = (a2*a0 - a1**2)
    #     lx = a2/denom
    #     mx = a1/denom
    #     u = (y - kern_centers[1,:]) / h[1]

    #     return ((lx - mx*u) * scipy.stats.norm.pdf(u)).mean(axis = 0)/h[1]
    
    # if proper:
    #     c_y = scipy.integrate.quad(kern_y, 0, 1)[0]
    # else:
    #     c_y = 1

    # c = np.vstack([c_x, c_y])

    
    def kern(x):
        trunc_points = x / h

        a0 = ka0(trunc_points, boundary = boundary, h = h)
        a1 = ka1(trunc_points, boundary = boundary, h = h)
        a2 = ka2(trunc_points, boundary = boundary, h = h)

        denom = (a2*a0 - a1**2)
        lx = a2/denom
        mx = a1/denom

        u = (x[:,:,None] - kern_centers[:,None,:]) / h[:,:,None]

        # print(u.shape)
        # print(lx[:,:,None].shape)

        # print((lx[:,:,None] - mx[:,:,None]*u).shape)
        # print(scipy.stats.norm.pdf(u).shape)

        # print(h[:,:,None].shape)
        # print(c[:,None].shape)
        

        temp = ((lx[:,:,None] - mx[:,:,None]*u) * scipy.stats.norm.pdf(u))/h[:,:,None] # / c[:,None]
        # print(temp.shape)
        return (temp[0,:,:] * temp[1,:,:] ).mean(axis = 1)
    

    
    c = scipy.integrate.dblquad( lambda x,y : kern(np.vstack([x,y])) ,0,1,0,1)[0]
    
    
    
    return lambda x : kern(x) / c


def KernelJones2D_nonnegative(kern_centers, h = None, boundary = "lower-upper", proper = True):
    if kern_centers.shape[0] != 2:
        raise AttributeError("Error: kern_centers shoud be a 2xn array")
    
    if (h != None):
        if (h.shape != (2,)):
            raise AttributeError("Error: h should be a (2,) array")
    
    n = kern_centers.shape[1]
    if h == None:
        h = n**(-1/5) * kern_centers.var(axis = 1)**0.5

    h = h[:,None]


    
    def kern(x):
        trunc_points = x / h

        a0 = ka0(trunc_points, boundary = boundary, h = h)
        a1 = ka1(trunc_points, boundary = boundary, h = h)
        a2 = ka2(trunc_points, boundary = boundary, h = h)

        denom = (a2*a0 - a1**2)
        lx = a2/denom
        mx = a1/denom

        u = (x[:,:,None] - kern_centers[:,None,:]) / h[:,:,None]

        # print(u.shape)
        # print(lx[:,:,None].shape)

        # print((lx[:,:,None] - mx[:,:,None]*u).shape)
        # print(scipy.stats.norm.pdf(u).shape)

        # print(h[:,:,None].shape)
        # print(c[:,None].shape)
        

        f_til = ((lx[:,:,None] - mx[:,:,None]*u) * scipy.stats.norm.pdf(u))/h[:,:,None] # / c[:,None]
        f_til = np.prod(f_til, axis = 0 ).mean(axis = 1)

        f_hat = scipy.stats.norm.pdf(u) / h[:,:,None] / a0[:,:,None]
        f_hat = np.prod(f_hat, axis = 0).mean(axis = 1)
        # print(temp.shape)
        return f_hat * np.exp( f_til / f_hat - 1 )
    
    
    print("Computing normalization constants...", end = "")
    i_res = scipy.integrate.dblquad( lambda x,y : kern(np.vstack([x,y])) ,0,1,0,1)
    c = i_res[0]
    print(" done. " + f"integral acc. : {i_res[1]}")
    
    
    return lambda x : kern(x) / c



def KernelJones2D_nonnegative_version_Jonas(kern_centers, h = None, boundary = "lower-upper", proper = True):
    if kern_centers.shape[0] != 2:
        raise AttributeError("Error: kern_centers shoud be a 2xn array")
    
    if not(h is None):
        if (h.shape != (2,)):
            raise AttributeError("Error: h should be a (2,) array")
    
    n = kern_centers.shape[1]
    if h is None:
        h = n**(-1/5) * kern_centers.var(axis = 1)**0.5

    h = h[:,None]


    
    def kern(x):
        trunc_points = x / h

        a0 = ka0(trunc_points, boundary = boundary, h = h)
        a1 = ka1(trunc_points, boundary = boundary, h = h)
        a2 = ka2(trunc_points, boundary = boundary, h = h)

        denom = (a2*a0 - a1**2)
        lx = a2/denom
        mx = a1/denom

        u = (x[:,:,None] - kern_centers[:,None,:]) / h[:,:,None]

        # print(u.shape)
        # print(lx[:,:,None].shape)

        # print((lx[:,:,None] - mx[:,:,None]*u).shape)
        # print(scipy.stats.norm.pdf(u).shape)

        # print(h[:,:,None].shape)
        # print(c[:,None].shape)
        

        f_til = ((lx[:,:,None] - mx[:,:,None]*u) * scipy.stats.norm.pdf(u))/h[:,:,None] # / c[:,None]
        f_til = np.prod(f_til, axis = 0 ).mean(axis = 1)

        f_hat = scipy.stats.norm.pdf(u) / h[:,:,None] / a0[:,:,None]
        f_hat = np.prod(f_hat, axis = 0).mean(axis = 1)
        # print(temp.shape)
        return f_hat * np.exp( (f_til - f_hat) / ( f_hat + .01) )
    
    
    print("Computing normalization constants...", end = "")
    # i_res = scipy.integrate.dblquad( lambda x,y : kern(np.vstack([x,y])) ,0,1,0,1)
    # c = i_res[0]
    c = 1
    # print(" done. " + f"integral acc. : {i_res[1]}")
    
    
    return lambda x : kern(x) / c


def Kernel2D_GaussCopula(kern_centers, h = None):
    if kern_centers.shape[0] != 2:
        raise AttributeError("Error: kern_centers shoud be a 2xn array")
    
    if not (h is None):
        if (h.shape != (2,)):
            raise AttributeError("Error: h should be a (2,) array")
   

    kern_centers_u = scipy.stats.norm.ppf(kern_centers)

    if h is None:
        mu_hat = kern_centers_u.mean(axis = 1)
        s2_hat = kern_centers_u.var(ddof = 1, axis = 1)
        n = kern_centers_u.shape[1]
        h_GC = s2_hat**0.5 * ( ( 2 * mu_hat**2 * s2_hat + 3 * (1 - s2_hat)**2 ) * n )**(-0.2)
        h = h_GC

        h = n**(-1/5) * kern_centers_u.var(axis = 1)**0.5

    h = h[:,None]

    # print(h.shape)

    exp_fac = (1-h**2) / 2 / h**2 / (2 - h**2)
    c = 1 / (h * (2 - h**2)**0.5)

    # print(kern_centers_u.shape)

    def kern(x):
        xinv = scipy.stats.norm.ppf(x)
        
        emp = np.exp( - (1 - h**2) * xinv**2 * exp_fac)


        temp = 2 * kern_centers_u[:,:,None] @ xinv[:,None,:]
        temp = ( 1 - h[:,None,:]**2 ) * kern_centers_u[:,:,None]**2 - temp
        

        s = np.exp( - exp_fac[:,:,None] * ( ( 1 - h[:,None,:]**2 ) * kern_centers_u[:,:,None]**2 - 2 * kern_centers_u[:,:,None] @ xinv[:,None,:] )) 

        res = s * emp[:,None,:] * c[:,:,None]

        return np.prod(res, axis = 0).mean(axis = 0)
    
    return kern


def gen_gaus_c(rho):
    # x,y in [0,1]
    c = 1 / (1 - rho**2) ** 0.5
    def kern(x,y):
        xinv = scipy.stats.norm.ppf(x)
        yinv = scipy.stats.norm.ppf(y)

        # print(np.abs(xinv).max(), np.abs(yinv).max())
        
        p = -1 / 2 / (1 - rho**2) * (rho**2 * (xinv**2 + yinv**2) - 2 * rho * xinv * yinv)
        return c * np.exp(p)

    return kern