
import numpy as np
from matplotlib.colors import ListedColormap, Colormap, LinearSegmentedColormap


import pandas as pd
import scipy
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from matplotlib import cm

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from networkx.drawing.nx_agraph import graphviz_layout


colormap_corr = LinearSegmentedColormap.from_list("", ["#8B0000","white","#00008B"])

def ND(G_obs, alpha = 1, beta = 0.99):
    G = G_obs.copy()   
    n = G.shape[0]
    
    # Filter out small values
    q = np.quantile(G[np.triu_indices(n)], 1-alpha)
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