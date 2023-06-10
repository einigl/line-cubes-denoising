""" Distributions helpers """

import warnings
from math import ceil
from typing import Optional

import numpy as np
from scipy.stats import norm

def normal_pdf(t, mu = 0, sigma = 1) :
    """ Returns """
    return norm.pdf(t, loc = mu, scale = sigma)

def l1(p : np.ndarray, q : np.ndarray, dt : float) :
    """ TODO """
    f = np.abs(p-q)
    return dt * f.sum(axis = -1)

def l2(p : np.ndarray, q : np.ndarray, dt : float) :
    """ TODO """
    f = (p-q)**2
    return np.sqrt( dt * f.sum(axis = -1) )

def kld(p : np.ndarray, q : np.ndarray, dt : float, base = None) :
    """ TODO """
    #f = p * np.log(p/q)
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore")
        f = p * (np.log(p)-np.log(q))
    if base is not None :
        f /= np.log(base)
    f[~np.isfinite(f)] = 0
    return dt * f.sum(axis = -1)

def jsd(p : np.ndarray, q : np.ndarray, dt : float, base : Optional[float] = None) :
    """ TODO """
    m = (p+q)/2
    #f = 0.5 * (m * np.log(m/q) + p * np.log(p/m))
    f = 0.5 * (p * (np.log(p)-np.log(m)) + q * (np.log(q)-np.log(m)))
    if base is not None :
        f /= np.log(base)
    f[~np.isfinite(f)] = 0
    return np.sqrt( dt * f.sum(axis = -1) )

def outliers_ratio(t : np.ndarray, p : np.ndarray, q : np.ndarray, dt : float, n_sigma : float) :
    """ TODO """
    prob_p = dt * p[..., np.abs(t) > n_sigma].sum(-1)
    prob_q = dt * q[..., np.abs(t) > n_sigma].sum(-1)
    return prob_p / prob_q