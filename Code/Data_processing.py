# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:03:50 2019
Finds Vg1 and Vg2 values above a threshold, determined by the ratio of the areas
of a Gaussian fit of the intensity histogram to the total area of the intensities
@author: Giovanni Oakes
"""

import numpy as np
import scipy.optimize as opt
from scipy.signal import medfilt2d, savgol_filter
from scipy.ndimage import correlate
from sklearn.neighbors import KDTree


def hist_data(z):
    """
    Finds x and y data from histogram
    :param z: input
    :return: x and y
    """
    data = np.histogram(z, bins='scott')
    x = data[1]
    x = np.array([(x[i] + x[i + 1]) / 2 for i in range(0, len(x) - 1)])
    return x, np.array(data[0])


def gauss(x, *params):
    return abs(params[2]) * np.exp(-(x - params[0]) ** 2 / (2 * params[1] ** 2))


def multi_gaussian(x, *params):
    """
    Fits multiple Gaussian distributions, number of which determined by the number of parameters inputted
    """
    y = np.zeros_like(x)
    index = np.arange(0, len(params), 3)
    if index.size > 1:
        for i in range(0, len(params) // 3):
            mu = params[i]
            sig = params[i + len(params) // 3]
            amp = params[i + 2 * len(params) // 3]
            y = y + abs(amp) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    else:
        y = y + abs(params[2]) * np.exp(-(x - params[0]) ** 2 / (2 * params[1] ** 2))
    return y


def multi_gauss_background(x, *params):
    y = np.zeros_like(x)
    index = np.arange(0, len(params) - 2, 3)
    if index.size > 1:
        y = y + params[0] * x + params[1]
        for i in range(0, (len(params) - 2) // 3):
            mu = params[i + 2]
            sig = params[i + 2 + (len(params) - 2) // 3]
            amp = params[i + 2 + 2 * (len(params) - 2) // 3]
            y = y + abs(amp) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    else:
        y = y + params[0] * x + params[1] + abs(params[4]) * np.exp(-(x - params[2]) ** 2 / (2 * params[3] ** 2))
    return y


def greedy_guess(guess, x, y):
    n = (len(guess) - 2) // 3
    m, sig, a = guess[2:n + 2], guess[n + 2:2 * n + 2], guess[2 * n + 2:]
    chi = (y - multi_gauss_background(x, *guess)) / multi_gauss_background(x, *guess)
    chi = savgol_filter(chi, 3, 2)
    m, a = np.append(m, float(x[np.where(chi == np.max(chi))])), np.append(a, float(y[np.where(chi == np.max(chi))]))
    sig = np.append(sig, sig[n - 1] / 2)
    return np.append(guess[:2], np.append(m, np.append(sig, a)))


def gradient(x, y, z):
    """
    Calculates gradient along x and y of intensities to reduce noise
    @param x: x vales
    @param y: y values
    @param z: intensities
    @return:
    """
    m_z = np.reshape(z, (len(np.unique(y)), len(np.unique(x))))# Transform array into matrix
    sg = savgol_filter(m_z, 5, 2) + savgol_filter(m_z, 5, 2, axis=0) # Savgol filter acts as a low pass band filter
    signal = sg - np.mean(sg) + np.mean(m_z)
    return np.reshape(signal, np.shape(x))


def gradient_exp(x, y, z):
    """
    Calculates gradient along x and y of intensities to reduce noise
    @param x: x vales
    @param y: y values
    @param z: intensities
    @return:
    """
    m_z = np.reshape(z, (len(np.unique(x)), len(np.unique(y))))# Transform array into matrix
    diff = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    z_diff = correlate(m_z, diff)
    sg = savgol_filter(z_diff, 5, 2) + savgol_filter(z_diff, 5, 2, axis=0) # Savgol filter acts as a low pass band filter
    signal = sg - np.mean(sg) + np.mean(m_z)
    return np.reshape(signal, np.shape(x))


def filtering(x, y, z):
    m_z = np.reshape(z, (len(np.unique(y)), len(np.unique(x))))  # Transform array into matrix
    s = medfilt2d(m_z)
    return np.reshape(s, (int(len(x)),))


def normalise(z):
    """
    Unity-based normalisation function, such that all values range between 0 and 1
    :param z: Raw data that needs normalising
    :return: Normalised data
    """
    return (z - np.min(z)) / (np.max(z) - np.min(z))


def fit_gauss(z):
    intensity = normalise(z)
    x, y = hist_data(intensity)
    guess = np.append(0, np.append(np.median(y), np.append(np.median(x[np.where(y == np.max(y))]),
                                                           np.append(np.std(x[np.where(y > np.median(y))]),
                                                                     np.max(y)))))

    fit_param, cov = opt.curve_fit(multi_gauss_background, x, y, guess)
    if fit_param[2] > 0.5:
        index = np.where(intensity<fit_param[2]-3*abs(fit_param[3]))
    else:
        index = np.where(intensity>fit_param[2]+3*abs(fit_param[3]))
    return index


def curved_plane(x, y, param):
    return param[0]*x + param[1]*x**2 + param[2]*y + param[3]*y**2 + param[4]*x*y + param[5]


def linear_plane(x, y, param):
    return param[0]*x + param[1]*y + param[2]


def minimise_plane(param, x, y, z):
    return np.sum((z - linear_plane(x, y, param))**2)

def linear(x, z):
    return (np.median(z[np.where(x==np.min(x))])-np.median(z[np.where(x==np.max(x))]))/(np.min(x)-np.max(x))


def remove_background(x, y, z):
    p = gradient_exp(x, y, z)
    param = np.array((linear(x, z), linear(y,z), np.median(p)))
    sol = opt.minimize(minimise_plane, param, args=(x, y, p)) 
    p_n = normalise(p - linear_plane(x, y, sol.x))
    return p_n*(np.max(z)-np.min(z)) + np.min(z)


def get_klpq_div(p_probs, q_probs):
    # Calcualtes the Kullback-Leibler divergence between pi and qi
    kl_div = 0.0
    
    for pi, qi in zip(p_probs, q_probs):
        kl_div += pi*np.log(pi/qi)
    
    return kl_div


def D_KL(threshold, x, y):
    # Finds best fit Gaussian distribution and calculates the corresponding Kullback-Leibler divergence
    index = np.where(np.logical_and(x>=threshold[0], x<=threshold[1]))
    xs, ys = x[index], y[index]
    ys = ys/np.trapz(ys)
    guess = np.append(np.median(xs[np.where(ys == np.max(ys))]),
                      np.append(np.std(xs[np.where(ys > np.median(ys))]),
                                np.max(ys)))

    fit_param, cov = opt.curve_fit(gauss, xs, ys, guess)
    return get_klpq_div(ys+1, gauss(xs, *fit_param)+1)


def minimise_DKL(x, y):
    # Estimate first guess and boundaries to use:
    guess = np.append(np.median(x[np.where(y == np.max(y))]),
                      np.append(np.std(x[np.where(y > np.median(y))]),
                                np.max(y)))

    fit_param, cov = opt.curve_fit(gauss, x, y, guess)    
    x0 = [fit_param[0]-2*fit_param[1], fit_param[0]+2*fit_param[1]]
    bound = ((np.min(x), fit_param[0]-fit_param[1]), (fit_param[0]+fit_param[1], np.max(x)))
    # Find optimal bound solutions
    sol = opt.minimize(D_KL, x0, jac=None, method='L-BFGS-B', options={'eps':1/len(x)}, args=(x, y), bounds=bound)
    return sol.x


def threshold_DKL(z):
    
    intensity = normalise(z)
    x, y = hist_data(intensity)
    y = y**0.5 # Broadens peak to allow to identify finer structure in the intensity
    
    threshold = minimise_DKL(x, y)
    index = np.where(np.logical_or(intensity<=threshold[0], intensity>=threshold[1]))
        
    return index


def intense(z, index):
    x, y = hist_data(z)
    guess = np.append(np.median(x[np.where(y == np.max(y))]),
                      np.append(np.std(x[np.where(y > np.median(y))]),
                                np.max(y)))
    fit_param, cov = opt.curve_fit(gauss, x, y, guess)  
    return z[index]-fit_param[0]


def threshold_experimental(vg1, vg2, i, q):
    
    i_g, q_g = remove_background(vg1, vg2, i), remove_background(vg1, vg2, q)
    m_i, m_q = threshold_DKL(i_g), threshold_DKL(q_g)
    index = np.unique(np.append(m_i, m_q))
    intensity = normalise(abs(intense(i_g, index)))+normalise(abs(intense(q_g, index)))
    return vg1[index], vg2[index], intensity


def threshold_theoretical(vg1, vg2, i):
    i_g = gradient(vg1, vg2, i)
    x, y = hist_data(i_g)
    x = normalise(x)
    fit_param = [np.median(x[np.where(y == np.max(y))]), np.std(x[np.where(y > np.median(y))]), np.max(y)]
    try:
        fit_one, _ = opt.curve_fit(multi_gaussian, x, y, fit_param)
        ind = np.where(x > fit_one[0] + fit_one[1])
        ys = y[ind] - multi_gaussian(x[ind], *fit_one)
        guess = [fit_one[0], np.median(x[ind][np.where(ys == np.max(ys))]),
                 fit_one[1], np.std(x[np.where(y > np.median(ys))]),
                 fit_one[2], np.max(ys)]
        try:
            fit_param, cov = opt.curve_fit(multi_gaussian, x, y, guess)
            error = np.sqrt(np.diag(cov))
            if error[1] * 10 > error[0]:
                index = np.where(normalise(i) > fit_param[1])
            else:
                index = np.where(normalise(i) > 0.4)
        except:
            val = np.min(x[np.where(x > fit_one[0] + fit_one[1])])
            index = np.where(normalise(i) > val)
    except:
        index = np.where(normalise(i) > 0.4)

    return vg1[index], vg2[index], i[index], x, y, fit_param


def averaging_xy(x, y, intensity, leaf, n_neighbours):
    """
    Uses KDTree to find n_neighbours and then calculates a weighted mean, resulting in thinning the data
    :param x: threshold x values
    :param y: threshold y values
    :param intensity: corresponding intensities
    :param leaf: determines how many neighbouring points to check, leaf > n_neighbours
    :param n_neighbours: number of neighbours to average through
    :return: thinned x and y values
    """
    data = np.transpose(np.vstack([x, y]))
    xs, ys, zs = [], [], []
    tree = KDTree(data, leaf_size=leaf)  # Finds relation between points
    for i in range(0, len(data) // n_neighbours):
        # Figure out which are the neighbouring points
        dist, ind = tree.query(np.reshape(data[i * n_neighbours, :], (1, -1)), k=n_neighbours)
        # takes weighted average of x and y values of given point
        x_m, y_m = np.average(x[ind], weights=intensity[ind]), np.average(y[ind], weights=intensity[ind])
        z_m = np.average(intensity[ind])
        xs, ys, zs = np.append(xs, x_m), np.append(ys, y_m), np.append(zs, z_m)
    return xs, ys, zs


def normalise_hough(data):
    """
    Normalised input data and sets theta range
    :param data: input data of threshold data points
    :return: theta, x and y
    """
    x = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
    y = (data[:, 1] - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))
    f = (np.max(data[:, 1]) - np.min(data[:, 1])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
    return x, y, f


def threshold_hough(theta, accumulator):
    """
    Takes the accumulator matrix and thresholds the data values
    :param theta: array of theta values used
    :param accumulator: accumulator of rho
    :return: threshold_theta, threshold_d
    """
    t = np.tile(theta, len(accumulator))
    h, angle, d = np.histogram2d(t, accumulator, bins=len(theta))  # Creating histogram of intensities
    index_t, index_d = np.where(h > np.max(h) / 3)
    threshold_theta, threshold_d = angle[index_t], d[index_d]  # Threshold values of angle and d
    return threshold_theta, threshold_d


def hough_transform(samples, theta):
    """
    Uses Hough transform to determine lines in stability diagram
    :param samples: data points above threshold
    :param theta: range of theta values taken into account
    :return: grad and y-spacing
    """
    x, y = np.reshape(samples[:, 0], (len(samples), 1)), np.reshape(samples[:, 1], (len(samples), 1))
    accumulator = np.matmul(x, np.cos(theta)) + np.matmul(y, np.sin(theta))

    return np.reshape(accumulator, (len(samples) * len(theta[0])))


def hough(data, theta, **kwargs):
    """
    Applies Hough transform on threshold data
    @param data: normalised x and y vales
    @param theta: theta values
    """
    w = kwargs.get('weights', None)
    a = hough_transform(data, theta)
    t = np.tile(theta, len(a) // len(theta[0]))
    if w is not None:
        weight = np.tile(w, len(theta[0]))
        h, angle, d = np.histogram2d(t[0], a, bins=len(theta[0]), weights=weight)
    else:
        h, angle, d = np.histogram2d(t[0], a, bins=len(theta[0]))
    hs = np.reshape(h, (len(theta[0]) * len(theta[0]), 1))
    hs = hs[np.where(hs > 0)]
    x, y = hist_data(hs)  # h histogram seems to follow exponential like decay
    x, y = x[np.where(y > 0)], y[np.where(y > 0)]
    h_min = np.min(x[np.where(y < np.max(y) / 100)])
    index_t, index_d = np.where(h > h_min)
    return h, angle, d, index_t, index_d, a


def hough_distribution(data, theta):
    """
    Estimates angles and distribution of difference in neighbouring rhos
    @param data: normalised x and y vales
    @param theta: theta values
    @return: distribution of threshold theta values and difference in rhos
    """
    h, angle, d, index_t, index_d, _ = hough(data, theta)
    i_p, _ = np.histogram(angle[index_t], weights=h[index_t, index_d] ** 2, bins=int(len(theta[0])),
                          range=(-np.pi / 2, np.pi / 2))
    i_d, _ = np.histogram(d[index_d], weights=h[index_t, index_d] ** 2, bins=len(theta[0]), range=(-2, 2))
    i_d = np.diff(i_d)
    return i_p / np.trapz(i_p), i_d / np.max(i_d)


def hough_theta(data, theta, **kwargs):
    """
     Estimates angles extracted
     @param data: normalised x and y vales
     @param theta: theta values
     @return: distribution of threshold theta values
     """
    w = kwargs.get('weights', None)
    h, angle, d, index_t, index_d, _ = hough(data, theta, weights=w)
    i_p, _ = np.histogram(angle[index_t], weights=h[index_t, index_d] ** 2, bins=int(len(theta[0])),
                          range=(0, np.pi / 2))
    return i_p / np.trapz(i_p)


def line_fit(x, y, z):
    """
    Obtain gradients and intercepts of the lines within a stability diagram

    :param x: Voltages along x axis
    :param y: Voltages along y axis
    :param z: Intensity
    :return: gradients, y-intercepts and R^2 values separated as positive or negative gradients
    """
    # Reduce number of points by taking the weighted average along x and y
    xs, ys, _ = averaging_xy(x, y, z, 50, 10)
    theta = np.reshape(np.linspace(0, np.pi / 2, 500 + 1), (1, 500 + 1))
    # frequency and theta values extracted from Hough transform
    freq = hough_theta(np.transpose(np.vstack([xs, ys])), theta)

    return xs, ys, freq
