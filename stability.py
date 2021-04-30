# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:44:03 2019

Functions required to generate stability diagrams

@author: Giovanni Oakes

"""
import itertools
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt


def rand_c(cs, r):
    return abs(np.random.normal(cs, cs / 10, 1)) * r


def random_c(ci, cc, cm, n_qd, ratio):
    """
    Generates random capacitance matrix when inputting an average value for:
    @param ci: Gate capacitance
    @param cc: Cross capacitance
    @param cm: Mutual capacitance
    @param n_qd: number of QDs
    @return: c, capacitance matrix
    @param ratio: ratio between capacitance parallel, perpendicular and diagonally across the nano-wire
    """
    # Setting up capacitance and cross capacitance matrices
    c, cg = np.zeros(shape=(n_qd, n_qd)), abs(np.random.normal(ci, ci / 5, int(n_qd)))  # C matrix and Gate capacitance
    ccs = np.identity(n_qd) * cg
    # capacitance perpendicular to the nano-wire
    for i in range(int(n_qd // 2)):
        c[i * 2, i * 2 + 1] = c[i * 2 + 1, i * 2] = -rand_c(cm, ratio[2])
        ccs[i * 2, i * 2 + 1], ccs[i * 2 + 1, i * 2] = rand_c(cc, ratio[2]), rand_c(cc, ratio[2])
    # capacitance parallel to the nano-wire
    for j in range(n_qd - 2):
        c[j, j + 2] = c[j + 2, j] = -rand_c(cm, ratio[1])
        ccs[j, j + 2], ccs[j + 2, j] = rand_c(cc, ratio[1]), rand_c(cc, ratio[1])
    # capacitance diagonally across the nano-wire
    for k in range((n_qd - 2) // 2):
        c[2 * k, 2 * k + 3] = c[2 * k + 3, 2 * k] = -rand_c(cm, ratio[0])
        c[2 * k + 1, 2 * k + 2] = c[2 * k + 2, 2 * k + 1] = -rand_c(cm, ratio[0])
        c[k, 3 - k] = c[3 - k, k] = -rand_c(cm, ratio[0])
        ccs[2 * k, 2 * k + 3], ccs[2 * k + 3, 2 * k] = rand_c(cc, ratio[0]), rand_c(cc, ratio[0])
        ccs[2 * k + 1, 2 * k + 2], ccs[2 * k + 2, 2 * k + 1] = rand_c(cc, ratio[0]), rand_c(cc, ratio[0])
    # Total capacitance on dot i
    for i in range(0, n_qd):
        c[i, i] = np.sum(abs(c[i])) + np.sum(ccs[i])
    return c, cg, ccs


def n_states(n_qd: int, max_e: int, diff: int):
    """
    Determines all possible electron configurations within a stability map
    :param n_qd: number of QDs
    :param max_e: maximum number of electrons within a specific QD
    :param diff: maximum difference between the two most populated QDs
    :return: n_st, all possible electron configurations considered
    """
    # Number of possible electron configurations taken into account,
    # the range determines the maximum electron configuration taken into account
    n_st = np.fromiter(itertools.product(range(0, max_e), repeat=n_qd), np.dtype('u1,' * n_qd))
    n_st = n_st.view('u1').reshape(-1, n_qd)

    # Eliminates transitions that are unlikely to be observed in stability
    # map range that is calculated. Can change the difference between max1 and max2
    # or completely comment this part out if needed
    n = np.array([])
    for i in range(0, len(n_st)):
        max1 = n_st[i, np.argmax(n_st[i])]
        max2 = np.delete(n_st[i], np.argmax(n_st[i]))
        max2 = max2[np.argmax(max2)]
        if max1 <= max2 + diff:
            n = np.append(n, n_st[i])

    return np.reshape(n, (int(len(n) / n_qd), n_qd))


def reduced_n(n_st, el, dots):
    n = n_states(2, el + 1, el)
    ns = np.zeros((len(n_st) * len(n), 4))
    ns[:, dots[0]] = np.tile(n_st[:, 0], len(n))
    ns[:, dots[1]] = np.tile(n_st[:, 1], len(n))
    nt = []
    for i in range(len(n)):
        nt = np.append(nt, np.ones(np.shape(n_st)) * n[i])
    nt = np.reshape(nt, (len(n_st) * len(n), 2))
    diff = np.setdiff1d([0, 1, 2, 3], dots)
    ns[:, diff[0]], ns[:, diff[1]] = nt[:, 0], nt[:, 1]
    return ns


def energy_tensor(n, v, c, cc):
    """
    Finds the electron configuration within the ones set in N that gives the lowest energy and assigns the index of
    the state of N to that particular value
    :param n: all possible electron configurations considered
    :param v: voltages being applied
    :param c: capacitance matrix
    :param cc: cross capacitance matrix
    :return: 2D array of indices of N that gave the lowest energy for that particular set of voltages
    """
    q_all = np.einsum('ij,jklm', cc, v) - np.tensordot(np.transpose(n), np.ones(np.shape(v[0, 0])), axes=0)
    inverse_c = np.linalg.inv(c)
    volt = np.einsum('ij,jklm', inverse_c, q_all)
    u = 0.5 * np.multiply(q_all, volt)
    energy = u.sum(axis=0)
    return np.argmin(energy, axis=0)


def voltage(n_qd, freq, res, n, cg, dots):
    """
    Creates numpy array with applied voltages considered
    :param n_qd: number of QDs
    :param freq: number of repeating honeycombs in stability diagram
    :param res: resolution (number of pixels)
    :param n: possible electron configurations
    :param cg: array of gate capacitance
    :param dots: array of which two QDs are being probed
    :return: numpy array of voltages applied
    """
    vs = np.zeros((n_qd, len(n), res + 1, res + 1))
    v = np.tile(np.arange(0, res + 1, 1) * freq / res, (res + 1, 1))
    vs[dots[0]] = np.tile(v / cg[dots[0]], (len(n), 1, 1))
    vs[dots[1]] = np.tile(np.transpose(v) / cg[dots[1]], (len(n), 1, 1))
    return vs


def virtual_volt(v, g):
    """
    Gives array of gate voltages that need to be applied in order to operate in virtual voltage space
    :param v: Virtual voltage array that would like to probe
    :param g: Transformation matrix G
    :return: Array of voltages in terms of gate voltages to navigate in virtual voltage space
    """
    a = np.linalg.inv(g)
    return np.einsum('ij, jklm', a, v)


def add_noise(res, z, blur):
    gauss = np.random.randn(res, res) / res
    # Adding gaussian and poisson noise to the intensity
    z_n = z + np.random.normal(np.mean(z), 1 / blur, np.shape(z)) + np.random.poisson(np.max(z) + 1 / blur, np.shape(z))
    return 2 * z_n + z_n * gauss  # Adding speckle noise


def transition(st, res, signal, blur):
    """
    Transforms array of electron configurations from energy_tensor into a stability diagram with added noise
    :param st: array of electron configuration (output from energy_tensor)
    :param res: resolution (number of pixels)
    :param signal: average signal intensity, defines signal to noise ratio
    :param blur: number of pixels to blur the sample by
    :return: intensity of stability diagram
    """
    # Convert states to transitions
    i1, i2 = np.zeros(shape=(res, res)), np.zeros(shape=(res, res))
    x1, y1 = np.where(st[:-1] != st[1:])
    x2, y2 = np.where(np.transpose(st)[:-1] != np.transpose(st)[1:])
    i1[x1 - 1, y1 - 1] = signal * np.random.uniform(5, 10, 1)
    i2[x2 - 1, y2 - 1] = signal * np.random.uniform(5, 10, 1)

    signal = i1 + np.transpose(i2)  # Pure signal

    # Blur pixels by averaging blur nearest neighbours
    kernel = np.ones((blur, blur)) / blur ** 2
    blurred_signal = convolve(signal, kernel, mode='same')

    # Adding noise to signal
    return add_noise(res, blurred_signal, blur)


def stability_diagram(c, cc, n, v, freq, dots, offset):
    """
     Generates stability diagram given capacitance matrix
     @param offset: voltage offset that might be applied
     @param dots: QDs being probed
     @param freq: number of repeating honeycombs in stability diagram
     @param v: voltages being applied
     @param n: electron configurations being taken into consideration
     @param cc: cross capacitance matrix
     @param c: capacitance matrix
     @return: stability diagram
     """
    signal, blur = np.random.uniform(50, 100, 1), 5
    st = energy_tensor(n, v, c, cc)
    intensity = transition(st, (len(st) - 1), signal, blur)
    x, y, z = matrix_to_array(intensity)
    x = x / cc[dots[0], dots[0]] / (len(st) - 1) * freq + int(offset[0]) / cc[dots[0], dots[0]]
    y = y / cc[dots[1], dots[1]] / (len(st) - 1) * freq + int(offset[1]) / cc[dots[1], dots[1]]
    return x, y, z


def rand_c_matrix(n_qd, ratio):
    c, cg, ccs, con = [], [], [], 3
    # To reduce numerical errors due to matrix inversion, only accept c matrix with a condition number below 1.5
    while con > 1.5:
        rand = np.random.uniform(1, 0.1, 2)
        ci, cm, cc = rand_c(1, 1), rand_c(rand[0], 1), rand_c(rand[0] * rand[1], 1)
        c, cg, ccs = random_c(ci, cc, cm, n_qd, ratio)
        con = np.linalg.cond(c)
    return c, cg, ccs, con


def stab_dqd(res):
    """
    @param res: resolution (number of pixels rastered)
    @return: Stability diagram of a DQD
    """
    con = 3
    while con > 1.25:
        rand = np.random.uniform(1, 0.1, 2)
        ci, cm, cc = rand_c(1, 1), rand_c(rand[0], 1), rand_c(rand[0] * rand[1], 1)
        c, cg, ccs = random_c(ci, cc, cm, 2, np.ones(3))
        con = np.linalg.cond(c)
    freq = int(np.random.randint(3, 6, 1))
    n = n_states(2, freq + 4, freq + 3)
    v = voltage(2, freq, res, n, cg, [0, 1])
    x, y, z = stability_diagram(c, ccs, n, v, freq, [0, 1], np.zeros(2))
    return x, y, z, c, ccs, [0, 1]


def stab_fqd(res):
    """
    @param res: resolution (number of pixels rastered)
    @return: Stability diagram of a 2x2 QD
    """
    con = 3
    ratio, dots = sorted(np.random.uniform(0.3, 1, 3)), [0, int(np.random.randint(1, 4, 1))]
    while con > 1.25:
        rand = np.random.uniform(1, 0.1, 2)
        ci, cm, cc = rand_c(1, 1), rand_c(rand[0], 1), rand_c(rand[0] * rand[1], 1)
        c, cg, ccs = random_c(ci, cc, cm, 4, ratio)
        con = np.linalg.cond(c)
    freq, offset = int(np.random.randint(3, 6, 1)), np.random.randint(1, 7, 2) / cg[dots]
    # Try to reduce amount of RAM required to run
    n = n_states(2, freq + 4, freq + 3)
    ns = reduced_n(n, freq, dots)
    ns[:, dots[0]], ns[:, dots[1]] = ns[:, dots[0]] + int(offset[0]), ns[:, dots[1]] + int(offset[1])
    v = voltage(4, freq, res, ns, cg, dots)
    v[dots[0]], v[dots[1]] = v[dots[0]] + int(offset[0]) / cg[dots[0]], v[dots[1]] + int(offset[1]) / cg[dots[1]]
    x, y, z = stability_diagram(c, ccs, ns, v, freq, dots, offset)
    return x, y, z, c, ccs, dots


def g2_matrix(grad):
    """
    Extract G matrix from gradients for two QDs. For more information see Supplementary Information of:
    Shuttling a single charge across a one-dimensional array of silicon quantum dots
    :param grad: array of gradients in ascending order
    :return: two by two matrix
    """
    g = np.identity(2)
    g[1, 0] = (grad[2] - grad[1]) / (grad[2] - grad[0])
    g[0, 1], g[1, 1] = -grad[1], -grad[0] * g[1, 0]
    return g


def rotate(g, xs, ys):
    """
    Rotate data according to G matrix (this only works for DQD)
    Otherwise use virtual_volt to then calculate stability diagram in virtual voltage space
    @param g: G transformation matrix for a DQD
    @param xs: Data points in x
    @param ys: Data points in y
    @return: Rotated data into virtual voltage space
    """
    return np.matmul(g, np.vstack([xs, ys]))


def matrix_to_array(int_matrix):
    """
    Converts stability diagram matrix into x, y and intensity arrays
    @param int_matrix: stability diagram matrix
    @return: x, y, intensity
    """
    res_x, res_y = len(int_matrix[0]), len(int_matrix)
    x, y = [], []
    x.extend([np.arange(0, res_x, 1) for i in range(res_y)])
    y.extend([[i] * res_y for i in range(res_x)])
    x, y = np.reshape(x, (res_x * res_y,)), np.reshape(y, (res_x * res_y,))
    intensity = np.reshape(int_matrix, (res_x * res_y,))

    return x, y, intensity


def grad_two_dot(c, cc):
    """
    Calculates exact gradients from classical stability map of a two quantum dot (CC not taken into account)
    @param c: capacitance matrix
    @param cc: cross capacitance matrix
    @return: array of ascending order of gradients
    """
    c1, c2 = c[0, 0], c[1, 1]
    cg1, cg2, cm = cc[0, 0], cc[1, 1], -c[0, 1]
    return np.array((-(cg1 * c2) / (cg2 * cm),
                     -(cg1 * cm) / (cg2 * c1),
                     (cg1 * (cm - c2)) / (cg2 * (cm - c1))))


def alpha_matrix(grad):
    """
    Extracts alpha matrix from negative gradients
    :param grad: negative fitted gradients
    :param N_QD: number of QDs
    :return: alpha matrix
    """
    t = grad - np.pi/2
    alpha = np.zeros((2, 2))

    alpha[0, 0], alpha[0, 1] = -np.sin(t[0]), np.cos(t[0])
    alpha[1, 0], alpha[1, 1] = -np.sin(t[1]), np.cos(t[1])

    return alpha / alpha[0, 0]


def analytical_grad(c, cc, dots):
    """
    Calculates exact reservoir to QD gradients from classical stability map of a 2xN array of QDs
    Although this is a more general solution, there can be numerical errors due to matrix inversion
    @param c: capacitance matrix
    @param cc: cross capacitance matrix
    @param dots: quantum dots being probed
    @return: array of ascending order of gradients
    """
    a = np.linalg.inv(c.astype(np.float64)).astype(np.float128)
    rx = -(np.dot(cc[:, dots[0]], a[dots[0]])) / (np.dot(cc[:, dots[1]], a[dots[0]]))
    ry = -(np.dot(cc[:, dots[0]], a[dots[1]])) / (np.dot(cc[:, dots[1]], a[dots[1]]))
    rm = -(np.dot(cc[:, dots[0]], (a[dots[1]] - a[dots[0]]))) / (np.dot(cc[:, dots[1]], (a[dots[1]] - a[dots[0]])))
    return [rx, ry, rm]


def plot_stab(x, y, volt, dots, **kwargs):
    z = kwargs.get('z', None)
    val = dots + np.ones(2)
    if z is not None:
        plt.scatter(x, y, c=z, s=5, cmap='inferno')
    else:
        plt.scatter(x, y, c='k', s=5)
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.gca().set_aspect('equal', adjustable='box')
    if volt == 'V':
        plt.xlabel(r'$V_{g%s}$ (V)' % int(val[0]), fontsize=24)
        plt.ylabel(r'$V_{g%s}$ (V)' % int(val[1]), fontsize=24)
    elif volt == 'U':
        plt.xlabel(r'$U_{%s}$ (V)' % int(val[0]), fontsize=24)
        plt.ylabel(r'$U_{%s}$ (V)' % int(val[1]), fontsize=24)
    plt.tight_layout()


def plot_c(c, cc):
    # Plot table of C matrix used
    fig = plt.figure(figsize=(8, 1))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.table(cellText=np.round(c, 3),
              loc="center"
              )
    ax1.set_title("C matrix")
    ax1.axis("off")
    # Plot table of CC matrix used
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.table(cellText=np.round(cc, 3),
              loc="center"
              )
    ax2.set_title("CC matrix")
    ax2.axis("off")
