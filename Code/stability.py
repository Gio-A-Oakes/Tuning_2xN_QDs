# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:44:03 2019

Functions required to generate stability diagrams

@author: Giovanni Oakes

"""
import itertools
import numpy as np
import scipy.signal as ss


def matrix_data(I, Q, Magnitude, Phase, sizeV, sizeS):
    """
    Transform 2xm array to nxn matrix, this is necessary to then obtain FFT data

    :param I: values of I in volts
    :param Q: values of Q in volts
    :param Magnitude: value of Magnitude in volts
    :param Phase: values of phase in volts
    :param sizeV: number of pixels along y axis
    :param sizeS: number of pixels along x axis (int(len(Vg2) / sizeV))
    :return: vi, vq, vm, vp
    """

    vi, vq, vm, vp = np.zeros(shape=(sizeS, sizeV)), np.zeros(shape=(sizeS, sizeV)), np.zeros(
        shape=(sizeS, sizeV)), np.zeros(shape=(sizeS, sizeV))

    for i in range(0, sizeS):
        vi[i, :] = I[range(sizeV * i, sizeV * (i + 1))]
        vq[i, :] = Q[range(sizeV * i, sizeV * (i + 1))]
        vm[i, :] = Magnitude[range(sizeV * i, sizeV * (i + 1))]
        vp[i, :] = Phase[range(sizeV * i, sizeV * (i + 1))]

    return vi, vq, vm, vp


def random_c(ci, Cc, Cm, n_qd, ratio):
    """
    Generates random capacitance matrix when inputting an average value for:
    :param ci: Gate capacitance
    :param Cc: Cross capacitance
    :param Cm: Mutual capacitance
    :param n_qd: number of QDs
    :param ratio: ratio between capacitance parallel, perpendicular or diagonally across the nano-wire
    :return: C, capacitance matrix
    """
    # Setting up capacitance matrix
    C, CM = np.zeros(shape=(n_qd, n_qd)), []
    Cg = abs(np.random.normal(ci, ci / 20, int(n_qd)))  # Gate capacitance
    CC = np.identity(n_qd) * Cg
    # Capacitor perpendicular to the nano-wire
    for j in range(int(n_qd // 2)):
        C[j * 2, j * 2 + 1] = C[j * 2 + 1, j * 2] = -abs(np.random.uniform(Cm, Cm / 20, 1)) / ratio[0]
        CC[j * 2, j * 2 + 1] = abs(np.random.uniform(Cc, Cc / 20, 1)) / ratio[0]
        CC[j * 2 + 1, j * 2] = abs(np.random.uniform(Cc, Cc / 20, 1)) / ratio[0]
        CM = np.append(CM, abs(C[j * 2, j * 2 + 1]))
    # Capacitor parallel to the nano-wire
    for k in range(n_qd - 2):
        C[k, k + 2] = C[k + 2, k] = -abs(np.random.uniform(Cm, Cm / 20, 1)) / ratio[1]
        CC[k, k + 2] = abs(np.random.uniform(Cc, Cc / 20, 1)) / ratio[1]
        CC[k + 2, k] = abs(np.random.uniform(Cc, Cc / 20, 1)) / ratio[1]
        CM = np.append(CM, abs(C[k, k + 2]))
    # Capacitor diagonally across to the nano-wire
    for l in range((n_qd - 2) // 2):
        C[2 * l, 2 * l + 3] = C[2 * l + 3, 2 * l] = -abs(np.random.uniform(Cm, Cm / 20, 1)) / ratio[2]
        C[2 * l + 1, 2 * l + 2] = C[2 * l + 2, 2 * l + 1] = -abs(np.random.uniform(Cm, Cm / 20, 1)) / ratio[2]
        C[l, 3 - l] = C[3 - l, l] = -abs(np.random.uniform(Cm, Cm / 20, 1)) / ratio[2]
        CC[2 * l, 2 * l + 3], CC[2 * l + 3, 2 * l] = abs(np.random.uniform(Cc, Cc / 20, 1)) / ratio[2], abs(
            np.random.uniform(Cc, Cc / 20, 1)) / ratio[2]
        CC[2 * l + 1, 2 * l + 2], CC[2 * l + 2, 2 * l + 1] = abs(np.random.uniform(Cc, Cc / 20, 1)) / ratio[2], abs(
            np.random.uniform(Cc, Cc / 20, 1)) / ratio[2]
        CM = np.append(CM, np.append(abs(C[2 * l, 2 * l + 3]), abs(C[2 * l + 1, 2 * l + 2])))

    for i in range(0, n_qd):
        C[i, i] = np.sum(abs(C[i])) + np.sum(CC[i])

    return C, Cg, CC, CM


def n_states(n_qd: int, max_e: int, diff: int):
    """
    Determines all possible electron configurations within a stability map
    :param n_qd: number of QDs
    :param max_e: maximum number of electrons within a specific QD
    :param diff: maximum difference between the two most populated QDs
    :return: N, all possible electron configurations considered
    """
    # Number of possible electron configurations taken into account,
    # the range determines the maximum electron configuration taken into account
    N = np.fromiter(itertools.product(range(0, max_e), repeat=n_qd), np.dtype('u1,' * n_qd))
    N = N.view('u1').reshape(-1, n_qd)

    # Eliminates transitions that are unlikely to be observed in stability
    # map range that is calculated. Can change the difference between max1 and max2
    # or completely comment this part out if needed
    n = np.array([])
    for i in range(0, len(N)):
        max1 = N[i, np.argmax(N[i])]
        max2 = np.delete(N[i], np.argmax(N[i]))
        max2 = max2[np.argmax(max2)]
        if max1 <= max2 + diff:
            n = np.append(n, N[i])

    return np.reshape(n, (int(len(n) / n_qd), n_qd))


def energy_tensor(N, V, C, CC):
    """
    Finds the electron configuration within the ones set in N that gives the lowest energy and assigns the index of
    the state of N to that particular value
    :param N: all possible electron configurations considered
    :param V: voltages being applied
    :param C: capacitance matrix
    :param CC: cross capacitance matrix
    :return: 2D array of indices of N that gave the lowest energy for that particular set of voltages
    """
    q_all = np.einsum('ij,jklm', CC, V) - np.tensordot(np.transpose(N), np.ones(np.shape(V[0, 0])), axes=0)
    inverse_c = np.linalg.inv(C)
    volt = np.einsum('ij,jklm', inverse_c, q_all)
    u = 0.5 * np.multiply(q_all, volt)
    E = u.sum(axis=0)
    return np.argmin(E, axis=0)


def voltage(N_QD, freq, res, N, Cg, dots):
    """
    Creates numpy array with applied voltages considered
    :param N_QD: number of QDs
    :param freq: number of repeating honeycombs in stability diagram
    :param res: resolution (number of pixels rastered)
    :param N: possible electron configurations
    :param Cg: array of gate capacitance
    :param dots: array of which two QDs are being probed
    :return: numpy array of voltages applied
    """
    V = np.zeros((N_QD, len(N), res + 1, res + 1))
    v = np.tile(np.arange(0, res + 1, 1) * freq / res, (res + 1, 1))
    V[dots[0]] = np.tile(v / Cg[dots[0]], (len(N), 1, 1))
    V[dots[1]] = np.tile(np.transpose(v) / Cg[dots[1]], (len(N), 1, 1))
    return V


def virtual_volt(v, g):
    """
    Gives array of gate voltages that need to be applied in order to operate in virtual voltage space
    :param v: Virtual voltage array that would like to probe
    :param g: Transformation matrix G
    :return: Array of voltages in terms of gate voltages to navigate in virtual voltage space
    """
    a = np.linalg.inv(g)
    return np.einsum('ij, jklm', a, v)


def transition(st, res, signal, blur):
    """
    Transforms array of electron configurations from energy_tensor into a stability diagram with added noise
    :param st: array of electron configuration (output from energy_tensor)
    :param res: resolution (number of pixels rastered)
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
    blurred_signal = ss.convolve(signal, kernel, mode='same')

    # Adding noise to signal
    gauss = np.random.randn(res, res) / res
    intensity = blurred_signal + np.random.normal(np.mean(blurred_signal), 1 / blur, np.shape(blurred_signal)) + \
                np.random.poisson(np.max(blurred_signal) + 1 / blur,
                                  np.shape(blurred_signal))  # Adding gaussian and poisson noise to the intensity
    intensity = 2 * intensity + intensity * gauss  # Adding speckle noise

    return intensity


def stability_diagram(N_QD: int, res: int, dots, ratio, blur, freq):
    """
     Generates random capacitance matrix (given in units of e) and corresponding stability diagram
     :param N_QD: number of quantum dots
     :param res: resolution in Vg1 and Vg2
     :param ratio: ratio of Cm and CC capacitance due to geometry of system
     :param dots: QDs being probed
     :param freq: number of repeating honeycombs in stability diagram
     :param blur: number of pixels to blur the sample by
     :return: stability diagram
     """
    c = sorted(np.random.uniform(1, 0.01, 3))  # Generate random average values for capacitors
    Cc, Cm, Ci = c[0], c[1], c[2]  # Average mutual, cross and gate capacitance for device
    signal = np.random.uniform(50, 100, 1)  # Random amount of noise introduced
    C, Cg, CC, CM = random_c(Ci, Cc, Cm, N_QD, ratio)
    N = n_states(N_QD, freq + 3, freq + 2)
    V = voltage(N_QD, freq, res, N, Cg, dots)
    st = energy_tensor(N, V, C, CC)
    intensity = transition(st, res, signal, blur)
    x, y, I = matrix_to_array(intensity)
    x, y = x / Cg[dots[0]] / res * freq, y / Cg[dots[1]] / res * freq
    return x, y, I, C, CC


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
    :param g: G transformation matrix for a DQD
    :param xs: Data points in x
    :param ys: Data points in y
    :return: Rotated data into virtual voltage space
    """
    return np.matmul(g, np.vstack([xs, ys]))


def matrix_to_array(int_matrix):
    """
    Converts stability diagram matrix into x, y and intensity arrays
    :param int_matrix: stability diagram matrix
    :return: x, y, intensity
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
    :param c: capacitance matrix
    :param cc: cross capacitance matrix
    :return: array of ascending order of gradients
    """
    c1, c2 = c[0, 0], c[1, 1]
    cg1, cg2, cm = cc[0, 0], cc[1, 1], -c[0, 1]
    return np.array((-(cg1 * c2) / (cg2 * cm),
                     -(cg1 * cm) / (cg2 * c1),
                     (cg1 * (cm - c2)) / (cg2 * (cm - c1))))
