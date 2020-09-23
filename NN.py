import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import skopt as sk
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as k

import stability as stab
from Data_processing import hough_theta, averaging_xy, threshold_theoretical


def stability_generator(n_qd, res):
    if n_qd == 2:
        x, y, z, c, cc, dots = stab.stab_dqd(res)
        return x, y, z, c, cc, dots
    elif n_qd == 4:
        x, y, z, c, cc, dots = stab.stab_fqd(res)
        return x, y, z, c, cc, dots
    else:
        print('Error, n_qd can only take values of 2 or 4')


def stab_red(n_qd, res):
    vg1, vg2, intensity, c, cc, dots = stability_generator(n_qd, res)
    x, y, z, _, _, _ = threshold_theoretical(vg1, vg2, intensity)
    xs, ys, _ = averaging_xy(x, y, z, int((len(x)) ** 0.5 * 10), int((len(x)) ** 0.2))  # Averaging points
    return xs, ys, c, cc, dots


def line_to_theta(n_qd: int, res: int):
    xs, ys, c, cc, dots = stab_red(n_qd, res)
    theta = np.reshape(np.linspace(0, np.pi / 2, 500 + 1), (1, 500 + 1))  # Creating theta array
    # frequency and theta values extracted from Hough transform
    freq_t = hough_theta(np.transpose(np.vstack([xs, ys])), theta)
    p = stab.analytical_grad(c, cc, dots)
    return freq_t, p[:2]


def create_grad_library(n: int):
    """
    Runs line_to_theta multiple times to create a pandas data-frame of histograms with corresponding gradients
    @param n: number of inputs in data-frame
    @return: pandas data-frame
    """
    # TODO try to parallelize for loop
    index = list(np.arange(0, 501 + 2, 1))
    df = pd.DataFrame(columns=index)

    for i in range(n):
        try:
            hist, grad = line_to_theta(int(2 * np.random.randint(1, 3, 1)), 300)
            df = df.append(np.transpose(pd.DataFrame(np.append(hist, grad))))
        except:
            print('An error occurred')

    return df


def save_df(df, name):
    df.to_csv((str(name) + '.csv'), index=False)


def create_model(learning_rate, num_dense_layers, num_dense_nodes, dropout=0.05):
    """
    Creates model architecture for a NN to train on given different parameters
    @param learning_rate: learning rate of NN
    @param num_dense_layers: number of layers in the NN
    @param num_dense_nodes: number of nodes on each layer
    @param dropout: dropout probability
    @return: model to be trained
    """
    # This model used ReLU for activation and a he_uniform initializer
    model = Sequential()
    model.add(Dense(501, activation='relu', input_shape=(501,), kernel_initializer=initializers.he_uniform()))
    # create a loop making a new dense layer for the amount passed to this model.
    # naming the layers helps avoid tensorflow error deep in the stack trace.
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(num_dense_nodes,
                        activation='relu',
                        name=name,
                        kernel_initializer=initializers.he_uniform()
                        ))
        model.add(Dropout(dropout))
    # add our classification layer.
    model.add(Dense(3, activation='linear', kernel_initializer=initializers.he_uniform()))

    # setup our optimizer and compile
    adam = Adam(lr=learning_rate)  # We use Adam as an optimiser
    model.compile(optimizer=adam, loss='mean_squared_error',  # MSE is the quantity we want to minimise
                  metrics=["mean_squared_error"])
    return model


def bayesian_optimisation(x_train, y_train, param_0):
    """
    Uses Bayesian optimisation to find optimal hyper-parameters to tune the NN
    @param x_train: x values of training data
    @param y_train: y values of training data
    @param param_0: initial parameters
    @return: optimised hyper-parameters for NN
    """
    # Set boundaries within which optimized hyper-parameters need to be in
    dim_learning_rate = sk.space.Real(low=1e-4, high=1e-2, prior='log-uniform',
                                      name='learning_rate')
    dim_num_dense_layers = sk.space.Integer(low=1, high=5, name='num_dense_layers')
    dim_num_dense_nodes = sk.space.Integer(low=30, high=512, name='num_dense_nodes')
    # dim_dropout = sk.space.Real(low=0, high=0.5, name='dropout')

    dimensions = [dim_learning_rate,
                  dim_num_dense_layers,
                  dim_num_dense_nodes
                  # dim_dropout
                  ]

    # Function to evaluate validation MSE for given hyper-parameters
    @sk.utils.use_named_args(dimensions=dimensions)
    def fitness(learning_rate, num_dense_layers, num_dense_nodes, dropout):
        # create model with given hyper-parameters
        model = create_model(learning_rate=learning_rate,
                             num_dense_layers=num_dense_layers,
                             num_dense_nodes=num_dense_nodes,
                             # dropout=dropout
                             )
        # Train NN with given hyper-parameters
        blackbox = model.fit(x=x_train.values,  # named blackbox because it represents the structure
                             y=y_train.values,
                             epochs=30,
                             validation_split=0.15,
                             )
        # return the MSE for the last epoch.
        rms = blackbox.history['val_mean_squared_error'][-1]
        print()
        print("MSE: {}".format(rms))
        print()
        del model

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        k.clear_session()

        return rms

    # Use GP optimization to find best set of hyper-parameters
    gp_result = sk.gp_minimize(func=fitness,
                               dimensions=dimensions,
                               n_calls=500,
                               x0=param_0)
    return gp_result


def nn_training_data(name: str, split=0.10):
    """
    Splits and normalises data_frame into train and test data
    @param split: split between train and test data, default is 0.1
    @param name: Name of pandas data frame to load
    @return:
    """
    # load data
    df = pd.read_csv(name, index_col=False)
    x, y = df.iloc[:, :501], df.iloc[:, 501:]
    y = np.arctan(-1 / y)
    # Split between training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
    # use range to normalise data
    r_x, y_train = np.std(x_train.values, axis=0), y_train * 2 / np.pi
    x_train, x_test = x_train / r_x, x_test / r_x
    return x_train, y_train, x_test, y_test, r_x


def fit_model(model, x_train, y_train, val_split=0.15):
    black_box = model.fit(x_train.values, y_train.values, epochs=100, verbose=0, validation_split=val_split)
    return black_box


def predict_model(model, x_test):
    return model.predict(x_test.values) * np.pi / 2


def evaluation(black_box):
    """
    Plots learning curve of trained model
    @param black_box: trained model
    @return:
    """
    history = black_box.history
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history['mean_squared_error'], c='k')
    plt.plot(history['val_mean_squared_error'], c='r')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], c='k')
    plt.plot(history['val_loss'], c='r')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(('Test', 'Validation'))
    plt.tight_layout()
    return history


def save_model(name, model, r_x):
    model.save((name + '.h5'))
    drx = pd.DataFrame(r_x)
    drx.to_csv(('rx_' + name + '.csv'), index=False)


def load_model(name):
    model = tf.keras.models.load_model((name + '.h5'))
    r_x = pd.read_csv(('rx_' + name + '.csv'), index_col=False)
    return model, r_x


def predict_exp(model, freq, r_x):
    index = list(np.arange(0, 501, 1))
    df = pd.DataFrame(columns=index)
    df = df.append(np.transpose(pd.DataFrame(freq)))
    df = df / np.transpose(r_x.values)
    t = model.predict(df.values) * np.pi / 2
    return -1 / np.tan(t[0])
