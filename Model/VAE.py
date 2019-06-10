from Model.Layers import *
from Model.utils import *
import tensorflow as tf
import numpy as np
import h5py


class MeshAE:
    def __init__(self, matpath, gc_dim, fc_dim, max_degree=2, sparse=False, padding=False, result_max1=0.9, result_min1=-0.9):
        self.logdr, self.s, self.e_neighbour, self.p_neighbour,\
            self.degree, self.logdr_min, self.logdr_max, self.ds_min,\
            self.ds_max, self.modelnum, self.pointnum, self.edgenum, self._old_maxdegree\
            = load_data(matpath, logdr_ismap=True, s_ismap=False)
        self.gc_dim = gc_dim
        self.fc_dim = fc_dim
        self.max_degree = max_degree
        self.sparse = sparse

        """Make Placholder for RIMD"""
        logdr_shape = np.shape(self.logdr)
        self.placeholder_logdr = tf.placeholder(tf.float32, [None, logdr_shape[1], logdr_shape[2]])

        s_shape = np.shape(self.s)
        self.placeholder_s = tf.placeholder(tf.float32, [None, s_shape[1], s_shape[2]])

        """Make Placeholder for Input"""
        self.input = tf.placeholder(tf.float32, [None, logdr_shape[1], 3])
        
        """Get Chebyshev Sequence"""
        self.cheb_e = chebyshev_polynomials(self.e_neighbour, self.max_degree)
        self.cheb_p = chebyshev_polynomials(self.p_neighbour, self.max_degree)
        if not self.sparse:
            self.cheb_e = tf.sparse.to_dense(self.cheb_e)
            self.cheb_p = tf.sparse.to_dense(self.cheb_p)

        """Logdr AutoEncoder"""
        self.ae_logdr = AutoEncoder(self.placeholder_logdr, self.gc_dim, self.fc_dim, self.cheb_e, self.sparse)

        """Logdr AutoEncoder"""
        self.ae_s = AutoEncoder(self.placeholder_s, self.gc_dim, self.fc_dim, self.cheb_p, self.sparse)

        """Output"""
        self.output_logdr = self.ae_logdr.output
        self.output_s = self.ae_s.output


class AutoEncoder:
    def __init__(self, input, gc_dim, fc_dim, cheb, sparse=False, lr=1e-3, output_dim=None):
        """Get Input and its dimension"""
        self.input = input
        input_dim = input.get_shape().as_list()
        self.input_dim = input_dim[-1]
        self.output_dim = output_dim
        self.point_num = input_dim[-2]
        self.batch_size = input_dim[0]
        self.gc_dim = gc_dim
        self.fc_dim = fc_dim
        """Get Other Members"""
        self.sparse = sparse
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.logging = True
        """Build the Network"""
        self.latent = None
        self.output = None
        self.cheb = cheb
        self.activation = []
        self.encoder_build()
        self.decoder_build()

    def encoder_build(self):
        with tf.variable_scope('Encoder'):
            """ Graph Convolutional Layers """
            self.activation.append(self.input)
            for i in range(len(self.gc_dim)):
                in_d = self.input_dim if i == 0 else self.gc_dim[i-1]
                out_d = self.gc_dim[i]
                gc_layer = GraphConvolution(input_dim=in_d,
                                            output_dim=out_d,
                                            name='GC_' + str(i),
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse=self.sparse,
                                            logging=self.logging)
                hidden = gc_layer(self.activation[-1], self.cheb)
                self.activation.append(hidden)
            last_dim = self.point_num * self.gc_dim[-1]
            hidden = tf.reshape(self.activation[-1], [-1, last_dim])
            self.activation.append(hidden)

            """ Fully Connected Layers """
            for i in range(len(self.fc_dim)):
                out_d = self.fc_dim[i]
                act_fun = None if i == len(self.fc_dim) - 1 else tf.nn.relu
                hidden = set_full(self.activation[-1], out_d, 'full_' + str(i), act_fun)
                self.activation.append(hidden)

            self.latent = self.activation[-1]

    def decoder_build(self):
        with tf.variable_scope('Decoder'):
            """ Fully Connected Layers """
            for i in list(range(len(self.fc_dim)-1))[::-1]:
                out_d = self.fc_dim[i]
                hidden = set_full(self.activation[-1], out_d, 'de_full_' + str(i), tf.nn.relu)
                self.activation.append(hidden)
            last_dim = self.point_num * self.gc_dim[-1]
            hidden = set_full(self.activation[-1], last_dim, 'de_full_' + str(0), tf.nn.relu)
            self.activation.append(hidden)

            """ Graph Convolutional Layers """
            hidden = tf.reshape(self.activation[-1], [-1, self.point_num, self.gc_dim[-1]])
            self.activation.append(hidden)
            for i in list(range(len(self.gc_dim)))[::-1]:
                in_d = self.gc_dim[i]
                out_d = (self.output_dim if self.output_dim else self.input_dim) if i == 0 else self.gc_dim[i-1]
                gc_layer = GraphConvolution(input_dim=in_d,
                                            output_dim=out_d,
                                            name='GC_' + str(i),
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging)
                hidden = gc_layer(self.activation[-1], self.cheb)
                self.activation.append(hidden)
            self.output = self.activation[-1]


def set_full(X, out_dim, scope=None, activate=tf.nn.relu, bn=False):
    with tf.variable_scope(scope or 'full', reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [X.get_shape().as_list()[1], out_dim], trainable=True, initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable('b', [out_dim], trainable=True, initializer=tf.random_normal_initializer)
        output = tf.add(tf.tensordot(X, W, [[1], [0]]), b)
        if bn:
            output = tf.contrib.layers.batch_norm(output, .9, epsilon=1e-5, activation_fn=None)
        if activate:
            output = activate(output)
        return output


def get_length(tensor):
    length = 1
    shape = tensor.shape.as_list()
    for i in range(1, len(np.shape(shape))):
        length *= shape[i]
    return length


def load_neighbour(neighbour, edges, is_padding=False):
    data = neighbour
    if is_padding:
        x = np.zeros((edges + 1, 4)).astype('int32')
        for i in range(0, edges):
            x[i + 1] = data[:, i] + 1

            for j in range(0, 4):
                if x[i + 1][j] == -1:
                    x[i + 1][j] = 0
    else:
        x = np.zeros((edges, 4)).astype('int32')
        for i in range(0, edges):
            x[i] = data[:, i]
    return x


def load_data(path, result_max=0.9, result_min=-0.9, logdr_ismap=False, s_ismap=False):
    data = h5py.File(path)
    logdr = data['logdr']
    s = data['s']
    e_neighbour = data['e_neighbour']
    p_neighbour = np.transpose(data['p_neighbour'])
    edgenum = len(logdr[0])
    pointnum = len(s[0])
    modelnum = len(logdr[0][0])
    e_nb = load_neighbour(e_neighbour, edgenum)

    maxdegree = p_neighbour.shape[1]
    p_nb = p_neighbour
    degree = np.zeros((p_neighbour.shape[0], 1)).astype('float32')
    for i in range(p_neighbour.shape[0]):
        degree[i] = np.count_nonzero(p_nb[i])

    logdr_x = logdr
    logdr_x = np.transpose(logdr_x, (2, 1, 0))

    s_x = s
    s_x = np.transpose(s_x, (2, 1, 0))
    if logdr_ismap:
        logdrmin = logdr_x.min() - 1e-6
        logdrmax = logdr_x.max() + 1e-6
        logdrnew = (result_max - result_min) * (logdr_x - logdrmin) / (logdrmax - logdrmin) + result_min

    else:
        logdrnew = logdr_x
        logdrmin = 0
        logdrmax = 0

    if s_ismap:
        smin = s_x.min() - 1e-6
        smax = s_x.max() + 1e-6
        snew = (result_max - result_min) * (s_x - smin) / (smax - smin) + result_min
    else:
        snew = s_x
        smin = 0
        smax = 0
    print(snew[0][0])

    return logdrnew, snew, e_nb, p_nb, degree, logdrmin, logdrmax, smin, smax, modelnum, pointnum, edgenum, maxdegree
