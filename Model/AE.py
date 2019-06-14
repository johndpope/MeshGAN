from Model.Layers import *
from Model.utils import *
import tensorflow as tf
import numpy as np
import h5py


class MeshAE:
    def __init__(self, matpath, gc_dim, fc_dim, max_degree=2, sparse=False, result_max=0.9, result_min=-0.9):
        self.logdr, self.s, self.e_neighbour, self.p_neighbour,\
            self.degree, self.logdr_min, self.logdr_max, self.ds_min,\
            self.ds_max, self.modelnum, self.pointnum, self.edgenum, self._old_maxdegree\
            = load_data(matpath, result_min=result_min, result_max=result_max, logdr_ismap=True, s_ismap=False)
        self.gc_dim = gc_dim
        self.fc_dim = fc_dim
        self.max_degree = max_degree
        self.sparse = sparse

        """Get Adjacency Matrix"""
        data = h5py.File(matpath)
        self.p_adj = np.transpose(data['p_adj'])
        self.e_adj = np.transpose(data['e_adj'])

        """Make Placholder for RIMD Label"""
        logdr_shape = np.shape(self.logdr)
        self.placeholder_logdr = tf.placeholder(tf.float32, [None, logdr_shape[1], logdr_shape[2]])

        s_shape = np.shape(self.s)
        self.placeholder_s = tf.placeholder(tf.float32, [None, s_shape[1], s_shape[2]])

        """Make Placeholder for Input"""
        self.input = tf.placeholder(tf.float32, [None, logdr_shape[1], 3])

        """Get Chebyshev Sequence"""
        self.cheb_e = chebyshev_polynomials(self.e_adj, self.max_degree)
        self.cheb_p = chebyshev_polynomials(self.p_adj, self.max_degree)

        if not self.sparse:
            self.cheb_e = tuple_to_dense(self.cheb_e)
            self.cheb_p = tuple_to_dense(self.cheb_p)

        """Mesh Encoder"""
        self.encoder = Encoder(self.input, self.gc_dim, self.fc_dim, self.cheb_e, self.sparse, 'Encoder')
        self.latent = self.encoder.latent

        """Logdr AutoEncoder"""
        self.decoder_logdr = Decoder(self.latent, 9, self.edgenum, self.gc_dim, self.fc_dim, self.cheb_e, self.sparse, 'Decoder_logdr')

        """S AutoEncoder"""
        self.decoder_s = Decoder(self.latent, 9, self.pointnum, self.gc_dim, self.fc_dim, self.cheb_p, self.sparse, 'Decoder_s')

        """Output"""
        self.output_logdr = self.decoder_logdr.output
        self.output_s = self.decoder_s.output


class Encoder:
    def __init__(self, input, gc_dim, fc_dim, cheb, sparse=False, name='AE', lr=1e-3):
        """Get Input and its dimension"""
        with tf.variable_scope(name):
            self.name = name
            self.input = input
            input_dim = input.get_shape().as_list()
            self.input_dim = input_dim[-1]
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
            self._build()
            self.print_layers()

    def _build(self):
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
                                            dropout=0.,
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

    def print_layers(self):
        print(self.name)
        for i in range(len(self.activation)):
            print(self.activation[i])


class Decoder:
    def __init__(self, input, output_dim, point_num, gc_dim, fc_dim, cheb, sparse=False, name='AE', lr=1e-3):
        """Get Input and its dimension"""
        with tf.variable_scope(name):
            self.name = name
            self.input = input
            self.output_dim = output_dim
            self.point_num = point_num
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
            self._build()
            self.print_layers()

    def _build(self):
        with tf.variable_scope('Decoder'):
            """ Fully Connected Layers """
            self.activation.append(self.input)
            for i in list(range(1, len(self.fc_dim)))[::-1]:
                out_d = self.fc_dim[i-1]
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
                out_d = self.output_dim if i == 0 else self.gc_dim[i-1]
                gc_layer = GraphConvolution(input_dim=in_d,
                                            output_dim=out_d,
                                            name='GC_' + str(i),
                                            act=tf.nn.relu,
                                            dropout=0.,
                                            logging=self.logging)
                hidden = gc_layer(self.activation[-1], self.cheb)
                self.activation.append(hidden)
            self.output = self.activation[-1]

    def print_layers(self):
        print(self.name)
        for i in range(len(self.activation)):
            print(self.activation[i])


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


def recover_data(x, x_min, x_max, result_min=-0.9, result_max=0.9):
    x = np.array(x)

    x = (x_max - x_min) * (x - result_min) / (result_max - result_min) + x_min

    return x
