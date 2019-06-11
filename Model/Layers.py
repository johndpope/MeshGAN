from Model.inits import *


class GraphConvolution:
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, name='GC_Layer',
                 max_degree=2, dropout=0., act=tf.nn.relu,
                 bias=False, logging=True, sparse=False):
        self.name = name
        self.max_degree = max_degree
        self.act = act
        self.bias = bias
        self.logging = logging
        self.dropout = dropout
        self.sparse = sparse
        self.vars = {}

        with tf.variable_scope(self.name + '_vars'):
            for i in range(max_degree):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, cheb):
        x = inputs

        # dropout
        if self.dropout > 1e-4:
            if self.sparse:
                x = sparse_dropout(x, 1-self.dropout)
            else:
                x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(cheb)):
            pre_sup = tf.tensordot(x, self.vars['weights_' + str(i)], [[2], [0]])
            support = tf.tensordot(cheb[i], pre_sup, [[1], [1]])
            support = tf.transpose(support, [1, 0, 2])
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def __call__(self, inputs, cheb):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, cheb)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


def sparse_dropout(x, keep_prob):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    noise_shape = len(x.indices.get_shape().as_list())
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.tensordot(x, y, axes=[[2], [0]])
    return res
