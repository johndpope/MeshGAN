from Model.Layers import *
from Model.utils import *
import tensorflow as tf
import numpy as np
import h5py
import os


class MeshAE:
    def __init__(self, matpath, gc_dim, fc_dim, max_degree=2, sparse=False, result_max=0.9, result_min=-0.9, ismap=False,
                 lambda_laplace=0.5, adj_or_geo=0, laplacian_k_range=None):
        if laplacian_k_range is None:
            laplacian_k_range = [10, 15]
        self.ismap = ismap
        self.result_max = result_max
        self.result_min = result_min

        self.vertex, self.acap, self.pointnum, self.p_adj, self.cot, self.p_neighbour, self.acapmax, self.acapmin = load_acap(matpath, result_min=result_min, result_max=result_max, ismap=self.ismap)

        self.gc_dim = gc_dim
        self.fc_dim = fc_dim
        self.max_degree = max_degree
        self.sparse = sparse
        self.lambda_laplace = lambda_laplace


        """Make Placholder for ACAP Label"""
        acap_shape = np.shape(self.acap)
        self.placeholder_acap = tf.placeholder(tf.float32, [None, acap_shape[1], acap_shape[2]])

        """Make Placeholder for Input"""
        self.input = tf.placeholder(tf.float32, [None, acap_shape[1], 3])
        self.output_gt = tf.placeholder(tf.float32, [None, acap_shape[1], acap_shape[2]])

        """Get Chebyshev Sequence"""
        self.cheb = chebyshev_polynomials(self.cot, self.max_degree)
        self.cheb_p = tf.sparse_placeholder(tf.float32)

        if not self.sparse:
            self.cheb = tuple_to_dense(self.cheb)
            self.cheb_p = tf.placeholder(tf.float32, shape=np.shape(self.cheb))

        """Mesh Encoder"""
        self.encoder = Encoder(self.input, self.gc_dim, self.fc_dim, self.cheb_p, self.max_degree, self.sparse, 'Encoder')
        self.latent = self.encoder.latent
        self.latent_placeholder = tf.placeholder(tf.float32, [None, self.fc_dim[-1]], 'latent_placeholder')

        """ACAP Decoder"""
        self.decoder = Decoder(self.latent, acap_shape[-1], self.pointnum, self.gc_dim, self.fc_dim, self.cheb_p, self.max_degree, self.sparse,
                               'Decoder')
        self.decoder_placeholder = Decoder(self.latent_placeholder, acap_shape[-1], self.pointnum, self.gc_dim, self.fc_dim, self.cheb_p, self.max_degree, self.sparse,
                                           'Decoder', reuse=True)


        """Output"""
        self.output = self.decoder.output
        self.output_p = self.decoder_placeholder.output
        self.recon_loss = tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(self.output_gt, self.output)), 0))

        """Laplacian Loss"""
        self.laplace_p = tf.placeholder(tf.float32, shape=[self.pointnum, self.pointnum])
        if adj_or_geo == 2:
            self.laplace = load_geodesic_weight()
            self.laplacian_norm = self.lambda_laplace * get_laplacian_norm_by_geo(self.laplace_p, self.encoder.fcpara,
                                                                                  self.decoder.fcpara, self.pointnum,
                                                                                  self.gc_dim[-1], self.fc_dim[0])
        elif adj_or_geo == 1:
            self.laplace = np.zeros([self.pointnum, self.pointnum])
            self.laplacian_norm = self.lambda_laplace * get_laplacian_norm_by_adj(self.p_adj, self.encoder.fcpara,
                                                                                  self.decoder.fcpara, self.pointnum,
                                                                                  self.gc_dim[-1], self.fc_dim[0],
                                                                                  laplacian_k_range)
        else:
            self.laplace = np.zeros([self.pointnum, self.pointnum])
            self.laplacian_norm = tf.zeros([], dtype=tf.float32)

        self.loss = self.recon_loss + self.laplacian_norm

        self.save_folder = './Param/'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self, epoch=2000, batchsize=10, lr=1e-3, continue_train=False):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if continue_train:
                latest = tf.train.latest_checkpoint(self.save_folder)
                self.saver.restore(sess, latest)

            total_step = []
            total_loss = []
            total_recon = []
            total_lapla = []
            temp_loss = []
            temp_recon = []
            temp_lapla = []
            name = 0
            for i in range(epoch):
                batch_index = np.random.randint(0, np.shape(self.acap)[0], [batchsize], np.int)
                input_data = self.vertex[batch_index]
                output_gt = self.acap[batch_index]
                output = sess.run(self.output, feed_dict={self.input: input_data, self.cheb_p: self.cheb})
                loss, loss_recon, loss_lapla, _ = sess.run([self.loss, self.recon_loss, self.laplacian_norm, optimizer],
                                                           feed_dict={self.input: input_data, self.cheb_p: self.cheb,
                                                                      self.output_gt: output_gt,
                                                                      self.laplace_p: self.laplace})
                loss_out = np.sum(np.average(np.square(output - output_gt), 0))

                print('Epoch: %03d/%03d| Loss: %05f| Recon: %05f| Lapla: %05f| Loss_out: %05f' % (i, epoch, loss, loss_recon, loss_lapla, float(loss_out)), ' ', batch_index)
                temp_loss.append(loss)
                temp_lapla.append(loss_lapla)
                temp_recon.append(loss_recon)
                if i > 0 and i % 20 == 0:
                    total_step.append(i)
                    total_loss.append(np.average(temp_loss))
                    total_recon.append(np.average(temp_recon))
                    total_lapla.append(np.average(temp_lapla))
                    temp_loss = []
                    temp_recon = []
                    temp_lapla = []
                    plot_multi_info(total_loss, total_recon, total_lapla, total_step, str(name))

                    if total_loss[0] > total_loss[-1] * 10 or int(i / 1000) > int((i - 1) / 1000):
                        name += 1
                        total_loss = []
                        total_step = []
                        total_recon = []
                        total_lapla = []
                    self.saver.save(sess, self.save_folder, i)

                    mid_path = './mid_result'
                    if not os.path.exists(mid_path):
                        os.makedirs(mid_path)
                    name_h5 = mid_path + '/result' + str(i) + '.h5'
                    print(name_h5)
                    f = h5py.File(name_h5, 'w')
                    gt_ = np.squeeze(output_gt)
                    output_ = np.squeeze(output)
                    f['test_mesh'] = output_
                    f['gt_mesh'] = gt_
                    f.close()

    def use(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            latest = tf.train.latest_checkpoint(self.save_folder)
            self.saver.restore(sess, latest)

            total_output = []
            total_gt = []
            batchsize = 10
            begin = 0
            end = begin + batchsize

            while begin < len(self.vertex):
                end = len(self.vertex) if end > len(self.vertex) - batchsize else begin + batchsize
                batch_index = np.array(range(begin, end))
                input_data = self.vertex[batch_index]
                output_gt = self.acap[batch_index]
                output = sess.run(self.output, feed_dict={self.input: input_data, self.cheb_p: self.cheb})
                loss_out = np.sum(np.average(np.square(output - output_gt), 0))
                print('Loss_out: %05f' % (float(loss_out)), ' ', batch_index)
                total_output = output if begin == 0 else np.concatenate([total_output, output], 0)
                total_gt = output_gt if begin == 0 else np.concatenate([total_gt, output_gt], 0)
                begin += batchsize

            name_h5 = './result.h5'
            print(name_h5)
            f = h5py.File(name_h5, 'w')
            gt_ = np.squeeze(total_gt)
            output_ = np.squeeze(total_output)
            f['test_mesh'] = output_
            f['gt_mesh'] = gt_
            f.close()

    def interpolation(self, i, j, num):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            latest = tf.train.latest_checkpoint(self.save_folder)
            self.saver.restore(sess, latest)

            batch_index = np.array([i, j])
            input_data = self.vertex[batch_index]
            latent = sess.run(self.latent, feed_dict={self.input: input_data, self.cheb_p: self.cheb})
            delta = (latent[1] - latent[0]) / num
            total_latent = np.zeros(np.concatenate([[num], np.shape(latent[0])]))
            total_latent[0] = latent[0]
            for ii in range(1, num):
                total_latent[ii] = total_latent[ii-1] + delta
            output = sess.run(self.output_p, feed_dict={self.latent_placeholder: total_latent, self.cheb_p: self.cheb})
            name_h5 = './interpolation/inter_' + str(i) + '_' + str(j) + '.h5'
            print(name_h5)
            f = h5py.File(name_h5, 'w')
            output = np.squeeze(output)
            f['test_mesh'] = output
            f.close()


class Encoder:
    def __init__(self, input, gc_dim, fc_dim, cheb, max_degree, sparse=False, name='AE', lr=1e-3):
        """Get Input and its dimension"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.name = name
            self.input = input
            input_dim = input.get_shape().as_list()
            self.input_dim = input_dim[-1]
            self.point_num = input_dim[-2]
            self.batch_size = input_dim[0]
            self.gc_dim = gc_dim
            self.fc_dim = fc_dim
            self.max_degree = max_degree
            """Get Other Members"""
            self.sparse = sparse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.logging = True
            """Build the Network"""
            self.latent = None
            self.output = None
            self.cheb = cheb
            self.activation = []
            self.fcpara = None
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
                                            max_degree=self.max_degree,
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
                model = self if i == 0 else None
                hidden = set_full(self.activation[-1], out_d, 'full_' + str(i), act_fun, model=model)
                self.activation.append(hidden)

            self.latent = self.activation[-1]

    def print_layers(self):
        print(self.name)
        for i in range(len(self.activation)):
            print(self.activation[i])


class Decoder:
    def __init__(self, input, output_dim, point_num, gc_dim, fc_dim, cheb, max_degree, sparse=False, name='AE', lr=1e-3, reuse=False):
        """Get Input and its dimension"""
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
                print('reuse')
            self.name = name
            self.input = input
            self.output_dim = output_dim
            self.point_num = point_num
            self.gc_dim = gc_dim
            self.fc_dim = fc_dim
            self.max_degree = max_degree
            """Get Other Members"""
            self.sparse = sparse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.logging = True
            """Build the Network"""
            self.latent = None
            self.output = None
            self.cheb = cheb
            self.activation = []
            self.fcpara = None
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
            hidden = set_full(self.activation[-1], last_dim, 'de_full_' + str(0), tf.nn.relu, model=self)
            self.activation.append(hidden)

            """ Graph Convolutional Layers """
            hidden = tf.reshape(self.activation[-1], [-1, self.point_num, self.gc_dim[-1]])
            self.activation.append(hidden)
            for i in list(range(len(self.gc_dim)))[::-1]:
                in_d = self.gc_dim[i]
                out_d = self.output_dim if i == 0 else self.gc_dim[i-1]
                act_fun = None if i == 0 else tf.nn.relu
                bn = False if i == 0 else True
                gc_layer = GraphConvolution(input_dim=in_d,
                                            output_dim=out_d,
                                            name='GC_' + str(i),
                                            max_degree=self.max_degree,
                                            act=act_fun,
                                            bn=bn,
                                            dropout=0.,
                                            logging=self.logging)
                hidden = gc_layer(self.activation[-1], self.cheb)
                self.activation.append(hidden)
            self.output = self.activation[-1]

    def print_layers(self):
        print(self.name)
        for i in range(len(self.activation)):
            print(self.activation[i])


def get_laplacian_norm_by_adj(p_adj, fcpara_e, fcpara_d, pointnum, channel, fdim, k_range=None, decay_rate=.5):
    """Laplacian Loss up to degree k with specified decay rate"""
    if k_range is None:
        k_range = [10, 15]
    fcpara_group_e = tf.transpose(tf.reshape(fcpara_e, [pointnum, channel, fdim]),
                                  perm=[2, 0, 1])
    fcpara_group_d = tf.reshape(fcpara_d, [fdim, pointnum, channel])

    selfdot_e = tf.reduce_sum(tf.pow(fcpara_group_e, 2.0), axis=2)
    selfdot_d = tf.reduce_sum(tf.pow(fcpara_group_d, 2.0), axis=2)
    selfdot = tf.add(selfdot_d, selfdot_e)
    maxdimension = tf.argmax(selfdot, axis=1)

    def get_adj_sum(p_adj, k_range, decay_rate):
        """Get the decaying laplacian matrix up to degree k"""
        output = p_adj
        rate = 1.
        adj = np.ones_like(p_adj)
        for i in range(1, k_range[1]+1):
            adj = tf.cast(tf.cast(tf.matmul(adj, p_adj), tf.bool), tf.float32)
            if i >= k_range[0]:
                rate *= decay_rate
                adj_com = np.ones_like(adj) - adj
                output += adj_com * rate
        return output
    adj = get_adj_sum(np.array(p_adj, dtype=np.float32), k_range, decay_rate)
    maxlaplacian = tf.cast(tf.gather(adj, maxdimension), tf.float32)
    laplacian_norm_e = tf.reduce_mean(tf.reduce_sum(tf.sqrt(selfdot_e) * maxlaplacian, 1))
    laplacian_norm_d = tf.reduce_mean(tf.reduce_sum(tf.sqrt(selfdot_d) * maxlaplacian, 1))
    laplacian_norm = laplacian_norm_d + laplacian_norm_e
    return laplacian_norm


def get_laplacian_norm_by_geo(laplacian, fcpara_e, fcpara_d, pointnum, channel, fdim):
    """Laplacian Loss up to degree k with specified decay rate"""
    fcpara_group_e = tf.transpose(tf.reshape(fcpara_e, [pointnum, channel, fdim]),
                                  perm=[2, 0, 1])
    fcpara_group_d = tf.reshape(fcpara_d, [fdim, pointnum, channel])

    selfdot_e = tf.reduce_sum(tf.pow(fcpara_group_e, 2.0), axis=2)
    selfdot_d = tf.reduce_sum(tf.pow(fcpara_group_d, 2.0), axis=2)
    selfdot = tf.add(selfdot_d, selfdot_e)
    maxdimension = tf.argmax(selfdot, axis=1)

    maxlaplacian = tf.cast(tf.gather(laplacian, maxdimension), tf.float32)
    laplacian_norm_e = tf.reduce_mean(tf.reduce_sum(tf.sqrt(selfdot_e) * maxlaplacian, 1))
    laplacian_norm_d = tf.reduce_mean(tf.reduce_sum(tf.sqrt(selfdot_d) * maxlaplacian, 1))
    laplacian_norm = laplacian_norm_d + laplacian_norm_e
    return laplacian_norm


def set_full(X, out_dim, scope=None, activate=tf.nn.relu, bn=True, model=None):
    with tf.variable_scope(scope or 'full', reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [X.get_shape().as_list()[1], out_dim], trainable=True, initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable('b', [out_dim], trainable=True, initializer=tf.random_normal_initializer)
        output = tf.add(tf.tensordot(X, W, [[1], [0]]), b)
        if bn:
            output = tf.contrib.layers.batch_norm(output, .9, epsilon=1e-5, activation_fn=None)
        if activate:
            output = activate(output)
        if model is not None:
            model.fcpara = W
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


def load_acap(path, result_max=0.9, result_min=-0.9, ismap=False):
    data = h5py.File(path)
    acap = data['acap']
    p_adj = np.transpose(data['p_adj'])
    cot = np.transpose(data['cot'])
    vertex = np.transpose(data['vertex'])
    p_neighbour = np.transpose(data['p_neighbour'])
    pointnum = len(acap[0])

    acap_x = acap
    acap_x = np.transpose(acap_x, (2, 1, 0))

    acapmin = acap_x.min() - 1e-6
    acapmax = acap_x.max() + 1e-6

    if ismap:
        acapnew = (result_max - result_min) * (acap_x - acapmin) / (acapmax - acapmin) + result_min

    else:
        acapnew = acap_x

    return vertex, acapnew, pointnum, p_adj, cot, p_neighbour, acapmax, acapmin


def load_data(path, result_max=0.9, result_min=-0.9, logdr_ismap=False, s_ismap=False):
    data = h5py.File(path, "r")
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
