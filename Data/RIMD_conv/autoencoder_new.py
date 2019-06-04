import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
from six.moves import xrange
import scipy.interpolate as interpolate
import h5py, time

epoch_num = 100000
latent_zdim = 64
timecurrent = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
logfolder = './'+timecurrent+'new_100'
if not os.path.isdir(logfolder):
    os.mkdir(logfolder)
logdr_ismap=True
s_ismap=False


def leaky_relu(input, alpha=0.02):
    return tf.maximum(input, tf.minimum(alpha * input, 0))
def leaky_relu2(input, alpha=0.02):
    return tf.maximum(tf.minimum(input,1), tf.maximum(alpha * input, -1))

def batch_norm_wrapper(inputs, name='batch_norm', is_training=False, decay=0.9, epsilon=1e-5):
    with tf.variable_scope(name) as scope:
        if is_training == True:
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True,
                                    initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float32))
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True,
                                   initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float32))
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False,
                                       initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float32))
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False,
                                      initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float32))
        else:
            scope.reuse_variables()
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

        if is_training:
            axis = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            # batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, epsilon)


# In[5]:

def load_data(path, result_max=0.9, result_min=-0.9,logdr_ismap=False,s_ismap=False):
    data = h5py.File(path)
    # feature = data['feature']
    logdr = data['logdr']
    s = data['s']
    e_neighbour = data['e_neighbour']
    p_neighbour =np.transpose(data['p_neighbour'])
    edgenum = len(logdr[0])
    pointnum = len(s[0])
    modelnum = len(logdr[0][0])
    e_nb = load_neighbour(e_neighbour, edgenum)




    maxdegree = p_neighbour.shape[1]
    p_nb = np.zeros((pointnum, maxdegree)).astype('int32')
    p_nb = p_neighbour
    degree = np.zeros((p_neighbour.shape[0], 1)).astype('float32')
    for i in range(p_neighbour.shape[0]):
        degree[i] = np.count_nonzero(p_nb[i])

    logdr_x = np.zeros((modelnum, edgenum, 3)).astype('float32')
    logdr_x = logdr
    logdr_x = np.transpose(logdr_x, (2, 1, 0))

    s_x = np.zeros((modelnum, pointnum, 6)).astype('float32')
    s_x = s
    s_x = np.transpose(s_x, (2, 1, 0))
    print(s_x[0][0])
    if logdr_ismap:
        logdrmin = logdr_x.min() - 1e-6
        logdrmax = logdr_x.max() + 1e-6


    #print(smin)
    #print(smax)
        logdrnew = (result_max - result_min) * (logdr_x - logdrmin) / (logdrmax - logdrmin) + result_min

    else:
        logdrnew=logdr_x
        
        logdrmin=0
        logdrmax=0

    if s_ismap:
        smin = s_x.min() - 1e-6
        smax = s_x.max() + 1e-6
        snew = (result_max - result_min) * (s_x - smin) / (smax - smin) + result_min
    else:
        snew=s_x
        smin=0
        smax=0
    print(snew[0][0])
    #x = np.concatenate((logdrnew, dsnew), axis=2)

    # x_min = x.min()
    # x_min = x_min - 1e-6
    # x_max = x.max()
    # x_max = x_max + 1e-6

    # x = (result_max-result_min)*(x-x_min)/(x_max - x_min) + result_min

    return logdrnew,snew, e_nb,p_nb, degree,logdrmin, logdrmax, smin, smax, modelnum,pointnum, edgenum,maxdegree


def load_neighbour(neighbour, edges, is_padding=False):
    data = neighbour

    if is_padding == True:
        x = np.zeros((edges + 1, 4)).astype('int32')

        for i in xrange(0, edges):
            x[i + 1] = data[:, i] + 1

            for j in xrange(0, 4):
                if x[i + 1][j] == -1:
                    x[i + 1][j] = 0

    else:
        x = np.zeros((edges, 4)).astype('int32')

        for i in xrange(0, edges):
            x[i] = data[:, i]

    return x


# neighbour = load_neighbour(neighbourfile, 'neighbour')

def recover_data(x, x_min, x_max, result_min=-0.9, result_max=0.9):
    x = np.array(x)

    x = (x_max - x_min) * (x - result_min) / (result_max - result_min) + x_min

    return x


def recover_data1(recover_feature, logrmin, logrmax, smin, smax):
    logdr = recover_feature[:, :, 0:3]
    ds = recover_feature[:, :, 3:9]

    resultmax = 0.9
    resultmin = -0.9

    ds = (smax - smin) * (ds - resultmin) / (resultmax - resultmin) + smin
    logdr = (logrmax - logrmin) * (logdr - resultmin) / (resultmax - resultmin) + logrmin
    x = np.concatenate((logdr, ds), axis=2)

    return x
def recover_data_new(logdr,s,logrmin, logrmax, smin, smax,logdr_ismap=False,s_ismap=False):
    resultmax=0.9
    resultmin=-0.9
    if logdr_ismap:
        logdr_new=(logrmax - logrmin) * (logdr - resultmin) / (resultmax - resultmin) + logrmin
    else:
        logdr_new=logdr
    if s_ismap:
        s_new = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
    else:
        s_new=s
    return logdr_new,s_new

# In[8]:

class meshVAE():
    def __init__(self, matpath, padding=False, result_max1=0.9, result_min1=-0.9):

        self.logdr,self.s, self.e_neighbour,self.p_neighbour,self.degree,self.logdr_min, self.logdr_max, self.ds_min, self.ds_max, self.modelnum, self.pointnum,self.edgenum,self.maxdegree = load_data(
            matpath,logdr_ismap=logdr_ismap,s_ismap=s_ismap)
        self.edges = self.edgenum

        if padding == True:
            self.edges += 1
            self.is_padding = True
        else:
            self.is_padding = False

        self.inputs_logdr = tf.placeholder(tf.float32, [None, self.edgenum, 3], name='input_logdr')
        self.inputs_s = tf.placeholder(tf.float32, [None, self.pointnum, 6], name='input_s')
        #self.random = tf.placeholder(tf.float32, [None, latent_zdim], name='random_samples')
        #self.input_test=tf.placeholder(tf.float32, [None, self.edgenum, 9], name='test_mesh')
        self.g_z_logdr=tf.placeholder(tf.float32,[None,latent_zdim],name='g_z_logdr')
        self.g_z_s=tf.placeholder(tf.float32,[None,latent_zdim],name='g_z_s')


        self.e_nb = tf.constant(self.e_neighbour, dtype='int32', shape=[self.edges, 4], name='e_nb_relation')
        self.p_nb = tf.constant(self.p_neighbour, dtype='int32', shape=[self.pointnum, self.maxdegree], name='p_nb_relation')
        self.degrees = tf.constant(self.degree, dtype='float32', shape=[self.pointnum, 1], name='degrees')




        self.logdr_n1, self.logdr_e1 = self.get_conv_weights(3, 3, name = 'logdr_convw1')
        #self.n2, self.e2 = self.get_conv_weights(9, 9, name = 'convw2')
        self.logdr_n3, self.logdr_e3 = self.get_conv_weights(3, 3, name = 'logdr_convw3')

        self.s_n1, self.s_e1 = self.get_conv_weights(6, 6, name = 's_convw1')
        self.s_n3, self.s_e3 = self.get_conv_weights(6, 6, name = 's_convw3')
		
        self.z_logdr=self.encoder_logdr(self.inputs_logdr,training=True)
        self.z_s=self.encoder_s(self.inputs_s,self.degrees,training=True)

        self.generated_mesh_train_logdr = self.decoder_logdr(self.z_logdr, training=True)
        self.generated_mesh_train_s = self.decoder_s(self.z_s,self.degrees, training=True)

		
        self.z_logdr_test=self.encoder_logdr(self.inputs_logdr,training=False)
        self.z_s_test=self.encoder_s(self.inputs_s,self.degrees,training=False)
        self.test_mesh_logdr = self.decoder_logdr(self.z_logdr_test, training=False)
        self.test_mesh_s = self.decoder_s(self.z_s_test,self.degrees, training=False)



        self.g_mesh_logdr=self.decoder_logdr(self.g_z_logdr,training=False)
        self.g_mesh_s=self.decoder_s(self.g_z_s,self.degree,training=False)

        self.loss_logdr = 1 * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs_logdr - self.generated_mesh_train_logdr,2), [1, 2]))
        self.loss_s=1 * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs_s - self.generated_mesh_train_s,2), [1, 2]))

        self.cost = self.loss_logdr+self.loss_s  # + self.r2

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.saver = tf.train.Saver()


    def get_conv_weights(self, input_dim, output_dim, name='convweight'):
        with tf.variable_scope(name) as scope:
            n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=0.02))
            e = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=0.02))

            return n, e

    def linear(self, input_, input_size, output_size, name='Linear', stddev=0.02, bias_start=0.0):
        with tf.variable_scope(name) as scope:
            matrix = tf.get_variable("weights", [input_size, output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size], tf.float32,
                                   initializer=tf.constant_initializer(bias_start))

            return tf.matmul(input_, matrix) + bias  # , matrix

    def fclayer(self, inputs, input_dim, output_dim, name='fclayer', training=True):
        with tf.variable_scope(name) as scope:
            fcdot = self.linear(inputs, input_dim, output_dim, name='fclinear')
            fcbn = batch_norm_wrapper(fcdot, name='fcbn', is_training=training, decay=0.9)
            fca = leaky_relu(fcbn)

            return fca

    def convlayer(self, input_feature, input_dim, output_dim, name='meshconv', training=True, special_activation=False,
                  no_activation=False, padding=False, bn=True):
        with tf.variable_scope(name) as scope:

            def compute_nb_feature(input_feature):
                return tf.gather(input_feature, self.nb)

            total_nb_feature = tf.map_fn(compute_nb_feature, input_feature)
            total_nb_feature = tf.reshape(total_nb_feature, [tf.shape(total_nb_feature)[0], self.edges, input_dim * 4])

            nb_weights = tf.get_variable("nb_weights", [input_dim * 4, output_dim], tf.float32,
                                         tf.random_normal_initializer(stddev=0.02))
            nb_bias = tf.get_variable("nb_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            nb_feature = tf.tensordot(total_nb_feature, nb_weights, [[2], [0]]) + nb_bias

            edge_weights = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32,
                                           tf.random_normal_initializer(stddev=0.02))
            edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2], [0]]) + edge_bias

            total_feature = edge_feature + nb_feature

            if bn == False:
                fb = total_feature
            else:
                fb = batch_norm_wrapper(total_feature, is_training=training)
            # fb = self.batch_norm_wrapper(total_feature, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = leaky_relu(fb)
            else:
                fa = leaky_relu2(fb)

            if padding == True:
                padding_feature = tf.zeros([tf.shape(fa)[0], 1, output_dim], tf.float32)

                _, true_feature = tf.split(fa, [1, self.edges - 1], 1)

                fa = tf.concat([padding_feature, true_feature], 1)

        return fa

    def newconvlayer(self, input_feature,nb, input_dim, output_dim, nb_weights, edge_weights, name='meshconv',on_edge=True,degrees=4.0,training=True, special_activation=False, no_activation=False, bn=True, padding=False):
        with tf.variable_scope(name) as scope:
            if not on_edge:
                padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)
                padded_input = tf.concat([padding_feature, input_feature], 1)
            else:
                padded_input=input_feature
            def compute_nb_feature(input_feature):
                return tf.gather(input_feature,nb)

            total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
            total_nb_feature = tf.reduce_sum(total_nb_feature, axis=2) / degrees

            nb_bias = tf.get_variable("nb_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            nb_feature = tf.tensordot(total_nb_feature, nb_weights, [[2], [0]]) + nb_bias


            edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2], [0]]) + edge_bias

            total_feature = edge_feature + nb_feature

            if bn == False:
                fb = total_feature
            else:
                fb = batch_norm_wrapper(total_feature, is_training=training)
            # fb = self.batch_norm_wrapper(total_feature, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = leaky_relu(fb)
            else:
                fa=leaky_relu2(fb)

            if padding == True:
                padding_feature = tf.zeros([tf.shape(fa)[0], 1, output_dim], tf.float32)

                _, true_feature = tf.split(fa, [1, self.edges - 1], 1)

                fa = tf.concat([padding_feature, true_feature], 1)

            return fa

    def encoder_logdr(self, input_feature, training=True, padding=False):
        with tf.variable_scope("encoder_logdr") as scope:
            if (training == False):
                scope.reuse_variables()

            conv1 = self.newconvlayer(input_feature, self.e_nb,3, 3, self.logdr_n1,self.logdr_e1,name='logdr_conv1', training=training, padding=padding)
            #conv2 = self.newconvlayer(conv1, 9, 9, self.n2,self.e2,name='conv2', training=training, padding=padding)
            conv3 = self.newconvlayer(conv1,self.e_nb, 3, 3, self.logdr_n3,self.logdr_e3,name='logdr_conv3', training=training, padding=padding,special_activation = False, bn=False)

            x0 = tf.reshape(conv3, [tf.shape(conv3)[0], self.edges * 3])

            # x1 = tf.matmul(x0, self.fcparams)

            mean = self.linear(x0, self.edges * 3, latent_zdim, 'mean')
            mean = leaky_relu(mean)

        return mean
    def encoder_s(self, input_feature,degree, training=True, padding=False):
        with tf.variable_scope("encoder_s") as scope:
            if (training == False):
                scope.reuse_variables()

            conv1 = self.newconvlayer(input_feature, self.p_nb,6, 6, self.s_n1,self.s_e1,name='s_conv1',on_edge=False,degrees=degree, training=training, padding=padding)
            #conv2 = self.newconvlayer(conv1, 9, 9, self.n2,self.e2,name='conv2', training=training, padding=padding)
            conv3 = self.newconvlayer(conv1,self.p_nb, 6, 6, self.s_n3,self.s_e3,name='s_conv3',on_edge=False,degrees=degree,training=training, padding=padding,special_activation = False, bn=False)


            x0 = tf.reshape(conv3, [tf.shape(conv3)[0], self.pointnum * 6])

            # x1 = tf.matmul(x0, self.fcpara100ms)

            mean = self.linear(x0, self.pointnum * 6, latent_zdim, 'mean')
            mean = leaky_relu(mean)

        return mean

    def decoder_logdr(self, z, training=True, padding=False):
        with tf.variable_scope("decoder") as scope:
            if (training == False):
                scope.reuse_variables()

            h1 = self.linear(z, latent_zdim, self.edges * 3, 'logdr_h1')
            #h1 = tf.nn.tanh(h1)
            h1=leaky_relu(h1)
            #h1 = leaky_relu(batch_norm_wrapper(h1, name='h1bn', is_training=training))

            x0 = tf.reshape(h1, [tf.shape(h1)[0], self.edges, 3])

            conv1 = self.newconvlayer(x0,self.e_nb, 3, 3, tf.transpose(self.logdr_n3),tf.transpose(self.logdr_e3),name='logdr_conv1', training=training, padding=padding)
            #conv2 = self.newconvlayer(conv1, 9, 9, tf.transpose(self.n2),tf.transpose(self.e2),name='conv2', training=training, padding=padding)
            conv3 = self.newconvlayer(conv1,self.e_nb, 3, 3, tf.transpose(self.logdr_n1),tf.transpose(self.logdr_e1),name='logdr_conv3', training=training, padding=padding, special_activation = False, bn=False)


            output = conv3

            output = tf.nn.tanh(output)

        return output
    def decoder_s(self, z,degree,training=True, padding=False):
        with tf.variable_scope("decoder") as scope:
            if (training == False):
                scope.reuse_variables()

            h1 = self.linear(z, latent_zdim, self.pointnum * 6, 'h1')
            #h1 = tf.nn.tanh(h1)
            h1=leaky_relu(h1)
            #h1 = leaky_relu(batch_norm_wrapper(h1, name='h1bn', is_training=training))

            x0 = tf.reshape(h1, [tf.shape(h1)[0], self.pointnum, 6])

            conv1 = self.newconvlayer(x0,self.p_nb, 6, 6, tf.transpose(self.s_n3),tf.transpose(self.s_e3),name='conv1',on_edge=False,degrees=degree, training=training, padding=padding)
            #conv2 = self.newconvlayer(conv1, 9, 9, tf.transpose(self.n2),tf.transpose(self.e2),name='conv2', training=training, padding=padding)
            conv3 = self.newconvlayer(conv1,self.p_nb, 6, 6, tf.transpose(self.s_n1),tf.transpose(self.s_e1),name='conv3',on_edge=False,degrees=degree, training=training, padding=padding, special_activation = False, bn=False)


            output = conv3

            #output = tf.nn.tanh(output)

        return output
    def train(self,restore=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            if not restore is None:
            #print ('xxx')
                self.saver.restore(sess, restore)
            file = open(logfolder+'/' + '_script_result.txt', 'w')
            
            # self.saver.save(sess, 'meshgan.model', global_step = 0)

            for epoch in xrange(0, epoch_num):
                rand_index = np.random.choice(len(self.logdr),size=32)
                timecurrent1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))



                sess.run([self.optimizer], feed_dict={self.inputs_logdr: self.logdr[rand_index],self.inputs_s:self.s[rand_index]})
                d_logdr=self.generated_mesh_train_logdr.eval({self.inputs_logdr: self.logdr[rand_index]})
                d_s=self.generated_mesh_train_s.eval({self.inputs_s: self.s[rand_index]})
                loss_logdr = self.loss_logdr.eval({self.inputs_logdr: self.logdr[rand_index]})
                #print(self.s[0][0])
                #print(d_s[0][0])
                loss_s = self.loss_s.eval({self.inputs_s:self.s[rand_index]})
                #cost_latent = self.latent_loss.eval({self.inputs: self.data})
                # cost_r2 = self.latent_loss.eval()
                print("%s Epoch: [%4d]loss_logdr: %.8f loss_s:%.8f" % (timecurrent1, epoch + 1, loss_logdr,loss_s))
                file.write("%s Epoch: [%4d]loss_logdr: %.8f loss_s:%.8f\n" % (timecurrent1, epoch + 1, loss_logdr,loss_s))
                if np.mod(epoch+1, 1000) == 0 and epoch != 0:
                    self.saver.save(sess, logfolder +'/'+'meshvae.model', global_step = epoch+1)

            # self.saver.save(sess, timecurrent + 'meshvae.model', global_step=epoch_num)

        return

    def randomresult(self, restore, times, scale, foldername):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)

            random_batch = np.random.normal(loc=0.0, scale=scale, size=(50, latent_zdim))
            test = sess.run([self.test_mesh], feed_dict={self.random: random_batch})[0]
            test = recover_data1(test, self.logdr_min, self.logdr_max, self.ds_min, self.ds_max)

            # test = recover_data(test, self.data_min, self.data_max)

            name = foldername + '/randomtest2' + str(times) + '_' + str(scale) + '.h5'
            print name
            # sio.savemat(name, {'test_mesh': test, 'latent_z':random_batch})
            f = h5py.File(name, 'w')
            f['test_mesh'] = test
            f['latent_z'] = random_batch
            f.close()
        return

    def save_z(self,path,restore,foldername):
        print 'load'
        self.logdr,self.s,self.e_neighbour,self.p_neighbour,self.degree,self.logdr_min,self.logdr_max,self.ds_min,self.ds_max,self.modelnum,self.pointnum,self.edgenum,self.maxdegree=load_data(path,logdr_ismap=logdr_ismap,s_ismap=s_ismap)
        print len(self.logdr)
 
        with tf.Session() as sess:
            self.saver.restore(sess, restore)
            index=list(xrange(0,len(self.logdr)-1,32))
            for i in xrange(len(index)-1):
                print i
                test_logdr = sess.run([self.test_mesh_logdr], feed_dict={self.inputs_logdr: self.logdr[index[i]:index[i+1]]})[0]
                test_s = sess.run([self.test_mesh_s], feed_dict={self.inputs_s: self.s[index[i]:index[i+1]]})[0]
            #test=self.test_mesh.eval({self.inputs:data[index[i]:index[i+1]]})
                logdr,s = recover_data_new(test_logdr,test_s, self.logdr_min, self.logdr_max, self.ds_min, self.ds_max,logdr_ismap=logdr_ismap,s_ismap=s_ismap)
                z_logdr = self.z_logdr_test.eval({self.inputs_logdr: self.logdr[index[i]:index[i+1]]})
                z_s = self.z_s_test.eval({self.inputs_s: self.s[index[i]:index[i+1]]})
                name = foldername + '/test_index_' + str(i) + '.h5'
                print name
                f = h5py.File(name, 'w')
                f['logdr'] = logdr
                f['s']=s
                f['feature_vector'] = np.concatenate((z_logdr,z_s),axis=1)
                f.close()

        return
    def get_g_mesh(self,path,restore,foldername):
        data=h5py.File(path)
        z=data['test_g']
        #z=np.transpose(z,[1,0])
        with tf.Session() as sess:
            self.saver.restore(sess, restore)
            index=list(xrange(0,len(self.data)-1,32))
            test = sess.run([self.g_mesh], feed_dict={self.g_z: z})[0]
            test = recover_data1(test, self.logdr_min, self.logdr_max, self.ds_min, self.ds_max)
            name = foldername + '/g_index.h5'
            print name
            f = h5py.File(name, 'w')
            f['test_mesh'] = test
            f['feature_vector'] = 0
            f.close()
        return
    def randomresult1(self, restore, random_batch, times, foldername):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)

            # random_batch = np.random.normal(loc=0.0, scale=scale, size=(10000, latent_zdim))
            test = sess.run([self.test_mesh], feed_dict={self.random: random_batch})

            test = recover_data(test, self.data_min, self.data_max)

            name = foldername + '/randomtest1' + str(times) + '.h5'
            print name
            # sio.savemat(name, {'test_mesh': test, 'latent_z':random_batch})
            f = h5py.File(name, 'w')
            f['test_mesh'] = test
            f['latent_z'] = random_batch
            f.close()

        return


# In[9]:

def main():
    matname = './data/new_100.mat'#
    # data, data_min, data_max, rownum, colnum = load_data(matname, 'feature', 0.9, -0.9)
    meshvae = meshVAE(matpath=matname)
    meshvae.train()
    #meshvae.train('./20180308_182924new_100/meshvae.model-17000')#
    #meshvae.save_z(matname,'./20180308_223538new_100/meshvae.model-7000','./result')#
    #meshvae.get_g_mesh('./test_index_100000.mat','./20180303_050727test/meshvae.model-10000','./result')#
    
    
    
    
    # scale = 1
    # foldername = './genmatconv_' + str(scale)
    # if not os.path.isdir(foldername):
    #     os.mkdir(foldername)
    #
    # for i in range(0, 1):
    #     meshvae.randomresult('./'+timecurrent+'meshvae.model-' + str(epoch_num), i + 1, scale, foldername)
    #
    # scale = 3
    # foldername = './genmatconv_' + str(scale)
    # if not os.path.isdir(foldername):
    #     os.mkdir(foldername)
    # for i in range(0, 1):
    #     meshvae.randomresult('./'+timecurrent+'meshvae.model-' + str(epoch_num), i + 1, scale, foldername)


def rando():
    matname = 'togFeature1.mat'
    # data, data_min, data_max, rownum, colnum = load_data(matname, 'feature', 0.9, -0.9)
    meshvae = meshVAE(matpath=matname)
    scale = 1
    foldername = './genmat_test_conv' + str(scale)
    if not os.path.isdir(foldername):
        os.mkdir(foldername)

    for i in range(0, 1):
        meshvae.randomresult('./meshvae.model-' + str(epoch_num), i + 1, scale, foldername)

        # scale = 3
        # foldername = './genmat_' + str(scale)
        # if not os.path.isdir(foldername):
        #     os.mkdir(foldername)
        # for i in range(0, 1):
        #     meshvae.randomresult('./meshvae.model-' + str(epoch_num), i + 1, scale, foldername)


def interpola():
    matname = 'togFeature.mat'
    scale = 1.0
    random_batch = np.random.normal(loc=0.0, scale=scale, size=(2, latent_zdim))
    random2_intpl = interpolate.griddata(
        np.linspace(0, 1, len(random_batch) * 1), random_batch,
        np.linspace(0, 1, len(random_batch) * 25), method='linear')

    meshvae = meshVAE(matpath=matname, neighbour=neighbour)
    foldername = './genmat_itpl' + '_'
    if not os.path.isdir(foldername):
        os.mkdir(foldername)

    for i in range(0, 1):
        meshvae.randomresult1('./meshvae.model-' + str(epoch_num), random2_intpl, i + 1, foldername)


# if __name__ == '__main__':
# rando()
# interpola()

# if __name__ == '__main__':
# rando()
main()
