from Model.AE import *

matpath = './Data/Dyna/mesh/50002_chicken_wings/ACAP_data.mat'
gc_dim = [10, 10, 10]
fc_dim = [1024, 512]
max_degree = 2

epoch = 5000
batchsize = 10
lr = 1e-3
continuous_training = False

model = MeshAE(matpath, gc_dim, fc_dim, max_degree)

model.train(epoch, batchsize, lr, continuous_training)
