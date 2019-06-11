from Model.VAE import *

matpath = 'D:\\CS\\AAAI2020\\MeshGAN\\Data\\Dyna\\mesh\\50002_chicken_wings\\new\\tog16_point_edge.mat'
gc_dim = [2, 2]
fc_dim = [1024, 512]
max_degree = 1

model = MeshAE(matpath, gc_dim, fc_dim, max_degree)
