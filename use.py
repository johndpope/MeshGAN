from Model.AE import *

matpath = './Data/Dyna/mesh/50002_chicken_wings/ACAP_data.mat'
gc_dim = [8, 16, 32]
fc_dim = [512, 128]
max_degree = 2

model = MeshAE(matpath, gc_dim, fc_dim, max_degree)

model.use()
