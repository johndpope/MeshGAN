from Model.AE import *

matpath = './Data/Dyna/mesh/50002_chicken_wings/ACAP_data.mat'
gc_dim = [10, 10, 10]
fc_dim = [1024, 512]
max_degree = 2

i = 0
j = 20
num = 20

model = MeshAE(matpath, gc_dim, fc_dim, max_degree)

model.interpolation(i, j, num)
