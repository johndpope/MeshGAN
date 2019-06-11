import numpy as np
import scipy.sparse as sp
import h5py
from scipy.sparse.linalg.eigen.arpack import eigsh


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def tuple_to_dense(sparse):
    """Convert Sparse Tuple to Dense Matrix"""
    def to_dense(sparse):
        dense = np.zeros(sparse[2], dtype=np.float32)
        for i in range(len(sparse[0])):
            dense[sparse[0][i][0]][sparse[0][i][1]] = float(sparse[1][i])
        return dense

    if isinstance(sparse, list):
        for i in range(len(sparse)):
            sparse[i] = to_dense(sparse[i])
    else:
        sparse = to_dense(sparse)

    return sparse


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def load_neighbour(neighbour, edges, is_padding=False):
    data = neighbour

    if is_padding == True:
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
    return logdrnew, snew, e_nb, p_nb, degree, logdrmin, logdrmax, smin, smax, modelnum, pointnum, edgenum, maxdegree

