import numpy as np
from scipy.spatial import distance


def generate_independent_matrix(row, dimension):
    """
    https://stackoverflow.com/questions/61905733/generating-linearly-independent-columns-for-a-matrix
    :param row:
    :param dimension:
    :return:
    """
    matrix = np.random.rand(row, 1)
    rank = 1
    while rank < dimension:
        t = np.random.rand(row, 1)
        if np.linalg.matrix_rank(np.hstack([matrix, t])) > rank:
            matrix = np.hstack([matrix, t])
            rank += 1
    return matrix


def basis_transform(matrix, dimension):
    independent_matrix = generate_independent_matrix(dimension, dimension)
    transformation_matrix = independent_matrix @ np.identity(dimension)
    return np.einsum('tac,cc->tac', matrix, transformation_matrix)


def explained_variance(eigenvalues, component):
    return np.sum(eigenvalues[:component]) / np.sum(eigenvalues)


def distance_matrix(size):
    initial_matrix = np.asarray([[a] for a in range(size)])
    weight_matrix = distance.squareform(distance.pdist(initial_matrix, 'euclidean'))  # canberra, sqeuclidean, None
    normalization = 'normal'
    if normalization == 'exponential':
        return (np.max(weight_matrix) * weight_matrix + 1) / (weight_matrix + 1)
    else:  # 'normal'
        return weight_matrix / np.max(weight_matrix)


def gkern(size, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def exponentiated_quadratic(size, sig=0.4):
    """Exponentiated quadratic  with σ=1"""
    xlim = (-1, 1)
    xa = np.expand_dims(np.linspace(*xlim, size), 1)
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * distance.cdist(xa, xa, 'sqeuclidean')
    return np.exp(sq_norm / np.square(sig))
