from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from utils.algorithms.tensor_dim_reductions.daanccer import DAANCCER

def cos_sim(angle_i, angle_j):
    return angle_i.T @ angle_j / (np.linalg.norm(angle_i) * np.linalg.norm(angle_j))

def get_ortho_unit(unit_vec):
    # Get random unit vector
    vector = np.random.normal(0, 1, unit_vec.shape)
    vector /= np.linalg.norm(vector)

    # Subtract projection of vector onto unit_vec and renormalize
    vector -= (vector.T @ unit_vec) * unit_vec
    vector /= np.linalg.norm(vector)

    return vector

def fake_protein_folds(length=100, n_points=40, dim=10, n_trajectories=3, step_size=0.1, initial_angle_min=-0.5):
    random_initial_pos = np.zeros([n_points, dim])
    angles = np.zeros([n_points, dim])
    angles[0] = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
    angles[0] /= np.linalg.norm(angles[0])
    for i in range(1, n_points):
        angle = -angles[i-1]
        while cos_sim(angle, angles[i-1]) < initial_angle_min:
            angle = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
            angle /= np.linalg.norm(angle)
        angles[i] = angle
        random_initial_pos[i] = random_initial_pos[i-1] + angle

    # Get orthogonal vectors to angles
    positions = np.zeros([n_trajectories, length, n_points, dim])
    positions[:, 0] = random_initial_pos
    for t in tqdm(range(n_trajectories)):
        for i in range(1, length):
            prev_angles = positions[t, i-1, np.arange(1, n_points), :] - positions[t, i-1, np.arange(0, n_points-1), :]
            first_index_step = np.random.normal(0, 1, [dim])
            first_index_step /= np.linalg.norm(first_index_step)
            positions[t, i, 0] = positions[t, i-1, 0] + first_index_step * step_size
            for j in range(1, n_points):
                prev_angle = prev_angles[j-1]
                # ortho_direction = get_ortho_unit(prev_angle)
                step = prev_angle + np.random.normal(0, step_size, prev_angle.shape) #ortho_direction * step_size
                step /= np.linalg.norm(step)
                positions[t, i, j] = positions[t, i, j-1] + step

    for i in range(length):
        plt.scatter(positions[0, i, :, 0], positions[0, i, :, 1], c=np.arange(n_points))
        plt.scatter(positions[0, i, :, 0], positions[0, i, :, 1], c=np.arange(n_points))
        plt.scatter(positions[0, i, :, 0], positions[0, i, :, 1], c=np.arange(n_points))
    plt.show()

    positions = np.transpose(positions, [0, 1, 3, 2])
    positions = np.reshape(positions, [n_trajectories, -1, n_points])

    return positions

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt_tsp(cities, improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
    return route # When the route is no longer improving substantially, stop searching and return the route.

# Below gives less separation for RMSE plot but PCA and DROPP have different curve shapes
# def generate_trajectories(n_copies=10, dim=10, length=500, n_landmarks=50, noise=0.25):
#     X = np.zeros((n_copies, dim, length))
#     landmark_time_stamps = np.arange(0, length + 1, length / (n_landmarks - 1))
# 
#     global_landmarks = np.zeros((n_landmarks, dim))
#     global_landmarks[0] = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
#     global_landmarks[0] /= np.linalg.norm(global_landmarks[0])
#     for i in range(1, n_landmarks):
#         step_dir = get_ortho_unit(global_landmarks[i-1])
#         next_landmark = global_landmarks[i-1] + step_dir * 0.005
#         next_landmark /= np.linalg.norm(next_landmark)
#         global_landmarks[i] = next_landmark
#     global_landmarks = np.array(global_landmarks)
# 
#     for trajectory in range(n_copies):
#         traj_landmarks = global_landmarks + np.random.normal(0, 0.05, (n_landmarks, dim))
#         traj_landmarks /= np.linalg.norm(traj_landmarks)
#         # FOR ANNA -- notice that here I'm adding a cubic spline to interpolate between the random walk points.
#         # The other settings do not have a cubic spline and just use the traj_landmarks to get the points that we will
#         # treat as similar or not similar
#         spline = CubicSpline(landmark_time_stamps, traj_landmarks)
#         locations = spline(np.arange(length))
#         X[trajectory] = locations.T
# 
#     return X

# Below gives really good separation for RMSE plot but PCA and DROPP look super similar in shape
def generate_trajectories(n_copies=10, dim=10, n_landmarks=50, noise=0.25):
    X = np.zeros((n_copies, dim, n_landmarks))

    global_landmarks = np.zeros((n_landmarks, dim))
    global_landmarks[0] = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
    global_landmarks[0] /= np.linalg.norm(global_landmarks[0])
    for i in range(1, n_landmarks):
        step_dir = get_ortho_unit(global_landmarks[i-1])
        next_landmark = global_landmarks[i-1] + step_dir * 0.1
        next_landmark /= np.linalg.norm(next_landmark)
        global_landmarks[i] = next_landmark
    global_landmarks = np.array(global_landmarks)

    for trajectory in range(n_copies):
        traj_landmarks = global_landmarks + np.random.normal(0, 0.01, (n_landmarks, dim))
        traj_landmarks /= np.linalg.norm(traj_landmarks)
        X[trajectory] = traj_landmarks.T

    return X

# Below gives really good separation for RMSE plot but PCA and DROPP look super similar in shape
# def generate_trajectories(n_copies=10, dim=10, n_landmarks=1000, noise=0.25):
#     X = np.zeros((n_copies, dim, n_landmarks))
# 
#     global_landmarks = np.zeros((n_landmarks, dim))
#     global_landmarks[0] = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
#     global_landmarks[0] /= np.linalg.norm(global_landmarks[0])
#     for i in range(1, n_landmarks):
#         step_dir = get_ortho_unit(global_landmarks[i-1])
#         next_landmark = global_landmarks[i-1] + step_dir * 0.005
#         next_landmark /= np.linalg.norm(next_landmark)
#         global_landmarks[i] = next_landmark
#     global_landmarks = np.array(global_landmarks)
# 
#     for trajectory in range(n_copies):
#         traj_landmarks = global_landmarks + np.random.normal(0, 0.05, (n_landmarks, dim))
#         traj_landmarks /= np.linalg.norm(traj_landmarks)
#         X[trajectory] = traj_landmarks.T
# 
#     return X

def reconstruction_score(x, eigvecs, eps=1e-5):
    # Get the projection of x onto the eigenvectors
    proj = np.zeros_like(x)
    for i, v in enumerate(x):
        for u in eigvecs:
            norm = u @ u.T
            if norm < 1e-5:
                continue
            proj[i] += (v @ u.T / norm) * u

    # Return the sum of squared errors between the projection and the input
    return np.mean(np.square(x - proj))

def median_similarity(train_components, test_components):
    assert train_components.shape == test_components.shape
    dot_prods = np.abs(train_components @ test_components.T)
    for i in range(len(dot_prods)):
        row = dot_prods[i]
        argmax = np.argmax(row)
        if i != argmax:
            temp = dot_prods[:, i]
            dot_prods[:, i] = dot_prods[:, argmax]
            dot_prods[:, argmax] = temp
    return np.diag(dot_prods)


def compare_reconstructions():
    # data = fake_protein_folds()
    data = generate_trajectories(normalize=True)
    train_traj = data[0]
    d = train_traj.shape[0]
    C = np.eye(d) - np.ones((d, d)) / float(d)

    rank = np.min(train_traj.shape)
    reconstructions = np.zeros((2, rank, len(data) - 1))
    use_std = True
    for n_components in tqdm(range(1, rank), total=rank - 1):
        pca_train = PCA(n_components).fit(train_traj)
        dancer_train = DAANCCER(ndim=2).fit(train_traj, n_components=n_components, use_std=use_std)

        for j, trajectory in enumerate(data[1:]):
            trajectory = C @ trajectory
            reconstructions[0, n_components, j] = reconstruction_score(trajectory, pca_train.components_)
            reconstructions[1, n_components, j] = reconstruction_score(trajectory, dancer_train.components_)

    pca_similarities = []
    dancer_similarities = []
    pca_models, dancer_models = [], []
    for traj in data:
        pca_models.append(PCA(rank).fit(traj))
        dancer_models.append(DAANCCER(ndim=2).fit(traj, n_components=rank, use_std=use_std))
    for i, i_traj in enumerate(data):
        pca_train = pca_models[i]
        dancer_train = dancer_models[i]
        for j, j_traj in enumerate(data[i:]):
            pca_test = pca_models[j]
            dancer_test = dancer_models[j]
            pca_similarities.append(median_similarity(pca_train.components_, pca_test.components_))
            dancer_similarities.append(median_similarity(dancer_train.components_, dancer_test.components_))

    med_pca_similarities = np.median(np.array(pca_similarities), axis=0)
    med_dancer_similarities = np.median(np.array(dancer_similarities), axis=0)

    C_0 = train_traj.T @ train_traj
    # Center the dataset
    cov_size = C_0.shape[0]
    centering = np.eye(cov_size) - np.ones([cov_size, cov_size]) / cov_size
    C_0 = centering @ C_0 @ centering
    C_0 /= np.expand_dims(np.diag(C_0), [-1])
    plt.imshow(C_0, cmap='hot', interpolation='nearest')
    cbar = plt.colorbar()
    plt.show()
    plt.close()

    plt.fill_between(
        np.arange(rank-1),
        np.min(reconstructions[0, 1:], axis=-1),
        np.max(reconstructions[0, 1:], axis=-1),
        alpha=0.4,
        color='b'
    )
    plt.fill_between(
        np.arange(rank-1),
        np.min(reconstructions[1, 1:], axis=-1),
        np.max(reconstructions[1, 1:], axis=-1),
        alpha=0.4,
        color='r'
    )
    plt.plot(np.arange(rank-1), np.median(reconstructions[0, 1:], axis=-1), color='b')
    plt.plot(np.arange(rank-1), np.median(reconstructions[1, 1:], axis=-1), color='r')
    plt.show()
    plt.close()

    plt.plot(np.arange(rank-1), med_pca_similarities[:-1], color='b')
    plt.plot(np.arange(rank-1), med_dancer_similarities[:-1], color='r')
    plt.show()

if __name__ == '__main__':
    compare_reconstructions()
