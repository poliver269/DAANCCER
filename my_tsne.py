import mdtraj as md
import scipy as sp
import sklearn.manifold as sk
import scipy.spatial as spat
from mdtraj import Trajectory

from trajectory import DataTrajectory

"""
Parts of this file were originally copied from the tltsne python module.  
https://github.com/spiwokv/tltsne/blob/master/tltsne/__init__.py
Since the results in the text file are complicated to reuse, this module was modified somewhat.
This way, the results of the models can be used and it's Object Oriented.  
"""


class TrajectoryTSNE(DataTrajectory):
    def __init__(self, filename, topology_filename, folder_path, params=None):
        super().__init__(filename, topology_filename, folder_path, params)
        if not self.topology_path.endswith('.pdb'):
            raise TypeError(f'Topology File is in a wrong format! Make sure, that you convert it in a pdb-file.')
        self.ref_pdb: Trajectory = md.load_pdb(self.topology_path)
        if params is None:
            params = {}
        params = {
            'superpose': params.get('superpose', False),
            'tica_dim': params.get('tica_dim', 2),
            'max_principal_components': params.get('max_principal_components', 50),
            'perplex_tsne': params.get('perplex_tsne', 10.0),
            'perplex_tltsne': params.get('perplex_tltsne', 10.0),
            'exaggeration': params.get('exaggeration', 12.0),
            'learning_rate': params.get('learning_rate', 200.0),
            'n_iter': params.get('n_iter', 1000)
        }
        self.params = {**self.params, **params}

        if self.params['superpose']:  # superimposing the trajectory to reference structure
            self.traj.superpose(self.ref_pdb)

        if self.params['lag_time'] > self.traj.n_frames:
            raise ValueError("Lag time higher than the number of frames.")

    def start_all_and_save(self, out_filename):
        projs_pca = self.get_model_and_projection_by_name('pca')[1]
        projs_tica = self.get_model_and_projection_by_name('tica')[1]
        x_emb_tsne = self.get_embedded('tsne')[1]
        x_emb_tltsne = self.get_embedded('time-lagged_tsne')[1]

        models = [projs_pca, projs_tica, [x_emb_tsne], [x_emb_tltsne]]
        self.save_results(models, out_filename)

    # noinspection PyTupleAssignmentBalance,PyUnresolvedReferences
    def get_embedded(self, model_name):
        print(f"Running {model_name}...")
        if model_name == 'tsne':
            model = sk.TSNE(
                n_components=self.params['n_components'], perplexity=self.params['perplex_tsne'],
                early_exaggeration=self.params['exaggeration'], learning_rate=self.params['learning_rate'],
                n_iter=self.params['n_iter'], metric="euclidean"
            )
            embedding = model.fit_transform(self.flattened_coordinates)
            return model, embedding
        elif model_name == 'time-lagged_tsne':
            # TODO speed up tl tSNE
            traj_mean = self.flattened_coordinates - sp.mean(self.flattened_coordinates, axis=0)
            traj_cov = sp.cov(sp.transpose(traj_mean))  # np.linalg.cov
            eigenvalue, eigenvector = sp.linalg.eig(traj_cov)
            eigenvalue_order = sp.argsort(eigenvalue)[::-1]
            eigenvector = eigenvector[:, eigenvalue_order]
            eigenvalue = eigenvalue[eigenvalue_order]
            projs = traj_mean.dot(eigenvector)
            projs = projs / sp.sqrt(eigenvalue)
            if self.params['lag_time'] <= 0:
                component1 = sp.transpose(projs[:, ]).dot(projs[:, ]) / (
                            self.traj.n_frames - 1)
            else:
                component1 = sp.transpose(projs[:-self.params['lag_time'], ]).dot(projs[self.params['lag_time']:, ]) / (
                            self.traj.n_frames - self.params['lag_time'] - 1)
            component1 = (component1 + sp.transpose(component1)) / 2
            eigenvalue2, eigenvector2 = sp.linalg.eig(component1)
            eigenvalue_order = sp.argsort(eigenvalue2)[::-1]
            eigenvector2 = eigenvector2[:, eigenvalue_order]
            eigenvalue2 = eigenvalue2[eigenvalue_order]
            projs = projs.dot(eigenvector2[:, :self.params['max_principal_components']])
            projs = projs * sp.sqrt(sp.real(eigenvalue2[:self.params['max_principal_components']]))
            traj_distance = spat.distance_matrix(projs, projs)
            model = sk.TSNE(
                n_components=self.params['n_components'], perplexity=self.params['perplex_tltsne'],
                early_exaggeration=self.params['exaggeration'], learning_rate=self.params['learning_rate'],
                n_iter=self.params['n_iter'], metric="precomputed"
            )
            embedding = model.fit_transform(traj_distance)
            return model, embedding

    def compare(self, model_name1):
        if model_name1 in ['tsne']:
            model, embedding = self.get_embedded(model_name1)
            projection = [embedding]
        else:  # tica / pca
            model, projection = self.get_model_and_projection_by_name(model_name1)
        tl_tsne, embedding = self.get_embedded('time-lagged_tsne')
        print(model, tl_tsne, sep='\n')
        self.compare_with_plot([{'model': model, 'projection': projection},
                                {'model': tl_tsne, 'projection': [embedding], 'title_prefix': 'time-lagged'}])

    def save_results(self, models, out_filename='output.txt'):
        projs_pca, projs_tica, x_emb_tsne, x_emb_tltsne = models
        print("Saving results...")
        with open(out_filename, 'w') as out_file:
            if self.params['superpose']:
                out_file.write("# structures were superimposed onto reference structure\n")
            else:
                out_file.write("# structures were NOT superimposed onto reference structure\n")
            out_file.write("# lag time set to %i frames\n" % self.params['lag_time'])
            out_file.write("# output dimension for PCA set to %i\n" % self.params['n_components'])
            out_file.write("# output dimension for TICA set to %i\n" % self.params['n_components'])
            out_file.write("# number of top principle components passed to time-lagged t-SNE set to %i\n" %
                           self.params['max_principal_components'])
            out_file.write(
                "# output dimension for t-SNE and time-lagged t-SNE set to %i\n" % self.params['n_components'])
            out_file.write("# perplexity of t-SNE set to %f\n" % self.params['perplex_tsne'])
            out_file.write("# perplexity of time-lagged t-SNE set to %f\n" % self.params['perplex_tltsne'])
            out_file.write("# early_exaggeration set to %f\n" % self.params['exaggeration'])
            out_file.write("# structure_ID")
            for model_name in ['PCA', 'TICA', 'tSNE', 'tltSNE']:
                for j in range(self.params['n_components']):
                    out_file.write(f" {model_name}%i" % (j + 1))
            out_file.write("\n")
            for i in range(self.traj.n_frames):
                output = " %i" % (i + 1)
                for model_projs in [projs_pca, projs_tica, x_emb_tsne, x_emb_tltsne]:
                    for j in range(self.params['n_components']):
                        output = output + " %f" % model_projs[0][i, j]
                out_file.write("%s\n" % output)
