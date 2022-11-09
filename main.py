from datetime import datetime

from my_tsne import TrajectoryTSNE
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory, TopologyConverter
from utils.param_key import *


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    run_option = 'compare'
    trajectory_name = 'prot2'
    params = {
        PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map'
        PLOT_TICS: True,  # True, False
        STANDARDIZED_PLOT: False,
        CARBON_ATOMS_ONLY: True,  # True, False
        INTERACTIVE: True,  # True, False
        LAG_TIME: 100,
        TRUNCATION_VALUE: 9,
        BASIS_TRANSFORMATION: False,
        USE_ANGLES: False,
        TRAJECTORY_NAME: trajectory_name
    }
    if trajectory_name == '2f4k':
        kwargs = {'filename': 'tr3_unfolded.xtc', 'topology_filename': '2f4k.pdb', 'folder_path': 'data/2f4k',
                  'params': params}
    elif trajectory_name == 'prot2':
        kwargs = {'filename': 'prod_r1_nojump_prot.xtc', 'topology_filename': 'prod_r1_pbc_fit_prot_last.pdb',
                  'folder_path': 'data/ProtNo2', 'params': params}
    elif trajectory_name == 'savinase':
        kwargs = {'filename': 'savinase_1.xtc', 'topology_filename': 'savinase.pdb',
                  'folder_path': 'data/Savinase', 'params': params}
    else:
        raise ValueError(f'No data trajectory was found with the name `{trajectory_name}`.')

    if run_option == 'covert_gro_to_pdb':
        kwargs = {'filename': 'tr3_unfolded.xtc', 'topology_filename': '2f4k.gro',
                  'goal_filename': '2f4k.pdb', 'folder_path': 'data/2f4k'}
        tc = TopologyConverter(**kwargs)
        tc.convert()
    elif run_option == 'compare_with_tltsne':
        tr = TrajectoryTSNE(**kwargs)
        tr.compare('tsne')
    elif run_option == 'plot_with_slider':
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == 'compare':
        tr = DataTrajectory(**kwargs)
        # tr.compare(['tensorPCA', 'mytica', 'tensor_tica', 'tensor_kernel_tica',
        #           'tensor_kp_tica', 'tensor_ko_tica', 'tensor_comad_tica'])
        # tr.compare(['pca', 'koPCA', 'tica', 'tensor_ko_tica'])  # best
        # tr.compare(['pca', 'koPCA'])
        # tr.compare(['pca', 'mypca', 'pca_kernel_only'])  # , 'mytica', 'trunc_tica'])
        tr.compare(['pca', 'tensor_ko_pca', 'tica', 'tensor_tica'])  # , 'tica_kernel_only'])
    elif run_option == 'compare_with_carbon_alpha_atoms':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_carbon_alpha_and_all_atoms('pca')
    elif run_option == 'base_transformation':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_basis_transformation(['tica'])
    elif run_option == 'calculate_pcc':
        tr = DataTrajectory(**kwargs)
        tr.calculate_pearson_correlation_coefficient()

    print('Finishing time: {}'.format(datetime.now()))


if __name__ == '__main__':
    main()
