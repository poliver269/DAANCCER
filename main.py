from datetime import datetime

from my_tsne import TrajectoryTSNE
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory, TopologyConverter, MultiTrajectory
from utils.param_key import *

COMPARE = 'compare'
MULTI_COMPARE_ALL_PCS = 'multi_compare_all_pcs'
MULTI_COMPARE_COMBO_PCS = 'multi_compare_combo_pcs'
COMPARE_WITH_TLTSNE = 'compare_with_tltsne'
PLOT_WITH_SLIDER = 'plot_with_slider'
COMPARE_WITH_CA_ATOMS = 'compare_with_carbon_alpha_atoms'
BASE_TRANSFORMATION = 'base_transformation'
CALCULATE_PEARSON_CORRELATION_COEFFICIENT = 'calculate_pcc'


def main():
    print(f'Starting time: {datetime.now()}')
    # TODO: Argsparser for options
    run_option = MULTI_COMPARE_ALL_PCS
    trajectory_name = '2f4k'
    file_element = 0
    params = {
        PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map'
        PLOT_TICS: True,  # True, False
        STANDARDIZED_PLOT: True,
        CARBON_ATOMS_ONLY: True,  # True, False
        INTERACTIVE: True,  # True, False
        N_COMPONENTS: 2,
        LAG_TIME: 10,
        TRUNCATION_VALUE: 0,
        BASIS_TRANSFORMATION: False,
        USE_ANGLES: False,
        TRAJECTORY_NAME: trajectory_name
    }

    if trajectory_name == '2f4k':
        filename_list = [f'2F4K-0-protein-{i:03d}.dcd' for i in range(0, 62 + 1)] + ['tr3_unfolded.xtc',
                                                                                     'tr8_folded.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': '2f4k.pdb', 'folder_path': 'data/2f4k',
                  'params': params}
    elif trajectory_name == 'prot2':
        filename_list = ['prod_r1_nojump_prot.xtc', 'prod_r2_nojump_prot.xtc', 'prod_r3_nojump_prot.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': 'prod_r1_pbc_fit_prot_last.pdb',
                  'folder_path': 'data/ProtNo2', 'params': params}
    elif trajectory_name == 'savinase':
        filename_list = ['savinase_1.xtc', 'savinase_2.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': 'savinase.pdb',
                  'folder_path': 'data/Savinase', 'params': params}
    elif trajectory_name == '2wav':
        filename_list = [f'2WAV-0-protein-{i:03d}.dcd' for i in range(0, 136)]
        kwargs = {'filename': filename_list[file_element], 'topology_filename': '2wav.pdb',
                  'folder_path': 'data/2WAV-0-protein', 'params': params}
    elif trajectory_name == '5i6x':
        filename_list = ['protein.xtc', 'system.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': '5i6x.pdb',
                  'folder_path': 'data/ser-tr', 'params': params}
    else:
        raise ValueError(f'No data trajectory was found with the name `{trajectory_name}`.')
    filename_list.pop(file_element)

    if run_option == 'covert_to_pdb':
        kwargs = {'filename': 'protein.xtc', 'topology_filename': 'protein.gro',
                  'goal_filename': 'protein.pdb', 'folder_path': 'data/ser-tr'}
        tc = TopologyConverter(**kwargs)
        tc.convert()
    elif run_option == COMPARE_WITH_TLTSNE:
        tr = TrajectoryTSNE(**kwargs)
        tr.compare('tsne')
    elif run_option == PLOT_WITH_SLIDER:
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == COMPARE:
        tr = DataTrajectory(**kwargs)
        # tr.compare(['tica', 'mytica', 'kernel_only_tica', 'tensor_tica', 'tensor_kernel_tica', 'tensor_kp_tica',
        #             'tensor_ko_tica', 'tensor_comad_tica', 'tensor_comad_kernel_tica'])
        model_params_list = [
            # {ALGORITHM_NAME: 'original_pca', NDIM: 2},
            {ALGORITHM_NAME: 'pca', NDIM: 3},
            # {ALGORITHM_NAME: 'pca', NDIM: 3, KERNEL: KERNEL_ONLY, KERNEL_TYPE: MY_GAUSSIAN},
            # {ALGORITHM_NAME: 'pca', NDIM: 3, KERNEL: KERNEL_DIFFERENCE, KERNEL_TYPE: MY_GAUSSIAN},
            # {ALGORITHM_NAME: 'original_tica', NDIM: 2, LAG_TIME: params[LAG_TIME]},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME]},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_DIFFERENCE},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_ONLY},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_MULTIPLICATION, KERNEL_TYPE: MY_LINEAR},
        ]
        tr.compare(model_params_list)
    elif run_option == COMPARE_WITH_CA_ATOMS:
        tr = DataTrajectory(**kwargs)
        tr.compare_with_carbon_alpha_and_all_atoms('pca')
    elif run_option == BASE_TRANSFORMATION:
        tr = DataTrajectory(**kwargs)
        tr.compare_with_basis_transformation(['tica'])
    elif run_option == CALCULATE_PEARSON_CORRELATION_COEFFICIENT:
        tr = DataTrajectory(**kwargs)
        tr.calculate_pearson_correlation_coefficient()
    elif run_option.startswith('multi'):
        kwargs_list = [kwargs]
        for filename in filename_list:
            new_kwargs = kwargs.copy()
            new_kwargs['filename'] = filename
            kwargs_list.append(new_kwargs)
        model_params_list = [
            # {ALGORITHM_NAME: 'original_pca', NDIM: MATRIX_NDIM},
            {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM},
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: False, CENTER_OVER_TIME: False},
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, KERNEL: KERNEL_DIFFERENCE, USE_STD: True,
            #  CENTER_OVER_TIME: False},
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, KERNEL: KERNEL_DIFFERENCE, USE_STD: False,
            #  CENTER_OVER_TIME: False},
            # {ALGORITHM_NAME: 'original_tica', NDIM: 2, LAG_TIME: params[LAG_TIME]},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME]},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_DIFFERENCE},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_ONLY},
            # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_MULTIPLICATION,
            #  KERNEL_TYPE: MY_LINEAR},
        ]
        if run_option == 'multi_trajectory':
            mtr = MultiTrajectory(kwargs_list, params)
            mtr.compare_pcs(['tensor_ko_pca', 'tensor_ko_tica'])
        elif run_option == MULTI_COMPARE_COMBO_PCS:
            mtr = MultiTrajectory(kwargs_list, params)
            mtr.compare_trajectory_combos(traj_nrs=[63, 64], model_params_list=model_params_list, pc_nr_list=[2, 9, 30])
        elif run_option == MULTI_COMPARE_ALL_PCS:
            mtr = MultiTrajectory(kwargs_list, params)
            # mtr.compare_similarity_of_pcs(traj_nrs=None, model_params_list=model_params_list,
            # pc_nr_list=[2, 9, 30], cosine_only=False)
            mtr.compare_all_trajectories(traj_nrs=None, model_params_list=model_params_list,
                                         pc_nr_list=None)

    print(f'Finishing time: {datetime.now()}')


if __name__ == '__main__':
    main()
