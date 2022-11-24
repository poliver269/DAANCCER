from datetime import datetime

from my_tsne import TrajectoryTSNE
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory, TopologyConverter, MultiTrajectory
from utils.param_key import *


def main():
    print(f'Starting time: {datetime.now()}')
    # TODO: Argsparser for options
    run_option = 'multi_2_pcs'
    trajectory_name = '2f4k'
    file_element = 0
    params = {
        PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map'
        PLOT_TICS: True,  # True, False
        STANDARDIZED_PLOT: False,
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
        filename_list = ['tr3_unfolded.xtc', 'tr8_folded.xtc'] + [f'2F4K-0-protein-{i:03d}.dcd' for i in range(0, 6)]
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
        filename_list = [f'2WAV-0-protein-{i:03d}.dcd' for i in range(0, 10)]
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
    elif run_option == 'compare_with_tltsne':
        tr = TrajectoryTSNE(**kwargs)
        tr.compare('tsne')
    elif run_option == 'plot_with_slider':
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == 'compare':
        tr = DataTrajectory(**kwargs)
        # tr.compare(['tica', 'mytica', 'kernel_only_tica', 'tensor_tica', 'tensor_kernel_tica', 'tensor_kp_tica',
        #             'tensor_ko_tica', 'tensor_comad_tica', 'tensor_comad_kernel_tica'])
        model_params_list = [
            {ALGORITHM_NAME: 'pca', NDIM: 3},
            {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME]},
            {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME], KERNEL: KERNEL_MULTIPLICATION,
             KERNEL_TYPE: MY_LINEAR},
            {ALGORITHM_NAME: 'pca', NDIM: 3, KERNEL: KERNEL_MULTIPLICATION, KERNEL_TYPE: MY_LINEAR},
        ]
        tr.compare(model_params_list)
    elif run_option == 'compare_with_carbon_alpha_atoms':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_carbon_alpha_and_all_atoms('pca')
    elif run_option == 'base_transformation':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_basis_transformation(['tica'])
    elif run_option == 'calculate_pcc':
        tr = DataTrajectory(**kwargs)
        tr.calculate_pearson_correlation_coefficient()
    elif run_option.startswith('multi'):
        kwargs_list = [kwargs]
        for filename in filename_list:
            new_kwargs = kwargs.copy()
            new_kwargs['filename'] = filename
            kwargs_list.append(new_kwargs)
        if run_option == 'multi_trajectory':
            mtr = MultiTrajectory(kwargs_list, params)
            mtr.compare_pcs(['tensor_ko_pca', 'tensor_ko_tica'])
        elif run_option == 'multi_2_pcs':
            mtr = MultiTrajectory(kwargs_list, params)
            model_params_list = [
                # {ALGORITHM_NAME: 'original_pca', NDIM: 2},
                # {ALGORITHM_NAME: 'pca', NDIM: 3},
                {ALGORITHM_NAME: 'pca', NDIM: 3, KERNEL: KERNEL_DIFFERENCE, KERNEL_TYPE: MY_GAUSSIAN},
                # {ALGORITHM_NAME: 'pca', NDIM: 3, KERNEL: KERNEL_MULTIPLICATION, KERNEL_TYPE: MY_LINEAR},
                # {ALGORITHM_NAME: 'original_tica', NDIM: 2, LAG_TIME: params[LAG_TIME]},
                # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME]},
                # {ALGORITHM_NAME: 'tica', NDIM: 3, LAG_TIME: params[LAG_TIME],
                #  KERNEL: KERNEL_DIFFERENCE, KERNEL_TYPE: MY_GAUSSIAN}
            ]
            # mtr.compare_similarity_of_pcs(traj_nrs=None, model_params_list=model_params_list,
            # pc_nr_list=[2, 9, 30], cosine_only=False)
            mtr.compare_similarity_of_pcs(traj_nrs=[0, 1], model_params_list=model_params_list,
                                          pc_nr_list=[2, 9, 30], cosine_only=True)

            pass

    print(f'Finishing time: {datetime.now()}')


if __name__ == '__main__':
    main()
