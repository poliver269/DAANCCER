from datetime import datetime

from my_tsne import TrajectoryTSNE
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory, TopologyConverter


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    run_option = 'calculate_pcc'
    params = {
        'plot_type': '3d_map',  # 'heat_map', 'color_map', '3d_map'
        'plot_tics': False,  # True, False
        'carbon_atoms_only': True,  # True, False
        'interactive': True,  # True, False
        'lag_time': 10,
        'truncation_value': 30,
        'basis_transformation': False
    }
    kwargs = {'filename': 'tr8_folded.xtc', 'topology_filename': '2f4k.pdb', 'folder_path': 'data/2f4k',
              'params': params}
    if run_option == 'covert_gro_to_pdb':
        kwargs = {'filename': 'tr3_unfolded.xtc', 'topology_filename': '2f4k.gro',
                  'goal_filename': '2f4k.pdb', 'folder_path': 'data/2f4k'}
        tc = TopologyConverter(**kwargs)
        tc.convert()
    elif run_option == 'compare_with_tltsne':
        tr = TrajectoryTSNE(**kwargs)
        tr.compare('tsne')
    elif run_option == 'compare_angles':
        tr = DataTrajectory(**kwargs)
        tr.compare_angles(['pca', 'tica'])
    elif run_option == 'plot_with_slider':
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == 'compare_with_msm':  # only with 3.5 and under
        tr = DataTrajectory(**kwargs)
        tr.compare_with_msmbuilder('tica', 'pca')
    elif run_option == 'compare_with_pyemma':
        tr = DataTrajectory(**kwargs)
        # tr.compare_with_pyemma(['pca', 'mypca', 'trunc_pca'])
        tr.compare_with_pyemma(['pca', 'tica'])  # , 'mytica', 'trunc_tica'])
        # tr.compare_with_pyemma(['tica', 'mytica'])
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
