from datetime import datetime

from my_tsne import TrajectoryTSNE
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory, TopologyConverter


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    run_option = 'base_transformation'
    params = {
        'plot_type': 'color_map',  # 'heat_map', 'color_map'
        'plot_tics': True,  # True, False
        'carbon_atoms_only': True,  # True, False
        'interactive': True,  # True, False
        'lag_time': 10,
        'basis_transformation': True
    }
    kwargs = {'filename': 'tr3_unfolded.xtc', 'topology_filename': '2f4k.pdb', 'folder_path': 'data/2f4k',
              'params': params}
    if run_option == 'covert_gro_to_pdb':
        kwargs = {'filename': 'tr3_unfolded.xtc', 'topology_filename': '2f4k.gro',
                  'goal_filename': '2f4k.pdb', 'folder_path': 'data/2f4k'}
        tc = TopologyConverter(**kwargs)
        tc.convert()
    elif run_option == 'compare_with_tltsne':
        tr = TrajectoryTSNE(**kwargs)
        tr.compare('tica')
    elif run_option == 'compare_degrees':
        tr = TrajectoryTSNE(**kwargs)
        tr.compare_angles('tica')
    elif run_option == 'plot_with_slider':
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == 'compare_with_msm':  # only with 3.5 and under
        tr = DataTrajectory(**kwargs)
        tr.compare_with_msmbuilder('tica', 'pca')
    elif run_option == 'compare_with_pyemma':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_pyemma('tica', 'pca')
    elif run_option == 'compare_with_ac':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_ac_and_all_atoms('tica')
    elif run_option == 'base_transformation':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_basis_transformation(['pca'])

    print('Finishing time: {}'.format(datetime.now()))


if __name__ == '__main__':
    main()
