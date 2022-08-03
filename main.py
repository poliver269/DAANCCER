from datetime import datetime

from my_tsne import TLtSNE
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    run_option = 'compare_with_pyemma'
    kwargs = {'filename': 'tr3_unfolded.xtc', 'topology_filename': '2f4k.pdb', 'folder_path':'data/2f4k'}
    if run_option == 'compare_with_tltsne':
        tr = TLtSNE(**kwargs)
        tr.compare('tsne')
    if run_option == 'compare_degrees':
        tr = TLtSNE(**kwargs)
        tr.compare_angles('tica')
    elif run_option == 'plot_with_slider':
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == 'compare_with_msm':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_msmbuilder('tica', 'pca')
    elif run_option == 'compare_with_pyemma':
        tr = DataTrajectory(**kwargs)
        tr.compare_with_pyemma('tica', 'pca')

    print('Finishing time: {}'.format(datetime.now()))


if __name__ == '__main__':
    main()
