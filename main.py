from datetime import datetime
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    run_option = 'compare_with_pyemma'
    tr = DataTrajectory('tr3_unfolded.xtc', topology_filename='2f4k.gro', folder_path='data/2f4k')
    if run_option == 'plot_with_slider':
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 10]
    elif run_option == 'c_msm':
        tr.compare_with_msmbuilder('tica', 'pca')
    else:  # compare_with_pyemma
        tr.compare_with_pyemma('tica', 'pca')

    print('Finishing time: {}'.format(datetime.now()))


if __name__ == '__main__':
    main()
