from datetime import datetime
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    plot_option = 'only_one'
    tr = DataTrajectory('tr3_unfolded.xtc', topology_filename='2f4k.gro', folder_path='data/2f4k')
    if plot_option == 'with_slider':
        TrajectoryPlotter(tr).original_data_with_timestep_slider(min_max=None)  # [0, 10]
    else:  # only_one
        tr.compare('tica', 'pca')
    print('Finishing time: {}'.format(datetime.now()))


if __name__ == '__main__':
    main()
