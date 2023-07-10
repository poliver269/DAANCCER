import argparse


class ArgParser(object):
    @staticmethod
    def parse(description):
        parser = argparse.ArgumentParser(description=description)

        parser.add_argument(
            '-o',
            '--options',
            dest='run_params_json',
            help='a path destination to the parameter configuration of the program',
        )
        parser.add_argument(
            '-a',
            '--algorithm-configs',
            dest='alg_params_json',
            help='a path destination to the configurations of the algorithms'
        )
        parser.add_argument(
            '-l',
            '--load',
            dest='result_load_files',
            nargs='+',
            help='the list of path destinations of the results to load of the analyse'
        )
        parser.add_argument(
            '-d',
            '--load-dir',
            dest='loading_directory',
            help='a directory root where the data should be loaded'
        )
        return parser.parse_args()
