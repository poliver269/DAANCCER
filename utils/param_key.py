# Trajectory Dimensions
TIME_FRAMES = 'time_frames'
TIME_DIM = 0
ATOMS = 'atoms'
ATOM_DIM = 1
COORDINATES = 'coordinates'
COORDINATE_DIM = 2

# Trajectory Params
PLOT_TYPE = 'plot_type'
PLOT_TICS = 'plot_tics'
STANDARDIZED_PLOT = 'standardized_plot'
CARBON_ATOMS_ONLY = 'carbon_atoms_only'
INTERACTIVE = 'interactive'
N_COMPONENTS = 'n_components'
LAG_TIME = 'lag_time'
TRUNCATION_VALUE = 'truncation_value'
BASIS_TRANSFORMATION = 'basis_transformation'
RANDOM_SEED = 'random_seed'
USE_ANGLES = 'use_angles'
TRAJECTORY_NAME = 'trajectory_name'
FILE_ELEMENT = 'file_element'

# Model Results
MODEL = 'model'
PROJECTION = 'projection'
TITLE_PREFIX = 'title_prefix'
EXPLAINED_VAR = 'explained_variance'
INPUT_PARAMS = 'input_params'

# Plot Types
HEAT_MAP = 'heat_map'
COLOR_MAP = 'color_map'
PLOT_3D_MAP = '3d_map'
PLOT_2D = 'plot_2d'
EXPL_VAR_PLOT = 'explained_variance_plot'

# Coordinates
X = 'x'
Y = 'y'
Z = 'z'

# Angel Tuple Index
ANGEL_INDICES = 0
DIHEDRAL_ANGEL_VALUES = 1

# Model Parameters
COV_STAT_FUNC = 'cov_stat_func'
KERNEL_STAT_FUNC = 'kernel_stat_func'
ALGORITHM_NAME = 'algorithm_name'
NDIM = 'ndim'
KERNEL = 'kernel'
KERNEL_TYPE = 'kernel_type'
CORR_KERNEL = 'corr_kernel'
ONES_ON_KERNEL_DIAG = 'ones_on_kernel_diag'
COV_FUNCTION = 'cov_function'
NTH_EIGENVECTOR = 'nth_eigenvector'
EXTRA_DR_LAYER = 'extra_dr_layer'
EXTRA_LAYER_ON_PROJECTION = 'extra_layer_on_projection'
ABS_EVAL_SORT = 'abs_eigenvalue_sorting'
USE_STD = 'use_std'
CENTER_OVER_TIME = 'center_over_time'

# Kernel functions
GAUSSIAN = 'gaussian'
EPANECHNIKOV = 'epanechnikov'
EXPONENTIAL = 'exponential'
LINEAR = 'linear'
MY_GAUSSIAN = 'my_gaussian'
MY_EPANECHNIKOV = 'my_epanechnikov'
MY_EXPONENTIAL = 'my_exponential'
MY_LINEAR = 'my_linear'  # 0/N\0
MY_LINEAR_INVERSE = 'my_linear_inverse'  # N\0/N
MY_LINEAR_INVERSE_P1 = 'my_linear_inverse+1'  # N-1\010/N-1
MY_LINEAR_NORM = 'my_linear_norm'  # 0/1\0
MY_LINEAR_INVERSE_NORM = 'my_linear_inverse_norm'  # 1\0/1

# Kernel Mappings:
KERNEL_ONLY = 'only'
KERNEL_DIFFERENCE = 'diff'
KERNEL_MULTIPLICATION = 'multi'

ARRAY_NDIM = 1
MATRIX_NDIM = 2
TENSOR_NDIM = 3
