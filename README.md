# DROPP (Dimensionality Reduction for Ordered Points with PCA)
A novel **dimensionality reduction technique** designed to preserve the inherent order in data. 
Unlike conventional methods, DROPP leverages a Gaussian kernel function to capture 
relationships between closely ordered data points.

# Set up
## 1. Create conda environment
```
conda env create -f .conda.yml
```

## 2. Get (example) Data
Molecular Dynamics Data (e.g. FS-Peptide):
> https://figshare.com/articles/dataset/Fs_MD_Trajectories/1030363

Climate Data:
> https://data.open-power-system-data.org/weather_data/

Save the data in the *data* folder. For climate data .cvs (Sub folder: weather_data)  run the preprocessing steps for the countries 
(Specify countries in *config_files\options\preprocess_data*)
```
python main.py -o config_files\optionsT\preprocess_climate_data.json
```


## 3. Configurate Model Parameters
To run the program, models with different parameters can be evaluated.
Use predefined ***.json* file** in the folder *config_files\algorithm* or 
config the parameters in a new file. Write the models in a list of dicts inside the .json file.
The different parameters are explained here, this includes important and optional parameters.

Note: The different parameters in upper-case can be imported `from utils.param_keys.model import *`,
although the string values of the parameters are written in lower-case and should be used in the *.json*-config-files.

### Algorithms
In the following some main algorithms with its parameter settings are listed:
1. PCA
    - `{ALGORITHM_NAME: 'original_pca', NDIM: MATRIX_NDIM}` or
    - `{ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: False, ABS_EVAL_SORT: False}`
2. TICA
    - `{ALGORITHM_NAME: 'original_tica', NDIM: MATRIX_NDIM}` or
    - `{ALGORITHM_NAME: 'tica', LAG_TIME: params[LAG_TIME], NDIM: MATRIX_NDIM, USE_STD: False, ABS_EVAL_SORT: False}`
3. raw MATRIX models (DROPP)
    - `{ALGORITHM_NAME: 'pca', NDIM: MATRIX_NDIM}`
    - `{ALGORITHM_NAME: 'tica', NDIM: MATRIX_NDIM, LAG_TIME: params[LAG_TIME]}`

4. raw TENSOR models (DROPP)
    - `{ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM}`
    - `{ALGORITHM_NAME: 'tica', NDIM: TENSOR_NDIM, LAG_TIME: params[LAG_TIME]}`

### Parameters
#### Required Parameters
These parameters are **mandatory** for a correct program run!
1. ALGORITHM_NAME:
    - 'pca'
    - 'tica'
    - ('original_pca')
    - ('original_tica')
    - ('original_ica')
2. NDIM:
    - MATRIX_NDIM (=2)
    - TENSOR_NDIM (=3)

#### Optional Parameters
The different parameters and their different options are listed here below.
1. KERNEL_KWARGS (Kernel parameters, which are used to define the kernel on the DROPP-Algorithm)
`from utils.param_keys.kernel_functions import *`
   1. KERNEL_MAP (Choose how to map the kernel-matrix onto the covariance matrix;
   *default: kernel-matrix not used*)
       - KERNEL_ONLY (*default for DROPP*)
       - KERNEL_DIFFERENCE
       - KERNEL_MULTIPLICATION
       - None
   2. KERNEL_FUNCTION (Choose the kernel-function which should be fitted on the covariance matrix;
   *default (if kernel set): MY_GAUSSIAN*) 
       - MY_GAUSSIAN
       - MY_EXPONENTIAL
       - MY_LINEAR
       - MY_EPANECHNIKOV
       - MY_SINC(_SUM)
       - MY_COS
       - GAUSSIAN, EXPONENTIAL, LINEAR, EPANECHNIKOV (only with interval 1 fitting)
   3. KERNEL_STAT_FUNC (Choose the statistical function which determines the threshold 
   for the rescaling the data before fitting the kernel function on the data; *default: statistical_zero*)
       - [str]
       - utils.statistical_zero
       - np.min
       - np.max
       - ... (numpy statistical function like)
   4. USE_ORIGINAL_DATA (If this parameter is set the rescaling of the data 
   before the fitting of kernel function is not happening; *default: False*)
       - [bool]
   5. CORR_KERNEL (Set this parameter in connection with ALGORITHM_NAME 'tica'.
   If *True*, then the fitted kernel-matrix is also mapped on the correlation matrix [Not recommended]; 
   *default: False*)
   6. ONES_ON_KERNEL_DIAG (This parameter is useful, to force the diagonally dominant matrix properties in some cases;
   *default: False*)
2. COV_FUNCTION (Choose how to calculate the Covariance-Matrix for the algorithm,
*default: np.cov*)
    - [str]
    - np.cov
    - np.corrcoef
    - utils.matrix_tools.co_mad
3. NTH_EIGENVECTOR (Set this parameter to *>1*, if you want to use the Eigenvalue Selection Approach
with every *n*-th eigenvector [Not recommended]; *default: 1*)
    - [int]
4. LAG_TIME (Set this parameter in connection with ALGORITHM_NAME: 'tica'; *default: 0*)
    - [int]
5. ANALYSE_PLOT_TYPE (Set this parameter to plot different information while using the model; *default: None*)
    - [str]
   1. Model Analysing strings
      - EIGENVECTOR_MATRIX_ANALYSE
      - COVARIANCE_MATRIX_PLOT
      - CORRELATION_MATRIX_PLOT
   2. Kernel Curves Analysing strings
      - PLOT_3D_MAP
      - PLOT_KERNEL_MATRIX_3D
      - WEIGHTED_DIAGONAL
      - FITTED_KERNEL_CURVES

### Boolean Parameters
1. USE_STD (An additional standardizing preprocessing step can be used within the algorithm;
*default: True*)
2. CENTER_OVER_TIME (An additional standardizing preprocessing step can be used within the algorithm;
*default: True*)
3. EXTRA_DR_LAYER (Set this parameter to True, if you want to use the Eigenvalue Selection approach with a second layer
[Not Recommended]; *default: False*)
4. ABS_EIGENVALUE_SORTING (Set this parameter to sort the eigenvalues and respectively the eigenvectors 
by the absolut eigenvalue; *default: True*)

## 4. Configure Run options/parameters
Additionally, use different options to run the program. Config the parameters in a ***.json* file** 
in the folder *config_files\options*.

*Note:* The different parameters in upper-case can be imported `from utils.param_keys.`,
although the string values of the parameters are written in lower-case and should be used in the *.json*-config-files.

#### Required Parameters
These parameters are **mandatory** for a correct program run!
1. 'run_option'
   1. `.run_potions` (Determines which evaluation process should be started)
      1. COMPARE: comparing different algorithms qualitatively (2 dimensions)
      2. PLOT_...: plot somehow
      3. LOAD_...: loads different evaluation results
      4. MULTI_...: uses multiple trajectories to evaluate the algorithms
      5. ...
2. 'data_set'
   1. 'protein'
   2. 'weather'
   3. 'sub_...' (protein or weather)
3. 'trajectory_name' (name of the folder in which the dataset is saved)
   1. e.g.: 'fs-peptide'

#### Optional Parameters
The different parameters and their different options are listed here below.
1. SUBSET_LIST [list] (list of int-s which elements should be loaded for a subset evaluation)
2. ENABLE_SAVE [bool] (if True: enables the saving of the evaluation results)
3. PLOT_TYPE (How or what to plot while evaluating the algorithms) `.analyses`
      1. COLOR_MAP (colored dots)
      2. ...
4. PLOT_TICS [Bool] (if True: enables the component-tics while using a COMPARE evaluation)
5. INTERACTIVE [Bool] (if True: uses an interactive plot window)
6. PLOT_FOR_PAPER [Bool] (if True: plots the figures more suited for the paper)
7. N_COMPONENTS [int, None] (if None: all the components are used, 
if int: the number of components are used while evaluating)
8. preprocessing parameters:
   1. BASIS_TRANSFORMATION
   2. CARBON_ATOMS_ONLY (for proteins only)
   3. RANDOM_SEED
   4. USE_ANGLES (for proteins only)
   5. SUPERPOSING_INDEX 
   6. REDUCEE_FEATURE ['temperature'/'radiation_direct_horizontal'/'radiation_diffuse_horizontal'] 
(For weather data only. Choose the feature you want to use for evaluation)
   7. MAIN_MODEL_PARAMS
   8. SEL_COL (For weather data only)
9. Subset trajectory parameters (for proteins mainly):
   1. QUANTITY [int] (determines the quantity of the subset size)
   2. TIME_WINDOW_SIZE [int] (determines the time window size of the subsets)
   3. PART_COUNT [int] (which part of the subset should be used as the main subset)
   4. TRANSFORM_ON_WHOLE [bool] (if True: the transformation is made on the whole data set and not on the subset)

## 5. Run the program
```python main.py -o config_files\options\run_params.json -a config_files\algorithm\pca+tica+dropp.json```