# Master thesis implementation - Olivér Palotás
## 1. Create conda environment
```
conda env create -f .conda.yml
```

## 2. Get Molecular Dynamics Data
(e.g.) FS-Peptide:
> https://figshare.com/articles/dataset/Fs_MD_Trajectories/1030363


## 3. Configurate Model Parameters
To run the program, models with different parameters can be used and trained.
Use predefined ***.json* file** or config the parameters in a new file in the folder *config_files\algorithm*.
The different parameters are explained here, this includes important and optional parameters.
Note: The different parameters in upper-case can be imported `from utils.param_key import *`,
although the string values of the parameters are written in lower-case and should be used in the *.json*-config-files.

### Algorithms
In the following some main algorithms with its parameter settings are listed:
1. PCA
    - `{ALGORITHM_NAME: 'original_pca', NDIM: MATRIX_NDIM}` or
    - `{ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: False, ABS_EVAL_SORT: False}`
2. TICA
    - `{ALGORITHM_NAME: 'original_tica', NDIM: MATRIX_NDIM}` or
    - `{ALGORITHM_NAME: 'tica', LAG_TIME: params[LAG_TIME], NDIM: MATRIX_NDIM, USE_STD: False, ABS_EVAL_SORT: False}`
3. raw MATRIX models
    - `{ALGORITHM_NAME: 'pca', NDIM: MATRIX_NDIM}`
    - `{ALGORITHM_NAME: 'tica', NDIM: MATRIX_NDIM, LAG_TIME: params[LAG_TIME]}`

4. raw TENSOR models
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
2. NDIM:
    - MATRIX_NDIM (=2)
    - TENSOR_NDIM (=3)

#### Optional Parameters
The different parameters and their different options are listed here below.
1. KERNEL (Choose how to map the kernel-matrix onto the covariance matrix,
*default: kernel-matrix not used*)
    - KERNEL_ONLY
    - KERNEL_DIFFERENCE
    - KERNEL_MULTIPLICATION
2. KERNEL_TYPE (Choose the kernel-function which should be fitted on the covariance matrix,
*default (if kernel set): MY_GAUSSIAN*)
    - MY_GAUSSIAN
    - MY_EXPONENTIAL
    - MY_LINEAR
    - MY_EPANECHNIKOV
    - GAUSSIAN, EXPONENTIAL, LINEAR, EPANECHNIKOV (only with interval 1 fitting)
3. COV_FUNCTION (Choose how to calculate the Covariance-Matrix for the algorithm,
*default: np.cov*)
    - np.cov
    - np.corrcoef
    - utils.matrix_tools.co_mad
4. NTH_EIGENVECTOR (Set this parameter to *>1*, if you want to use the Eigenvalue Selection Approach
with every *n*-th eigenvector [Not recommended], *default: 1*)
    - [int]
5. LAG_TIME (Set this parameter in connection with ALGORITHM_NAME: 'tica', *default: 0*)
    - [int]

### Boolean Parameters
1. CORR_KERNEL (Set this parameter in connection with ALGORITHM_NAME 'tica'.
If *True*, then the fitted kernel-matrix is also mapped on the correlation matrix [Not recommended], *default: False*)
2. ONES_ON_KERNEL_DIAG (This parameter is useful, to force the diagonally dominant matrix properties in some cases,
*default: False*)
3. USE_STD (An additional standardizing preprocessing step can be used within the algorithm,
*default: True*)
4. CENTER_OVER_TIME (An additional standardizing preprocessing step can be used within the algorithm,
*default: True*)
5. EXTRA_DR_LAYER (Set this parameter to True, if you want to use the Eigenvalue Selection approach with a second layer
[Not Recommended], *default: False*)

## 4. Configure Run options/parameters
Additionally, use different options to run the program. Config the parameters in a in a ***.json* file** 
in the folder *config_files\options*.
The different parameters are explained here.
*Note:* The different parameters in upper-case can be imported `from utils.param_key import *`,
although the string values of the parameters are written in lower-case and should be used in the *.json*-config-files.

## 5. Run the program
```python main.py -o config_files\options\run_options_file.json -a config_files\algorithms\pca+daanccer+tica.json```