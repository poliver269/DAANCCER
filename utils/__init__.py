from sklearn.utils import deprecated

from math import log10, floor
from utils.param_keys import DUMMY_ZERO


def function_name(function: callable) -> str:
    """
    Does the same as function.__name__
    :param function: callable
    :return: str
        The function name extracted from the function
    """
    return str(function).split()[1]


def statistical_zero(_) -> int:
    """
    Is equally to a numpy min/max function to set the statistical zero
    :param _: np.ndarray but basically any
    :return: 0
    """
    return 0


def ordinal(n: int) -> str:
    """
    https://stackoverflow.com/a/20007730/11566305
    :param n: number
    :return:
    """
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


@deprecated
def pretify_dict_model(ugly_dict: dict) -> dict:
    fill_dict = {}
    for k, v in ugly_dict.items():
        if k.startswith('[PCA'):
            fill_dict['PCA'] = v
        elif k.startswith('[TICA'):
            fill_dict['TICA'] = v

        elif k.startswith('Tensor-pca, my_gaussian-only'):
            if k.startswith('Tensor-pca, my_gaussian-only-3rd'):
                fill_dict['DAANCCER Kernel-Only\n3rd EVS'] = v
            elif k.startswith('Tensor-pca, my_gaussian-only-2nd'):
                fill_dict['DAANCCER Kernel-Only\n2nd-PCA EVS'] = v
            else:
                fill_dict['DAANCCER Kernel-Only'] = v
        elif k.startswith('Tensor-pca, my_gaussian-diff'):
            fill_dict['DAANCCER Kernel-Subtraction'] = v
        elif k.startswith('Tensor-pca, my_gaussian-multi'):
            fill_dict['DAANCCER Kernel-Product'] = v

        elif k.startswith('Tensor-tica, my_gaussian-only'):
            fill_dict['DAANCCER Kernel-Only'] = v
        elif k.startswith('Tensor-tica, my_gaussian-diff'):
            fill_dict['DAANCCER Kernel-Subtraction'] = v
        elif k.startswith('Tensor-tica, my_gaussian-multi'):
            fill_dict['DAANCCER Kernel-Product'] = v
    return fill_dict


def get_algorithm_name(model) -> str:
    """
    Extracts the algorithm name from the model (e.g. PCA).
    This is split by the first opening bracket and takes everything before that
    :param model: object
        Dimension Reduction Model class (such as PCA)
    :return: str
        Model algorithm name
    """
    return str(model).split('(')[DUMMY_ZERO]


def nr_in_human_format(num: int):
    ends = ["", "k", "m", "b", "t", "q"]
    index = min(int(floor(log10(num) / 3)), len(ends) - 1)
    return str(num // 1000 ** index) + ends[index]
