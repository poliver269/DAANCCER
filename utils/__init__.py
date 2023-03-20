def function_name(function: callable):
    return str(function).split()[1]


def statistical_zero(_):
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
