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
