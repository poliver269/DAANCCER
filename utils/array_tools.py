import numpy as np


def split_list_in_half(array):
    """
    https://stackoverflow.com/questions/752308/split-list-into-smaller-lists-split-in-half
    :param array: array (list, ndarray)
    :return: tuple with the first and the second half of the list
    """
    half = len(array) // 2
    return array[:half], array[half:]


def interpolate_center(symmetrical_array, stat_func: callable = np.median):
    """
    Interpolates the data in the array center, and zeroes all the other values
    :param symmetrical_array: (nd)array symmetrical
    :param stat_func: Some statistical function of an array: np.median (default), np.mean, ...
    :return:
    """
    symmetrical_array = interpolate_array(symmetrical_array, stat_func)
    return extinct_side_values(symmetrical_array)


def interpolate_array(array, stat_func: callable = np.median, interp_range=None):
    """
    Interpolate an array from a range, of its statistical value (mean, median, min, ...) to the maximum,
    into a new range `interp_range` (default: 0-1)
    :param array:
    :param stat_func: Some statistical function of an array: np.median (default), np.mean, ...
    :param interp_range: the new range for the interpolation
    :return: the new interpolated array
    """
    if interp_range is None:
        interp_range = [0, 1]
    statistical_value = stat_func(array)
    return np.interp(array, [statistical_value, array.max()], interp_range)


def extinct_side_values(symmetrical_array, smaller_than=0):
    """
    Takes an array and searches the first value from the center smaller than the given value.
    All the border values from that criteria are zeroed
    :param symmetrical_array: symmetrical (nd)array
    :param smaller_than: int
        Sets the criteria to find the index between the center and the border.
    :return:
    """
    right_i = np.argmax(split_list_in_half(symmetrical_array)[1] <= smaller_than)
    center_i = len(symmetrical_array) // 2
    new_y = np.zeros_like(symmetrical_array)
    new_y[center_i - right_i:center_i + right_i] = symmetrical_array[center_i - right_i:center_i + right_i]
    return new_y
