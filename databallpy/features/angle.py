import numpy as np

# def get_smallest_angle(a, b, angle_format="radian"):
#     """
#     Function to calculate the smallest angle between 2 2D vectors.

#     :param a: list, first vector
#     :param b: list, second vector
#     :param angle_format: str, how to return the angle {"degree", "radian"}
#     :returns: float, the smallest angle
#     """
#     import pdb; pdb.set_trace()
#     assert len(a) == len(b) == 2, "a and b should be of len(2)"
#     assert (
#         angle_format == "degree" or angle_format == "radian"
#     ), f"input 'format' must be 'degree' or 'radian', not '{angle_format}'."

#     angle_a = np.arctan2(a[1], a[0])
#     angle_b = np.arctan2(b[1], b[0])
#     smallest_angle_radians = np.min(
#         [abs(angle_a - angle_b), 2 * np.pi - abs(angle_a - angle_b)]
#     )

#     if angle_format == "radian":
#         return smallest_angle_radians
#     else:
#         return np.rad2deg(smallest_angle_radians)


def get_smallest_angle(a_vec, b_vec, angle_format="radian"):
    """
    Function to calculate the smallest angle between 2 2D vectors.

    :param a: numpy array, first vector of shape (n, 2)
    :param b: numpy array, second vector of shape (n, 2)
    :param angle_format: str, how to return the angle {"degree", "radian"}
    :returns: numpy array, the smallest angle of shape (n,)
    """
    if not isinstance(a_vec, np.ndarray):
        a_vec = np.array(a_vec)
    if not isinstance(b_vec, np.ndarray):
        b_vec = np.array(b_vec)

    b_vec = b_vec.astype("float64")
    a_vec = a_vec.astype("float64")

    assert a_vec.shape == b_vec.shape, "a and b should have the same shape"

    if len(a_vec.shape) == 1:  # 1D array
        a_vec = a_vec.reshape(1, -1)
        b_vec = b_vec.reshape(1, -1)

    assert a_vec.shape[1] == 2, "a and b should have shape (n, 2)"
    assert (
        angle_format == "degree" or angle_format == "radian"
    ), f"input 'format' must be 'degree' or 'radian', not '{angle_format}'."

    angle_a = np.arctan2(a_vec[:, 1], a_vec[:, 0])
    angle_b = np.arctan2(b_vec[:, 1], b_vec[:, 0])

    smallest_angle_radians = np.min(
        [np.abs(angle_a - angle_b), 2 * np.pi - np.abs(angle_a - angle_b)], axis=0
    )
    if len(smallest_angle_radians) == 1:
        smallest_angle_radians = smallest_angle_radians[0]

    if angle_format == "radian":
        return smallest_angle_radians
    else:
        return np.rad2deg(smallest_angle_radians)
