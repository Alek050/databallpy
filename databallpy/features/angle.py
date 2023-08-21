import numpy as np


def get_smallest_angle(a, b, angle_format="radian"):
    """
    Function to calculate the smallest angle between 2 2D vectors.

    :param a: list, first vector
    :param b: list, second vector
    :param angle_format: str, how to return the angle {"degree", "radian"}
    :returns: float, the smallest angle
    """

    assert len(a) == len(b) == 2, "a and b should be of len(2)"
    assert (
        angle_format == "degree" or angle_format == "radian"
    ), f"input 'format' must be 'degree' or 'radian', not '{angle_format}'."

    angle_a = np.arctan2(a[1], a[0])
    angle_b = np.arctan2(b[1], b[0])
    smallest_angle_radians = np.min(
        [abs(angle_a - angle_b), 2 * np.pi - abs(angle_a - angle_b)]
    )

    if angle_format == "radian":
        return smallest_angle_radians
    else:
        return np.rad2deg(smallest_angle_radians)
