import numpy as np

from databallpy.utils.logging import logging_wrapper


@logging_wrapper(__file__)
def get_smallest_angle(
    a_vec: np.ndarray, b_vec: np.ndarray, angle_format="radian"
) -> np.ndarray | float:
    """Calculate the smallest angle between two vectors.

    Args:
        a_vec (np.ndarray): A numpy array of shape (n, 2) or (2,). The first column is
            the x-component and the second column is the y-component.
        b_vec (np.ndarray): A numpy array of shape (n, 2) or (2,). The first column is
            the x-component and the second column is the
        angle_format (str, optional): The angle format, either `radian` or `degree`.
            Defaults to "radian".

    Returns:
        np.ndarray | float: The smallest angle between the two vectors.
    """
    for input_list in [a_vec, b_vec]:
        if not isinstance(input_list, list) and not isinstance(input_list, np.ndarray):
            raise TypeError(
                f"Input must be a numpy array, not {type(input_list).__name__}"
            )

    a_vec = (
        np.array(a_vec).astype("float64")
        if isinstance(a_vec, list)
        else a_vec.astype("float64")
    )
    b_vec = (
        np.array(b_vec).astype("float64")
        if isinstance(b_vec, list)
        else b_vec.astype("float64")
    )

    if not a_vec.shape == b_vec.shape:
        raise ValueError(
            "a and b should have the same shape a is "
            f"{a_vec.shape} and b is {b_vec.shape}"
        )
    if angle_format not in ["degree", "radian"]:
        raise ValueError(
            f"input 'format' must be 'degree' or 'radian', not '{angle_format}'."
        )

    if len(a_vec.shape) == 1:  # 1D array
        a_vec = a_vec.reshape(1, -1)
        b_vec = b_vec.reshape(1, -1)

    if not a_vec.shape[1] == 2 or not b_vec.shape[1] == 2:
        raise ValueError(
            f"a and b should have shape (n, 2), not {a_vec.shape} and {b_vec.shape}"
        )

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
