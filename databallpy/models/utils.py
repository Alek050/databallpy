import numpy as np


def scale_and_predict_logreg(input: np.array, params: dict) -> float:
    """Function that scales the input data and predict using a logistic regression
    model.

    Args:
        input (np.array): The input data to be scaled and predicted.
        params (dict): The parameters for scaling and prediction.

    Returns:
        float: The predicted value.
    """

    if not isinstance(input, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if not len(input.shape) == 2:
        raise ValueError("Input must be a 2D array.")
    if not input.shape[1] == len(params["standard_scaler"]["mean"]):
        raise ValueError(
            "Input shape does not match the number of features in the model."
        )

    mean_values = np.array([*params["standard_scaler"]["mean"].values()])
    var_values = np.array([*params["standard_scaler"]["var"].values()])
    scaled_data = (input - mean_values) / np.sqrt(var_values)

    coefs = np.array([x for x in params["logreg"]["coefs"].values()])
    intercept = params["logreg"]["intercept"]

    return 1 / (1 + np.exp(-(np.dot(scaled_data, coefs) + intercept)))


def get_xt_prediction(
    x: float | np.ndarray, y: float | np.ndarray, xt_model: np.ndarray
) -> float | np.ndarray:
    """Function to get the predicted xT based on a position and a xT model

    Args:
        x (float | np.ndarray): x coordinate on the pitch (-53, 53).
        y (float| np.ndarray): y coordinate on the pitch (-34, 34).
        xt_model (np.ndarray): The xT model to use.

    Returns:
        float | np.ndarray: xT value(s)
    """
    x_cells, y_cells = xt_model.shape

    x_idx = np.clip((x + 53) * (x_cells / 106), a_min=0, a_max=x_cells - 1).astype(int)
    y_idx = np.clip((y + 34) * (y_cells / 68), a_min=0, a_max=y_cells - 1).astype(int)

    return xt_model[x_idx, y_idx]
