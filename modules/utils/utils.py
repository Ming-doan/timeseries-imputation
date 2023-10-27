"""
Utility functions for the project.
"""

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def ml_shape_repair(*arr: npt.NDArray[np.float32]):
    """
    Repairs the shape of the input data for machine learning models.
    """
    return tuple(a.squeeze() for a in arr)


def forecast_support(predict_func, x: npt.NDArray[np.float32], steps: int, **kwargs):
    """
    Support for forecast functions.
    """
    forecasted = []
    for _ in tqdm(range(steps), desc='Forecasting'):
        # Predict the output
        y = predict_func(x, **kwargs)
        # Append the output
        forecasted.append(y)
        # Update the input
        x = np.concatenate((x, y.reshape(1, -1)), axis=1)[:, 1:]
    return np.array(forecasted).squeeze()


def show_table(data: list[list[str]], cols: list[str], max_length: int = 20):
    """
    Show a table in the console.
    """
    # Calculate max padding
    max_padd = [len(col) for col in cols]
    for row in data:
        for i, col in enumerate(row):
            if len(str(col)) > max_length:
                max_padd[i] = max_length
            elif len(str(col)) > max_padd[i]:
                max_padd[i] = len(str(col))
    # Trim data
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if len(str(col)) > max_length:
                data[i][j] = str(col)[:max_length - 3] + '...'
    # Show top border
    print('┌', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┬', end='')
    print('┐')
    # Show columns
    print('│', end='')
    for i, col in enumerate(cols):
        print(f' {col}{" " * (max_padd[i] - len(str(col)))} │', end='')
    print()
    # Show divider
    print('├', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┼', end='')
    print('┤')
    # Show content
    for row in data:
        print('│', end='')
        for i, col in enumerate(row):
            print(f' {col}{" " * (max_padd[i] - len(str(col)))} │', end='')
        print()
    # Show bottom border
    print('└', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┴', end='')
    print('┘')
