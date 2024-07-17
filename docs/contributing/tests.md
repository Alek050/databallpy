# Testing Code


When you are writing code, you generally want to make sure that the code you write is working as expected.

```{image} https://i.pinimg.com/originals/fa/7b/68/fa7b688bcfd98c8ab70726d2d223ff73.jpg
:alt: Testing
:width: 400px
:align: center
```

In the short term, writing tests for your code is the most boring thing to do, and it often feels useless; you know the code is working since you just wrote it. Often, you are wrong, you just do not know it yet. In the long term, writing tests and thinking about everything that could go wrong in your code will make the whole developing process way more efficient since you do not have to come back to code you wrote months ago because you did not think of an edge case. Testing code is good for at least two things:

1. Making sure your code works; it does the job and handles edge cases well.
2. Making sure you keep your functions short and logical; if you have large functions, you need a lot of tests for one function, this is a sign that you might want to split up the function into different smaller functions to make the code more readable and easier to test. 

To show you what is meant with testing your code we will look at an example; let's say you're writing code to obtain the acceleration out of positional data:

```Python
import pandas as pd


def get_acceleration(
    df: pd.DataFrame, input_columns: list, framerate: int
) -> pd.DataFrame:
    """Function that adds acceleration columns based on the position 
    columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which velocity should 
        						    be calculated
        framerate (int): framerate of the tracking data

    Returns:
        pd.DataFrame: tracking data with the added acceleration columns
    """

    accelerations = []
    for input_column in input_columns:
        velocity = df[input_column].diff() / (1 / framerate)
        acceleration = velocity.diff() / (1 / framerate)
        accelerations.append(acceleration)

    accelerations_df = pd.concat(accelerations, axis=1)
    accelerations_df.columns = [f"{col}_a" for col in input_columns]
    df = pd.concat([df, accelerations_df], axis=1)

    return df
```

First note that clear variable names and build-in functions are used. This is also a small and readable function with clear flow and added docstrings to make the code even more understandable. The last thing it is missing is a test. For testing code you can use modules like `unittest`. A test for this function could look something like this:

```Python
import unittest

import numpy as np
import pandas as pd

from databallpy.features.acceleration import get_acceleration


class TestAcceleration(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
                "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
            }
        )

    def test_get_acceleration_framerate(self):
        output_1 = get_acceleration(self.input, ["ball_x", "ball_y"], 1)
        expected_output_1 = pd.DataFrame(
            {
               "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
               "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
               "ball_x_a": [np.nan, np.nan, -6, 18, -1, np.nan, np.nan],
               "ball_y_a": [np.nan, np.nan, -48, 84, -24, np.nan, np.nan],
            }
        )
        pd.testing.assert_frame_equal(output_1, self.expected_output)
        
        output_2 = get_acceleration(self.input, ["ball_x", "ball_y"], 2)
        expected_output_2 = pd.DataFrame(
            {
               "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
               "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
               "ball_x_a": [np.nan, np.nan, -24, 72, -4, np.nan, np.nan],
               "ball_y_a": [np.nan, np.nan, -192, 336, -96, np.nan, np.nan],
            }
        )
        pd.testing.assert_frame_equal(output_2, self.expected_output_2)
    
    def test_get_acceleration_one_column(self):
	    output = get_acceleration(self.input, ["ball_x"], 1)
	    expected_output_1 = pd.DataFrame(
            {
               "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
               "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
               "ball_x_a": [np.nan, np.nan, -6, 18, -1, np.nan, np.nan],
            }
        )
	   	
```

As you can see, we now test whether the acceleration is calculated as expected by using different framerate intervals and adding `np.nan` values to the input. Additionally, we verify that acceleration is only calculated for the specified columns. So is this right? Not really. If this were a internal function, never to be called by the users of the package, it would be fine. But this function is a feature that users might call themselves. Therefore it needs to be more elaborate. For instance, what happens when someone calls the function like this: `get_acceleration(df, "ball_x", 1)`, now the input type of the second argument is not a list but a string. In this case the function will loop over the letters in the string instead of over the columns in the list. Therefore, we need to test whether the input types are right as well. The expected behaviour is to raise a `TypeError` when the input value is not right, with an explanation of what went wrong so the user understands the error. Take for instance the example above, the error that would be raised if the type of the input is not checked is `KeyError: "b"`. Knowing the function, it is clear that "b" is not a column name in the dataframe, and that is what the error refers to, but for a user that is not known with the function code, this is a very vague error. We have to change the function, and thus also the test, to tackle this problem:

```Python
import pandas as pd


def get_acceleration(
    df: pd.DataFrame, input_columns: list, framerate: int
) -> pd.DataFrame:
    """Function that adds acceleration columns based on the position 
    columns

    Args:
        df (pd.DataFrame): tracking data
        input_columns (list): columns for which velocity should 
                              be calculated
        framerate (int): framerate of the tracking data

    Returns:
        pd.DataFrame: tracking data with the added acceleration columns
    """
    # Check input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, not a {type(df)}")
    if not isinstance(input_columns, list):
        raise TypeError(
            f"input_columns must be a list, not a {type(input_columns)}"
        )
    if not all(isinstance(col, str) for col in input_columns):
        raise TypeError("All elements in input_columns must be strings")
    if not isinstance(framerate, int):
        raise TypeError(
            f"framerate must be a integer, not a {type(framerate)}"
        )

    accelerations = []
    for input_column in input_columns:
        velocity = df[input_column].diff() / (1 / framerate)
        acceleration = velocity.diff() / (1 / framerate)
        accelerations.append(acceleration)

    accelerations_df = pd.concat(accelerations, axis=1)
    accelerations_df.columns = [f"{col}_a" for col in input_columns]
    df = pd.concat([df, accelerations_df], axis=1)

    return df
```

Now we also have to update our tests:

```Python
import unittest

import numpy as np
import pandas as pd

from databallpy.features.acceleration import get_acceleration


class TestAcceleration(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
                "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
            }
        )

    def test_get_acceleration_framerate(self):
        output_1 = get_acceleration(self.input, ["ball_x", "ball_y"], 1)
        expected_output_1 = pd.DataFrame(
            {
               "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
               "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
               "ball_x_a": [np.nan, np.nan, -6, 18, -1, np.nan, np.nan],
               "ball_y_a": [np.nan, np.nan, -48, 84, -24, np.nan, np.nan],
            }
        )
        pd.testing.assert_frame_equal(output_1, self.expected_output)
        
        output_2 = get_acceleration(self.input, ["ball_x", "ball_y"], 2)
        expected_output_2 = pd.DataFrame(
            {
               "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
               "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
               "ball_x_a": [np.nan, np.nan, -24, 72, -4, np.nan, np.nan],
               "ball_y_a": [np.nan, np.nan, -192, 336, -96, np.nan, np.nan],
            }
        )
        pd.testing.assert_frame_equal(output_2, self.expected_output_2)
    
    def test_get_acceleration_one_column(self):
	    output = get_acceleration(self.input, ["ball_x"], 1)
	    expected_output_1 = pd.DataFrame(
            {
               "ball_x": [1, 2, -3, 10, 22, np.nan, 21, 20],
               "ball_y": [10, 13, -32, 7, 22, np.nan, 21, 20],
               "ball_x_a": [np.nan, np.nan, -6, 18, -1, np.nan, np.nan],
            }
        )
        
     def test_get_acceleration_wrong_input(self):
        # dataframe
        with self.assertRaises(TypeError) as cm:
            get_acceleration({"ball_x": [1, 2, 3, 4]}, ["ball_x"], 1)
        self.assertEqual(
            str(cm.exception), 
            "df must be a pandas DataFrame, not a str"
        )
        
        # input_columns
     	with self.assertRaises(TypeError) as cm:
            get_acceleration(self.input, "ball_x", 1)
        self.assertEqual(
            str(cm.exception), 
            "input_columns must be a list, not a str"
        )

        with self.assertRaises(TypeError) as cm:
            get_acceleration(self.input, ["ball_x", 123], 1)
        self.assertEqual(
            str(cm.exception), 
            "All elements in input_columns must be strings"
        )
     	 
     	 # framerate
        with self.assertRaises(TypeError) as cm:
            get_acceleration(self.input, ["ball_x", "ball_y"], "1")
        self.assertEqual(
            str(cm.exception), 
            "framerate must be a int, not a str"
        )
```

As you can see, we now updated the function and the tests for the function. When a user will call `get_acceleration(df, "ball_x", 1)`, it will now raise a `TypeError`: `TypeError: "input_columns must be a list, not a str"` which is very clear for the user. The biggest advantage of testing your code, is that you can be confident that the code works as expected in different cases and will not raise vague errors when you are not expecting them. Of course, the coverage can not be perfect, but at least you tackle the biggest problems with this. Also, whenever you decide to refactor your code to make it more readable or efficient, you can keep your tests to ensure that the outcome of the code is still the same.


```{image} https://miro.medium.com/v2/resize:fit:700/1*bOASH2rdFfCZ8x_Jd4FFmg.png
:alt: Testing
:width: 800px
:align: center
```
