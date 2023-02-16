import unittest

from databallpy.visualize import requires_ffmpeg


class TestVisualize(unittest.TestCase):
    def test_requires_ffmpeg_wrapper(self):
        @requires_ffmpeg
        def test_function():
            return "Hello World"

        self.assertEqual(test_function(), "Hello World")
