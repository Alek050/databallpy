import logging
import unittest
from logging import Logger

from databallpy.utils.logging import create_logger


class TestCreateLogger(unittest.TestCase):
    def test_create_logger(self):
        logger = create_logger(
            "test_logger", user_config_path="tests/test_data/test_config.ini"
        )

        self.assertIsInstance(logger, Logger)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(len(logger.handlers), 2)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        self.assertEqual(logger.handlers[0].level, logging.DEBUG)
        self.assertIsInstance(logger.handlers[1], logging.FileHandler)
        self.assertEqual(logger.handlers[1].level, logging.INFO)
