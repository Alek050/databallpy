from databallpy.load_data.metadata import Metadata
from typing import Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm

import numpy as np
import pandas as pd
import bs4

def load_fifa_format_data(fifa_loc:str, meta_data_loc, verbose:bool) -> Tuple[pd.DataFrame, Metadata]:
    """

    Args:
        fifa_loc (str): _description_
        verbose (bool): _description_

    Returns:
        Tuple[pd.DataFrame, Metadata]: _description_
    """
    #tracking_data = _get_tracking_data(fifa_loc, verbose)
    meta_data = _get_meta_data(meta_data_loc, verbose)

    return meta_data

def _get_tracking_data(fifa_loc:str, verbose:bool) -> pd.DataFrame:
    """_summary_

    Args:
        fifa_loc (str): _description_
        verbose (bool): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if verbose:
        print(f"Reading in {tracab_loc}", end="")

    file = open(tracab_loc, "r")
    lines = file.readlines()
    
    if verbose:
        print(" - Completed")

    file.close()
    

def _get_meta_data(meta_data_loc:str, verbose:bool) -> Metadata:
    """_summary_

    Args:
        meta_data_loc (str): _description_
        verbose (bool): _description_

    Returns:
        Metadata: _description_
    """
    
    file = open(meta_data_loc, "r", encoding="UTF-8").read()
    soup = BeautifulSoup(file, "xml")
    import pdb;pdb.set_trace()