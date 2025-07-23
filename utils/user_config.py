"""
Class to encode the user configuration yaml
"""
import logging
from typing import Optional, Dict

import utils.misc as misc

import yaml
from pydantic import BaseModel


class CutConfig(BaseModel):
    """
    Class to define the cut dictionary structure
    """
    threshold: float
    operation: str

class UserConfig(BaseModel):
    """
    Class to define the user config structure
    """

    # signal/background setup
    signal: str
    signal_mass: Optional[str] = None
    backgrounds: list[str]
    cuts: Dict[str, CutConfig]
    cutstring: Optional[str] = None
    feature_dim: Optional[int] = None

    # user running settings
    run_with_cuda: bool
    n_folds: int

    # input paths
    ntuple_path: str
    feature_h5_path: str
    kinematic_h5_path: str
    ll_path: str
    dist_path: str

    # output paths
    plot_path: str
    adj_path: str
    model_path: str
    score_path: str

    # Pydantic type settings
    model_config = {"coerce_numbers_to_str": True}

    @classmethod
    def from_yaml(cls, yaml_path:str) -> "UserConfig":
        """
        Function to import yaml and set all the variables
            for the user config
        
        Args:
            yaml_path: path to user config yaml file
        
        Returns:
            all the user config flat variables and dicts.
        """
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        if data["signal_mass"] is not None:
            data["signal_mass"] = str(data["signal_mass"])
        assert data["signal"] in ["hhh", "LQ", "stau"], "Invalid signal type"

        data["cutstring"] = misc.get_cutstring(data["cuts"])

        for key, val in data.items():
            logging.info("%s: %s", key, str(val))

        return cls(**data)
