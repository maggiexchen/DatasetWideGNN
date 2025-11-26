"""
Class to encode the user configuration yaml
"""
import logging
from typing import Optional

import yaml
from pydantic import BaseModel

class MLConfig(BaseModel):
    """
    Class to define the user config structure
    """

    kinematic_variable: str
    embedding_variable: Optional[str]
    distance: str
    friend_graph: bool
    edge_weights: bool
    edge_frac: Optional[float]
    targettarget_eff: Optional[float]
    linking_length: Optional[int]
    hidden_sizes_gcn: list[int]
    hidden_sizes_mlp: list[int]
    dropout_rates: list[float]
    LR: float
    patience_LR: int
    num_nb_list: list[int]
    batch_size: int
    gnn_type: str
    epochs: int
    patience_early_stopping: int
    single_fold: Optional[bool]
    plot_conv_kinematics: bool

    # Pydantic type settings
    model_config = {"coerce_numbers_to_str": True}

    @classmethod
    def from_yaml(cls, yaml_path:str) -> "MLConfig":
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

        if data["kinematic_variable"] is None:
            raise ValueError("Need to specify a type of kinematic variable in the ML config")

        if data["embedding_variable"] is None:
            data["embedding_variable"] = data["kinematic_variable"]

        if data["distance"] is None:
            raise ValueError("Need to specify a type of distance metric in the ML config")

        for key, val in data.items():
            logging.info("%s: %s", key, str(val))

        return cls(**data)
