from typing import Union, Dict, List

import os
import torch
from torch.nn import Module


def get_unique_experience_name(experience_name, save_dir):
    """
        Generate a unique experience name by checking for existing directories in the save directory.

        If a directory with the same experience name already exists, the function appends a numeric suffix 
        (e.g., '_1', '_2', etc.) to ensure uniqueness. The function creates the save directory if it does not exist.

        Args:
            experience_name (str): The base name of the experience.
            save_dir (str): The path to the directory where experience directories are saved.

        Returns:
            str: A unique experience name in the format 'experience_name_x', where x is the next available number.
    """
    os.makedirs(save_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    matching_dirs = [d for d in existing_dirs if d.startswith(experience_name)]
    
    if not matching_dirs:
        return f"{experience_name}_{1}"

    return f"{experience_name}_{len(matching_dirs)+1}"

def load_weights_from_source(
    source: Union[Module, Dict[str, torch.Tensor]],
    target_model: Module,
    exclude_layers: List[str] = [],
    detach: bool = True
) -> None:
    """
        Load weights from a source model or dictionary into a target model, with optional exclusion of specific layers.

        This function can load weights either from a PyTorch model (`nn.Module`) or a dictionary of layer weights. 
        Users can specify layers to exclude and choose whether to detach the weights to prevent shared gradients.

        Args:
            source (Union[torch.nn.Module, Dict[str, torch.Tensor]]): The source of the weights, either a PyTorch model 
                or a dictionary containing pre-extracted weights.
            target_model (torch.nn.Module): The target model to load the weights into.
            exclude_layers (List[str], optional): A list of layer names to exclude from loading. Defaults to an empty list.
            detach (bool, optional): Whether to detach the weights to avoid sharing gradients between models. Defaults to True.

        Returns:
            None

        Raises:
            ValueError: If the `source` is not an instance of `torch.nn.Module` or a dictionary of weights.
    """
    if isinstance(source, torch.nn.Module):
        source_state_dict = source.state_dict()
    elif isinstance(source, dict):
        source_state_dict = source
    else:
        raise ValueError("Source must be either an nn.Module or a dictionary of weights.")

    target_state_dict = target_model.state_dict()
    if detach:
        filtered_state_dict = {k: v.clone().detach() for k, v in source_state_dict.items() if k not in exclude_layers}
    else:
        filtered_state_dict = {k: v.clone() for k, v in source_state_dict.items() if k not in exclude_layers}

    for name, param in filtered_state_dict.items():
        if name in target_state_dict and target_state_dict[name].shape == param.shape:
            target_state_dict[name] = param
        else:
            print(f"Skipping layer: {name} (shape mismatch or missing in target)")
    target_model.load_state_dict(target_state_dict)


def extract_layer_weights(model, layer_name, detach=True):
    """
    Extract the weights of a specific layer from a model and return them as a dictionary.

    Args:
        model (nn.Module): The model containing the layer.
        layer_name (str): The name of the layer to extract.
        detach (bool): Whether to detach the weights from the computation graph.

    Returns:
        dict: A dictionary containing the extracted weights.
    """
    state_dict = model.state_dict()
    if detach:
        layer_weights = {k: v.clone().detach() for k, v in state_dict.items() if k.startswith(layer_name)}
    else:
        layer_weights = {k: v.clone() for k, v in state_dict.items() if k.startswith(layer_name)}

    return layer_weights
