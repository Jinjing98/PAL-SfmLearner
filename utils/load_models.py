"""
Utility functions for loading and managing model weights.
"""

from typing import Dict, List, Tuple
import torch

# # Import existing utilities from depth_anything_3
# try:
#     from depth_anything_3.utils.model_loading import (
#         convert_general_state_dict,
#         load_pretrained_weights as da3_load_pretrained_weights,
#     )
#     from depth_anything_3.utils.logger import logger
# except ImportError:
# Fallback if imports fail
convert_general_state_dict = None
da3_load_pretrained_weights = None
logger = None


def get_key_prefixes(keys, max_levels=3):
    """
    Extract unique key prefixes up to max_levels deep from state dict keys.
    
    Args:
        keys: List of state dict key strings (e.g., ['backbone.pos_embed', 'head.weight'])
        max_levels: Maximum depth to show (default: 3)
    
    Returns:
        Set of unique key prefixes (e.g., {'backbone.pos_embed', 'head.weight'})
    """
    if not keys:
        return set()
    
    prefixes = set()
    for key in keys:
        parts = key.split('.')
        # Take up to max_levels parts
        prefix = '.'.join(parts[:max_levels])
        prefixes.add(prefix)
    
    return prefixes


def print_state_dict_info(info, model_name="Model", max_levels=3, verbose=False):
    """
    Print state dict loading information in a clean format.
    
    Args:
        info: NamedTuple returned by load_state_dict() with missing_keys and unexpected_keys
        model_name: Name of the model for display
        max_levels: Maximum depth of key paths to show (default: 3)
        verbose: If True, print all full keys; if False, print prefixes up to max_levels
    """
    missing = info.missing_keys if hasattr(info, 'missing_keys') else []
    unexpected = info.unexpected_keys if hasattr(info, 'unexpected_keys') else []
    
    print(f"\n{model_name} state dict info:")
    print(f"  Missing keys: {len(missing)}")
    if missing:
        if verbose:
            print(f"    Details: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        else:
            missing_prefixes = get_key_prefixes(missing, max_levels=max_levels)
            print(f"    Key prefixes (up to {max_levels} levels): {sorted(missing_prefixes)}")
    
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        if verbose:
            print(f"    Details: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        else:
            unexpected_prefixes = get_key_prefixes(unexpected, max_levels=max_levels)
            print(f"    Key prefixes (up to {max_levels} levels): {sorted(unexpected_prefixes)}")


def prepare_state_dict(state_dict, remove_prefixes=None):
    """
    Prepare state dict by removing specified prefixes from keys.
    
    This function removes prefixes that appear anywhere in the key path as complete segments.
    For example, with remove_prefixes=['model.', 'pretrained.']:
    - 'model.pretrained.pos_embed' -> 'pos_embed'
    - 'model.da3.pretrained.blocks.0' -> 'da3.blocks.0'
    - 'backbone.pretrained.pos_embed' -> 'backbone.pos_embed'
    
    Args:
        state_dict: Original state dictionary
        remove_prefixes: List of prefixes to remove from keys (e.g., ['model.', 'pretrained.'])
    
    Returns:
        Modified state dictionary with cleaned keys
    """
    if remove_prefixes is None or not remove_prefixes:
        return state_dict
    
    cleaned_dict = {}
    for key, value in state_dict.items():
        # Split key into path segments
        parts = key.split('.')
        
        # Remove segments that match any prefix (without the trailing dot)
        # e.g., 'pretrained.' matches segment 'pretrained'
        filtered_parts = []
        for part in parts:
            # Check if this part + '.' matches any prefix
            segment_with_dot = part + '.'
            should_remove = False
            for prefix in remove_prefixes:
                # Check if prefix matches this segment (with or without trailing dot)
                if prefix == segment_with_dot or prefix.rstrip('.') == part:
                    should_remove = True
                    break
            if not should_remove:
                filtered_parts.append(part)
        
        # Reconstruct the key
        new_key = '.'.join(filtered_parts)
        cleaned_dict[new_key] = value
    
    return cleaned_dict


def filter_state_dict(state_dict, disable_modules=None):
    """
    Filter state dict by excluding keys that start with specified module names.
    
    Args:
        state_dict: Original state dictionary
        disable_modules: List of module names to exclude (e.g., ['head', 'cam_dec'])
                        Keys starting with any of these will be filtered out
    
    Returns:
        Filtered state dictionary
    """
    if disable_modules is None or not disable_modules:
        return state_dict
    
    # Normalize module names (ensure they end with '.' for prefix matching)
    normalized_modules = []
    for module in disable_modules:
        if not module.endswith('.'):
            normalized_modules.append(module + '.')
        else:
            normalized_modules.append(module)
    
    filtered_dict = {}
    excluded_count = 0
    for key, value in state_dict.items():
        should_exclude = False
        for module_prefix in normalized_modules:
            if key.startswith(module_prefix):
                should_exclude = True
                excluded_count += 1
                break
        if not should_exclude:
            filtered_dict[key] = value
    
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} keys from pretrained weights based on disabled modules: {disable_modules}")
    
    return filtered_dict


def load_pretrained_weights(model, pretrained_model, model_name="Model", 
                           remove_prefixes=None, disable_modules=None, strict=False, 
                           max_levels=3, verbose=False):
    """
    Load pretrained weights into a model with clean reporting.
    
    Args:
        model: Target model to load weights into
        pretrained_model: Source model with pretrained weights
        model_name: Name for display purposes
        remove_prefixes: List of prefixes to remove from state dict keys
        disable_modules: List of module names to exclude from loading (e.g., ['head', 'cam_dec'])
        strict: Whether to use strict loading
        max_levels: Maximum depth of key paths to show in info
        verbose: Whether to show full key details
    
    Returns:
        Loading info (NamedTuple with missing_keys and unexpected_keys)
    """
    pretrained_state_dict = pretrained_model.state_dict()
    
  
    if remove_prefixes:
        pretrained_state_dict = prepare_state_dict(pretrained_state_dict, remove_prefixes)

    if disable_modules:
        pretrained_state_dict = filter_state_dict(pretrained_state_dict, disable_modules)
      
    info = model.load_state_dict(pretrained_state_dict, strict=strict)
    print_state_dict_info(info, model_name=model_name, max_levels=max_levels, verbose=verbose)
    
    return info

