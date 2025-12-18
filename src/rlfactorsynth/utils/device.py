"""Device management utilities."""

import torch
from typing import Union


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        torch.device
    """
    if isinstance(device, torch.device):
        return device
    
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return torch.device(device)


def get_device_info(device: torch.device) -> dict:
    """Get information about the device."""
    info = {
        "type": device.type,
        "index": device.index,
    }
    
    if device.type == "cuda":
        if torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(device)
            info["memory_total"] = torch.cuda.get_device_properties(device).total_memory
            info["memory_allocated"] = torch.cuda.memory_allocated(device)
            info["memory_reserved"] = torch.cuda.memory_reserved(device)
    
    return info


def print_device_info(device: torch.device) -> None:
    """Print device information."""
    info = get_device_info(device)
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"  Name: {info.get('name', 'N/A')}")
        total_gb = info.get('memory_total', 0) / 1e9
        allocated_gb = info.get('memory_allocated', 0) / 1e9
        print(f"  Memory: {allocated_gb:.2f} / {total_gb:.2f} GB")
