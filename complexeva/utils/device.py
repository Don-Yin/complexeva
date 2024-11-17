import torch


def select_device(priority=["cuda", "cpu"]):  # mps in the middle
    """
    Selects the device based on the given priority list.
    If top priority device is not available, it will try the next device in the list.
    E.g., it will try to select "cuda" first, then "mps", and finally "cpu"
    So we can have the same code on local machine and on the server.
    Just with

    - local: select_device(["cuda", "cpu"])
    - server: select_device(["cuda", "cpu"])

    Parameters:
        - priority (list): List of strings representing device priorities.

    Returns:
        - torch.device: Device selected based on priority.
    """

    if "cuda" in priority and torch.cuda.is_available():
        return torch.device("cuda")
    if "mps" in priority and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if "cpu" in priority:
        return torch.device("cpu")

    raise ValueError("No valid device found in priority list.")
