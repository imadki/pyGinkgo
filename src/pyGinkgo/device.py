from pyGinkgo import pyGinkgoBindings as pGB

from enum import StrEnum, auto
from typing import Optional

class DeviceType(StrEnum):
    cpu = auto()
    omp = auto()
    cuda = auto()
    hip = auto()
    dpcpp = auto()
    

def device(type: str = "cpu", index: Optional[int] = None) -> pGB.Executor:
    """
    Get Ginkgo executor device.

    Parameters
    ----------
    type : str
        The type of device to set.
        Or 'cuda:<index>' or 'hip:<index>' to set a specific device index.
    index : int, optional
        The index of the device to set. Default is 0.
        It can only be specified when type doesn't include it already.
    """
    params = type.split(":")

    if len(params) + (index is not None) > 2:
        raise ValueError("Too many index parameters. Only one index is allowed.")
    
    if index is None and len(params) == 2:
        try:
            index = int(params[1])
        except ValueError:
            raise ValueError("Invalid device index. It should be an integer.")
    
    type = params[0].lower()
    if DeviceType.cpu in type:
        return pGB.ReferenceExecutor()
    elif DeviceType.omp in type:
        return pGB.OmpExecutor()
    
    # All the other types require an index
    if index is None:
        index = 0
    
    if DeviceType.cuda in type:
        return pGB.CudaExecutor(
            device_id=index,
        )
    elif DeviceType.hip in type:
        return pGB.HipExecutor(
            device_id=index,
        )
    elif DeviceType.dpcpp in type:
        return pGB.DpcppExecutor(
            device_id=index,
        )

    raise ValueError(
        f"Unknown device type: {type}. Valid types are: {', '.join(t for t in DeviceType)}."
    )