from dataclasses import dataclass
from typing import Tuple, Optional
import xarray as xr

@dataclass
class Chunk:
    id: str
    coords: Tuple[int, int, int, int]  # y_start, y_end, x_start, x_end
    data: Optional[xr.DataArray] = None
    status: str = "pending"
    worker: Optional[str] = None