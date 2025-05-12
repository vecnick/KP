import dask.array as da
import numpy as np
import geopandas as gpd
from rasterio import features

class ZonalOperations:
    def __init__(self, dask_array, transform=None, crs=None):
        self.dask_array = dask_array
        self.transform = transform
        self.crs = crs

    def from_mask(self, mask):
        """Создает маскированные данные с использованием Dask"""
        # Заменяем значения вне маски на NaN
        masked_data = da.where(mask, self.dask_array, np.nan)
        return ZonalOperations(masked_data, self.transform, self.crs)

    def from_vector(self, vector_file, attribute=None):
        """Создает маску зон из векторного файла"""
        gdf = gpd.read_file(vector_file)
        shapes = [(geom, 1) for geom in gdf.geometry]

        mask = features.rasterize(
            shapes,
            out_shape=self.dask_array.shape[-2:],
            transform=self.transform,
            fill=0,
            dtype=np.uint8
        )

        mask_dask = da.from_array(mask, chunks=self.dask_array.chunks[-2:])
        return mask_dask

    def apply_zonal(self, func, zone_mask=None):
        """Применяет агрегационную функцию к зоне"""
        if zone_mask is not None:
            masked_data = da.where(zone_mask, self.dask_array, np.nan)
        else:
            masked_data = self.dask_array

        return func(masked_data, axis=(-2, -1))  # Агрегация по высоте и ширине

    # Стандартные зональные операции
    def zonal_sum(self, zone_mask=None):
        return self.apply_zonal(da.nansum, zone_mask)

    def zonal_mean(self, zone_mask=None):
        return self.apply_zonal(da.nanmean, zone_mask)

    def zonal_min(self, zone_mask=None):
        return self.apply_zonal(da.nanmin, zone_mask)

    def zonal_max(self, zone_mask=None):
        return self.apply_zonal(da.nanmax, zone_mask)

    def zonal_std(self, zone_mask=None):
        return self.apply_zonal(da.nanstd, zone_mask)

    def compute(self):
        return self.dask_array.compute()