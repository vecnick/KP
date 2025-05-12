import os
import geopandas as gpd
import numpy as np
import dask.array as da
from pathlib import Path
from rasterio import features
from shapely.geometry import shape

class Rasterizer:
    def __init__(self, raster_converter):
        self.raster_converter = raster_converter
        self.transform = raster_converter.dataset.transform
        self.crs = raster_converter.dataset.crs
        self.shape = raster_converter.dataset.shape
        self.dtype = raster_converter.dataset.dtypes[0]

    def _validate_vector_file(self, vector_path: str):
        """Проверяет существование файла и его корректность."""
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Файл '{vector_path}' не найден.")

        base_path = Path(vector_path).with_suffix('')
        required_exts = ['.shp', '.shx', '.dbf']
        for ext in required_exts:
            if not os.path.exists(f"{base_path}{ext}"):
                raise FileNotFoundError(f"Необходимый файл '{base_path}{ext}' отсутствует.")

    def from_vector(
            self,
            vector_path: str,
            attribute: str = None,
            fill: float = 0,
            dtype: np.dtype = None
    ) -> da.Array:
        # Проверка файла
        self._validate_vector_file(vector_path)

        try:
            # Чтение данных
            gdf = gpd.read_file(vector_path)
            gdf = gdf.to_crs(self.crs)

            # Подготовка геометрий
            if attribute:
                shapes = [(geom, row[attribute]) for _, (geom, row) in zip(gdf.geometry, gdf.iterrows())]
            else:
                shapes = [(geom, 1) for geom in gdf.geometry]

            # Растеризация
            raster = features.rasterize(
                shapes,
                out_shape=self.shape,
                transform=self.transform,
                fill=fill,
                dtype=dtype or self.dtype
            )

            return da.from_array(raster, chunks=self.raster_converter.dask_array.chunks[-2:])

        except Exception as e:
            raise RuntimeError(f"Ошибка растеризации: {str(e)}")

    def to_vector(
            self,
            dask_array: da.Array,
            output_path: str,
            target_value: float = 1,
            min_area: float = 0,
            simplify_tolerance: float = None,
            driver: str = "ESRI Shapefile"
    ) -> gpd.GeoDataFrame:
        """
        Конвертация растровых данных в векторный формат

        :param dask_array: Входной Dask-массив
        :param output_path: Путь для сохранения векторного файла
        :param target_value: Значение пикселей для векторизации
        :param min_area: Минимальная площадь объектов (в единицах CRS)
        :param simplify_tolerance: Допуск для упрощения геометрий
        :param driver: Формат выходных данных (Shapefile, GeoJSON и т.д.)
        :return: GeoDataFrame с векторными объектами
        """
        # Конвертация Dask-массива в NumPy
        raster = dask_array.compute()

        # Генерация геометрий
        geometries = []
        values = []
        for geom, val in features.shapes(
                raster.astype(np.float32),
                transform=self.transform
        ):
            if val == target_value:
                shp = shape(geom)
                if simplify_tolerance:
                    shp = shp.simplify(simplify_tolerance)
                if shp.area >= min_area:
                    geometries.append(shp)
                    values.append(val)

        # Создание GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'value': values},
            geometry=geometries,
            crs=self.crs
        )

        # Сохранение в файл
        if driver == "GeoJSON":
            output_path = os.path.splitext(output_path)[0] + ".geojson"
        gdf.to_file(output_path, driver=driver)

        return gdf