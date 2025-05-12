import numpy as np

from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
from raster.Rasterizer import Rasterizer
from zonal.ZonalOperations import ZonalOperations

# Пример использования
if __name__ == "__main__":
    # Инициализация конвертера растра
    converter = GeoTIFFToDaskConverter("../input.tif", chunk_size=(256, 256))

    # Создание растеризатора
    rasterizer = Rasterizer(converter)

    # Растеризация векторного файла с атрибутом 'class_id'
    vector_mask = rasterizer.from_vector(
        vector_path="regions.shp",
        attribute="class_id",
        fill=0,
        dtype=np.uint8
    )

    # Растеризация GeoJSON (пример)
    geojson_data = {
        "type": "FeatureCollection",
        "features": [...]
    }
    json_mask = rasterizer.from_geojson(geojson_data, attribute="value")

    # Использование в зональных операциях
    zonal = ZonalOperations(converter.dask_array)
    stats = zonal.zonal_mean(vector_mask)

    converter.close()