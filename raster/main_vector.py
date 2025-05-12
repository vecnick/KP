from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
from raster.Rasterizer import Rasterizer
import matplotlib.pyplot as plt

# Пример использования
if __name__ == "__main__":
    # Инициализация конвертера растра
    converter = GeoTIFFToDaskConverter("../input.tif", chunk_size=(256, 256))

    # Создание бинарной маски
    mask = converter.dask_array > 0.5

    # Векторизация
    rasterizer = Rasterizer(converter)
    vector_data = rasterizer.to_vector(
        dask_array=mask,
        output_path="output_polygons.shp",
        target_value=1,
        min_area=100,  # Минимальная площадь 100 м²
        simplify_tolerance=0.5  # Упрощение геометрий
    )

    # Визуализация
    print(vector_data.head())
    vector_data.plot()
    plt.show()

    converter.close()