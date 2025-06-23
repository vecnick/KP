from rasterio._warp import Resampling

from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
import matplotlib.pyplot as plt

from resampling.Resampler import Resampler

# Пример использования
if __name__ == "__main__":
    # Инициализация
    src_converter = GeoTIFFToDaskConverter("input.tif", chunk_size=(1, 256, 256))
    resampler = Resampler(src_converter)

    # Ресемплинг с логированием
    resampled = resampler.resample(
        dst_res=(20, 20),
        resampling_method=Resampling.average,
        dst_chunks = (1, 256, 256)
    )

    # Регриддинг с явными чанками
    # target_converter = GeoTIFFToDaskConverter("target.tif")
    # regridded = resampler.regrid(
    #     target_converter,
    #     dst_chunks=(1, 256, 256)
    # )

    # Проверка результатов
    print("Ресемплированные чанки:", resampled.chunks)
    # print("Регриддинг чанки:", regridded.chunks)

    # Визуализация
    # plt.imshow(regridded.compute(), cmap='viridis')
    # plt.show()

    src_converter.close()
    # target_converter.close()