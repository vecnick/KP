import rasterio

from focal.FocalOperations import FocalOperations
from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
import matplotlib.pyplot as plt

# Пример использования
if __name__ == "__main__":
    # Инициализация конвертера
    converter = GeoTIFFToDaskConverter("input.tif", chunk_size=(256, 256))
    print("Исходные чанки:", converter.dask_array.chunks)
    print("Количество чанков:", converter.dask_array.chunksize)

    block = converter.dask_array.blocks[0, 10, 15].compute()
    plt.imshow(block[0], cmap='gray')
    plt.title("необработанный блок")
    plt.show()

    # Применение фокальной операции
    focal_processor = FocalOperations(converter.dask_array)
    result = focal_processor.focal_mean(window_size=5)

    # Проверка параметров результата
    print("Чанки результата:", result.chunks)
    print("Количество чанков:", converter.dask_array.chunksize)
    print("Общий размер:", result.shape)

    # Визуализация случайного блока
    block = result.blocks[0, 10, 15].compute()
    plt.imshow(block[0], cmap='gray')
    plt.title("Обработанный блок (среднее 5x5)")
    plt.show()

    # Сохранение результата
    with rasterio.open("output.TIF", "w", **converter.dataset.profile) as dst:
        dst.write(result.compute())


    converter.close()