import rasterio
import time

from focal.FocalOperations import FocalOperations
from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
import matplotlib.pyplot as plt

# Пример использования
if __name__ == "__main__":
    # Инициализация конвертера
    converter = GeoTIFFToDaskConverter("../input.tif", chunk_size=(256, 256))
    print("Исходные чанки:", converter.dask_array.chunks)
    print("Количество чанков:", converter.dask_array.chunksize)

    # block = converter.dask_array.blocks[0, 10, 15].compute()
    block = converter.dask_array.compute()
    plt.imshow(block[0], cmap='gray')
    plt.title("необработанный блок")
    plt.show()

    window_size = 15;
    # Применение фокальной операции
    focal_processor = FocalOperations(converter.dask_array)
    result = focal_processor.focal_sum(window_size=window_size)

    # Проверка параметров результата
    print("Чанки результата:", result.chunks)
    print("Количество чанков:", converter.dask_array.chunksize)
    print("Общий размер:", result.shape)

    # Визуализация случайного блока
    start_total = time.time()
    # block = result.blocks[0, 10, 15].compute()
    block = result.compute()
    print("Время выполнения операции", start_total - time.time())
    plt.imshow(block[0], cmap='gray')
    plt.title("Обработанный блок (среднее 15x15)")
    plt.show()

    # Сохранение результата
    with rasterio.open("output.TIF", "w", **converter.dataset.profile) as dst:
        dst.write(result.compute())

    converter.close()