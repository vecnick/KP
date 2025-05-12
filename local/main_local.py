import rasterio

from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
import matplotlib.pyplot as plt

from local.LocalOperations import LocalOperations

# Пример использования
if __name__ == "__main__":
    # Создание конвертера
    converter = GeoTIFFToDaskConverter("../input.tif", chunk_size=(256, 256))

    # Инициализация операций
    ops = LocalOperations(converter.dask_array)

    # Пример 1: Расчет NDVI
    # nir = ops.select_band(3)  # NIR канал
    # red = ops.select_band(2)  # Red канал
    # ndvi = (nir.subtract(red)).divide(nir.add(red).add(1e-10))  # (NIR - Red) / (NIR + Red)

    # Пример 2: Условная операция
    flood_mask = ops.conditional(
        condition=ops.less_than(50),
        true_value=1,  # Затопленные области
        false_value=0  # Сухие области
    )

    # Пример 3: Комбинирование операций
    result = (
        ops.multiply(2)
        .add(10)
        .sqrt()
        # .greater_than(50)
    )

    # Визуализация результата
    plt.imshow(result.compute()[0], cmap='gray')
    plt.show()

    # Сохранение результата
    with rasterio.open("output.tif", "w", **converter.dataset.profile) as dst:
        dst.write(result.compute())
    converter.close()