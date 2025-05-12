from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Инициализация
    converter = GeoTIFFToDaskConverter("input.tif", chunk_size=(8, 8))

    # Проверка параметров
    print("Общий размер:", converter.dask_array.shape)
    print("Количество чанков:", converter.dask_array.numblocks)
    print("Размер чанков:", converter.dask_array.chunks)

    # Доступ к центральному блоку (пример для 7781x7641)
    block = converter.dask_array.blocks[0, 500, 600].compute()
    plt.imshow(block[0], cmap='gray')
    plt.title("Центральный блок")
    plt.show()

    converter.close()