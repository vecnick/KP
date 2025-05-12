from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
from GlobalOperations import GlobalOperations

# Пример использования
if __name__ == "__main__":
    # Инициализация конвертера
    converter = GeoTIFFToDaskConverter("../input.tif", chunk_size=(256, 256))

    # Создание объекта глобальных операций
    global_ops = GlobalOperations(converter.dask_array)

    # Базовые операции
    total_sum = global_ops.sum()  # Сумма всех пикселей
    channel_means = global_ops.mean(axis=(1, 2))  # Среднее по каналам

    # Статистика
    global_min = global_ops.min()  # Минимум всего изображения
    global_std = global_ops.std()  # Стандартное отклонение

    # Специальные операции

    # 95-й процентиль по всему изображению
    top_5_percent = global_ops.percentile(95).compute()

    print("95-й процентиль (глобальный):", top_5_percent)

    pixel_count = global_ops.count()  # Количество валидных пикселей

    print(f"Общая сумма: {total_sum.compute()}")
    print(f"Среднее по каналам: {channel_means.compute()}")
    print(f"Глобальный минимум: {global_min.compute()}")

    converter.close()