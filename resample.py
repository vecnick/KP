import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import rioxarray
import glob
import os
from rasterio.enums import Resampling

if __name__ == '__main__':
    # Исправление для Windows/macOS
    from multiprocessing import set_start_method

    set_start_method('spawn')  # Или 'forkserver' для Linux

    # Локальный кластер Dask
    cluster = LocalCluster(
        n_workers=3,
        processes=False,
        memory_limit='5GB'
    )
    client = Client(cluster)
    print("Dashboard:", client.dashboard_link)

    # Пути к объединенным файлам B4 и B5
    b4_merged_path = "b4/b4.tif"  # Путь к объединенному B4
    b5_merged_path = "b5/b5.tif"  # Путь к объединенному B5

    # Загрузка данных с использованием rioxarray
    print("Загрузка объединенных данных B4 и B5...")
    b4_data = rioxarray.open_rasterio(b4_merged_path, chunks={"x": 1024, "y": 1024})
    b5_data = rioxarray.open_rasterio(b5_merged_path, chunks={"x": 1024, "y": 1024})

    # Удаление лишних размерностей
    b4_data = b4_data.squeeze()
    b5_data = b5_data.squeeze()

    print(f"B4 данные: Форма: {b4_data.shape}, Размерности: {b4_data.dims}, Разрешение: {b4_data.rio.resolution()}")
    print(f"B5 данные: Форма: {b5_data.shape}, Размерности: {b5_data.dims}, Разрешение: {b5_data.rio.resolution()}")

    # Проверка, нужен ли ресемплинг
    if b4_data.shape != b5_data.shape or b4_data.rio.resolution() != b5_data.rio.resolution():
        print("Обнаружены различия в размере или разрешении. Выполняем ресемплинг...")

        # Определяем, какое разрешение использовать (можно выбрать любое)
        target_resolution = b4_data.rio.resolution()  # Используем разрешение B4
        print(f"Целевое разрешение для ресемплинга: {target_resolution}")

        # Ресемплинг B5 к разрешению B4
        b5_resampled = b5_data.rio.reproject(
            b4_data.rio.crs,
            shape=b4_data.shape,
            transform=b4_data.rio.transform(),
            resampling=Resampling.bilinear
        )
        print(f"B5 после ресемплинга: Форма: {b5_resampled.shape}, Разрешение: {b5_resampled.rio.resolution()}")
    else:
        print("Ресемплинг не требуется, данные имеют одинаковый размер и разрешение.")
        b5_resampled = b5_data

    # Создание маски облаков на основе значений
    print("Создание маски облаков и теней...")

    # Вычисляем маску на основе значений
    # 1. Маска для невалидных данных (обычно 0 или очень высокие значения)
    # Для Landsat 8 SR, валидные значения обычно в диапазоне 0-10000
    valid_min = 1  # Минимальное валидное значение
    valid_max = 10000  # Максимальное валидное значение

    # Создаем маску, где True = валидные данные
    b4_valid_mask = (b4_data > valid_min) & (b4_data < valid_max)
    b5_valid_mask = (b5_resampled > valid_min) & (b5_resampled < valid_max)

    # Объединяем маски - данные валидны, если оба канала валидны
    valid_mask = b4_valid_mask & b5_valid_mask

    # 2. Маска облаков на основе отношения B4/B5
    # Для Landsat 8, облака обычно имеют высокие значения в обоих каналах,
    # но отношение B4/B5 для облаков обычно ниже, чем для земли

    # Вычисляем отношение B4/B5 (избегаем деления на ноль)
    ratio = b4_data / b5_resampled.where(b5_resampled > 0)

    # Маска облаков: True = не облака
    # Настройте пороги в зависимости от ваших данных
    cloud_threshold_low = 0.4  # Типичное отношение для облаков
    cloud_threshold_high = 1.0  # Типичное отношение для земли
    cloud_mask = (ratio > cloud_threshold_low) & (ratio < cloud_threshold_high)

    # 3. Маска теней на основе низких значений в B4
    # Тени обычно имеют низкие значения в обоих каналах
    shadow_threshold = 300  # Настройте в зависимости от ваших данных
    shadow_mask = (b4_data > shadow_threshold)

    # Объединяем все маски
    final_mask = valid_mask & cloud_mask & shadow_mask

    # Применяем маску к данным
    b4_masked = b4_data.where(final_mask)
    b5_masked = b5_resampled.where(final_mask)

    # Вычисляем NDVI с использованием маскированных данных
    # NDVI = (NIR - Red) / (NIR + Red)
    # Для Landsat 8, B5 = NIR, B4 = Red
    ndvi = (b5_masked - b4_masked) / (b5_masked + b4_masked)

    # Визуализация результатов
    plt.figure(figsize=(20, 15))

    # Из-за большого размера данных (18646, 24309), мы будем использовать
    # уменьшенную версию для визуализации
    print("Подготовка данных для визуализации...")


    # Функция для безопасного вычисления персентилей
    def safe_percentile(data, q):
        # Преобразуем в numpy массив и вычисляем персентили
        data_np = data.compute().values
        # Используем только конечные значения для вычисления персентилей
        finite_mask = np.isfinite(data_np)
        if np.any(finite_mask):
            return np.percentile(data_np[finite_mask], q)
        else:
            return [0, 1] if isinstance(q, list) else 0


    # Уменьшаем размер данных для визуализации
    # Выбираем каждый 10-й пиксель по обеим осям
    step = 10

    # Отображение B4 (красный канал)
    plt.subplot(2, 3, 1)
    b4_plot = b4_data[::step, ::step].compute()
    vmin, vmax = safe_percentile(b4_data, [2, 98])
    plt.imshow(b4_plot, cmap='Reds', vmin=vmin, vmax=vmax)
    plt.title("B4 (Red)")
    plt.colorbar()

    # Отображение B5 (NIR канал)
    plt.subplot(2, 3, 2)
    b5_plot = b5_resampled[::step, ::step].compute()
    vmin, vmax = safe_percentile(b5_resampled, [2, 98])
    plt.imshow(b5_plot, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.title("B5 (NIR)")
    plt.colorbar()

    # Отображение маски
    plt.subplot(2, 3, 3)
    mask_plot = final_mask[::step, ::step].compute()
    plt.imshow(mask_plot, cmap='gray')
    plt.title("Маска (белый = валидные данные)")
    plt.colorbar()

    # Отображение маскированного B4
    plt.subplot(2, 3, 4)
    b4_masked_plot = b4_masked[::step, ::step].compute()
    vmin, vmax = safe_percentile(b4_masked, [2, 98])
    plt.imshow(b4_masked_plot, cmap='Reds', vmin=vmin, vmax=vmax)
    plt.title("Маскированный B4")
    plt.colorbar()

    # Отображение маскированного B5
    plt.subplot(2, 3, 5)
    b5_masked_plot = b5_masked[::step, ::step].compute()
    vmin, vmax = safe_percentile(b5_masked, [2, 98])
    plt.imshow(b5_masked_plot, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.title("Маскированный B5")
    plt.colorbar()

    # Отображение NDVI
    plt.subplot(2, 3, 6)
    ndvi_plot = ndvi[::step, ::step].compute()
    plt.imshow(ndvi_plot, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    plt.title("NDVI")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("resampled_masked_results.png", dpi=300)
    plt.show()

    # Сохранение результатов
    print("Сохранение результатов...")

    # Сохраняем маскированные данные
    print("Сохранение маскированного B4...")
    b4_masked.rio.to_raster("b4_masked.tif", driver="GTiff")

    print("Сохранение маскированного B5...")
    b5_masked.rio.to_raster("b5_masked.tif", driver="GTiff")

    # Сохраняем NDVI
    print("Сохранение NDVI...")
    ndvi.rio.to_raster("ndvi.tif", driver="GTiff")

    # Сохраняем маску
    print("Сохранение маски...")
    final_mask.astype(np.uint8).rio.to_raster("mask.tif", driver="GTiff")

    print("Обработка завершена!")

    # Завершение работы
    client.close()
    cluster.close()