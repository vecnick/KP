import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import rioxarray
import os
from rasterio.enums import Resampling
import logging
import gc  # Для принудительной сборки мусора
import dask

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Исправление для Windows/macOS
    from multiprocessing import set_start_method

    set_start_method('spawn')  # Или 'forkserver' для Linux

    # Увеличиваем лимит памяти для Dask
    dask.config.set({"distributed.worker.memory.target": 0.85})  # 85% памяти для активных вычислений
    dask.config.set({"distributed.worker.memory.spill": 0.9})  # 90% до начала сброса на диск
    dask.config.set({"distributed.worker.memory.pause": 0.95})  # 95% до приостановки обработки

    # Локальный кластер Dask с оптимизированными настройками
    cluster = LocalCluster(
        n_workers=2,  # Один рабочий процесс для лучшего управления памятью
        processes=True,
        memory_limit='6GB',  # Увеличиваем лимит памяти
        threads_per_worker=2,
        silence_logs=logging.WARNING
    )
    client = Client(cluster)
    logger.info(f"Dashboard: {client.dashboard_link}")

    # Пути к объединенным файлам B4 и B5
    b4_merged_path = "b4/b4.tif"
    b5_merged_path = "b5/b5.tif"

    # Загрузка данных с использованием rioxarray
    logger.info("Загрузка объединенных данных B4 и B5...")
    # Используем более мелкие чанки для лучшего управления памятью
    b4_data = rioxarray.open_rasterio(b4_merged_path, chunks={"x": 1024, "y": 1024})
    b5_data = rioxarray.open_rasterio(b5_merged_path, chunks={"x": 1024, "y": 1024})

    # Удаление лишних размерностей
    b4_data = b4_data.squeeze()
    b5_data = b5_data.squeeze()

    logger.info(
        f"B4 данные: Форма: {b4_data.shape}, Размерности: {b4_data.dims}, Разрешение: {b4_data.rio.resolution()}")
    logger.info(
        f"B5 данные: Форма: {b5_data.shape}, Размерности: {b5_data.dims}, Разрешение: {b5_data.rio.resolution()}")

    # Проверка, нужен ли ресемплинг
    if b4_data.shape != b5_data.shape or b4_data.rio.resolution() != b5_data.rio.resolution():
        logger.info("Обнаружены различия в размере или разрешении. Выполняем ресемплинг...")
        # Ресемплинг B5 к разрешению B4
        b5_resampled = b5_data.rio.reproject(
            b4_data.rio.crs,
            shape=b4_data.shape,
            transform=b4_data.rio.transform(),
            resampling=Resampling.bilinear
        )
    else:
        logger.info("Ресемплинг не требуется, данные имеют одинаковый размер и разрешение.")
        b5_resampled = b5_data

    # Анализ образца данных для понимания диапазона значений
    logger.info("Анализ образца данных...")
    # Берем образец из центра изображения
    sample_size = 500
    y_center = b4_data.shape[0] // 2
    x_center = b4_data.shape[1] // 2
    half_sample = sample_size // 2

    y_slice = slice(max(0, y_center - half_sample), min(b4_data.shape[0], y_center + half_sample))
    x_slice = slice(max(0, x_center - half_sample), min(b4_data.shape[1], x_center + half_sample))

    b4_sample = b4_data[y_slice, x_slice].compute().values
    b5_sample = b5_resampled[y_slice, x_slice].compute().values

    # Проверка наличия данных
    b4_finite = np.isfinite(b4_sample)
    b5_finite = np.isfinite(b5_sample)

    logger.info(f"Процент конечных значений в образце B4: {np.sum(b4_finite) / b4_finite.size * 100:.2f}%")
    logger.info(f"Процент конечных значений в образце B5: {np.sum(b5_finite) / b5_finite.size * 100:.2f}%")

    if np.sum(b4_finite) > 0 and np.sum(b5_finite) > 0:
        b4_min, b4_max = np.nanmin(b4_sample), np.nanmax(b4_sample)
        b5_min, b5_max = np.nanmin(b5_sample), np.nanmax(b5_sample)

        logger.info(f"B4 диапазон значений: [{b4_min:.4f}, {b4_max:.4f}]")
        logger.info(f"B5 диапазон значений: [{b5_min:.4f}, {b5_max:.4f}]")

    # Прямой расчет NDVI для всей сцены
    logger.info("Расчет NDVI для всей сцены...")

    # Создаем маску для избежания деления на ноль
    sum_nonzero = (b5_resampled + b4_data) != 0

    # Вычисляем NDVI с маской для деления на ноль
    ndvi = xr.where(
        sum_nonzero,
        (b5_resampled - b4_data) / (b5_resampled + b4_data),
        np.nan
    )

    # Создание простой маски для фильтрации невалидных значений NDVI
    logger.info("Создание маски для NDVI...")

    # Фильтруем только явно невалидные значения
    ndvi_mask = (ndvi >= -1) & (ndvi <= 1)  # Теоретически возможный диапазон NDVI

    # Также создаем маску для фильтрации невалидных исходных данных
    data_mask = (b4_data > 0.01) & (b5_resampled > 0.01)  # Отфильтровываем только явно невалидные значения

    # Объединяем маски
    final_mask = ndvi_mask & data_mask

    # Проверка маски на образце
    mask_sample = final_mask[y_slice, x_slice].compute().values
    mask_percent = np.sum(mask_sample) / mask_sample.size * 100
    logger.info(f"Процент валидных пикселей в маске (образец): {mask_percent:.2f}%")

    # Если маска слишком строгая, используем только базовую маску данных
    if mask_percent < 10:
        logger.warning("Маска слишком строгая. Используем только базовую маску данных.")
        final_mask = data_mask
        mask_sample = final_mask[y_slice, x_slice].compute().values
        mask_percent = np.sum(mask_sample) / mask_sample.size * 100
        logger.info(f"Новый процент валидных пикселей (образец): {mask_percent:.2f}%")

    # Применяем маску к NDVI
    logger.info("Применение маски к NDVI...")
    ndvi_masked = ndvi.where(final_mask)

    # Сохранение NDVI для всей сцены
    logger.info("Сохранение NDVI для всей сцены...")
    ndvi_masked.rio.to_raster("ndvi_full_scene.tif", driver="GTiff")

    # Освобождаем память
    del ndvi
    gc.collect()

    # Создаем визуализацию для всей сцены
    logger.info("Создание обзорного изображения всей сцены...")

    # Для визуализации всей сцены уменьшим разрешение
    # Определяем коэффициент уменьшения, чтобы получить изображение разумного размера
    target_width = 1200  # Целевая ширина для визуализации
    scale_factor = max(1, int(b4_data.shape[1] / target_width))

    logger.info(f"Уменьшение разрешения для визуализации (коэффициент: {scale_factor})...")

    try:
        # Создаем уменьшенную версию NDVI для визуализации
        # Используем более эффективный подход с прореживанием
        ndvi_small = ndvi_masked[::scale_factor, ::scale_factor]

        # Вычисляем уменьшенную версию
        logger.info("Вычисление уменьшенной версии NDVI для визуализации...")
        ndvi_downsampled = ndvi_small.compute()

        # Создаем цветовую карту для NDVI
        from matplotlib.colors import LinearSegmentedColormap

        # Создаем кастомную цветовую карту для NDVI
        colors = [
            (0.0, 0.0, 0.5),  # Темно-синий (вода, NDVI < 0)
            (0.0, 0.0, 1.0),  # Синий (вода, NDVI около 0)
            (0.5, 0.5, 0.5),  # Серый (голая почва, NDVI около 0)
            (1.0, 1.0, 0.0),  # Желтый (редкая растительность)
            (0.0, 1.0, 0.0),  # Зеленый (умеренная растительность)
            (0.0, 0.5, 0.0)  # Темно-зеленый (густая растительность)
        ]

        ndvi_cmap = LinearSegmentedColormap.from_list('ndvi_cmap', colors, N=256)

        # Создаем и сохраняем визуализацию
        plt.figure(figsize=(15, 15))
        plt.imshow(ndvi_downsampled, cmap=ndvi_cmap, vmin=-0.2, vmax=0.8)
        plt.colorbar(label='NDVI')
        plt.title('NDVI для всей сцены')
        plt.axis('off')  # Убираем оси

        # Сохраняем изображение в высоком разрешении
        plt.savefig("ndvi_full_scene_preview.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Освобождаем память
        del ndvi_downsampled, ndvi_small
        gc.collect()

        logger.info("Обзорное изображение всей сцены успешно создано.")
    except Exception as e:
        logger.error(f"Ошибка при создании обзорного изображения: {e}")

    # Создаем цветную карту NDVI для всей сцены
    logger.info("Создание цветной карты NDVI для всей сцены...")

    try:
        # Создаем цветную карту для всей сцены, обрабатывая данные по частям
        # Разбиваем сцену на блоки и обрабатываем их по отдельности

        # Определяем размер блока
        block_size = 2000

        # Получаем размеры всей сцены
        height, width = ndvi_masked.shape

        # Создаем выходной файл для цветной карты
        import rasterio
        from rasterio.transform import Affine

        # Получаем трансформацию и CRS из исходных данных
        transform = ndvi_masked.rio.transform()
        crs = ndvi_masked.rio.crs

        # Создаем профиль для выходного файла
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 3,  # 3 канала (RGB)
            'dtype': 'uint8',
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',  # Используем сжатие LZW
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        }

        # Создаем выходной файл
        output_path = "ndvi_colormap_full.tif"

        with rasterio.open(output_path, 'w', **profile) as dst:
            # Обрабатываем данные блоками
            for y_start in range(0, height, block_size):
                y_end = min(y_start + block_size, height)

                for x_start in range(0, width, block_size):
                    x_end = min(x_start + block_size, width)

                    logger.info(f"Обработка блока: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")

                    # Определяем срез для текущего блока
                    y_slice_block = slice(y_start, y_end)
                    x_slice_block = slice(x_start, x_end)

                    # Загружаем блок NDVI
                    ndvi_block = ndvi_masked[y_slice_block, x_slice_block].compute().values

                    # Создаем RGB-представление для блока
                    rgb_block = np.zeros((3, y_end - y_start, x_end - x_start), dtype=np.uint8)

                    # Заполняем RGB значения на основе NDVI

                    # Маска для воды (синий)
                    water_mask = (ndvi_block < 0) & np.isfinite(ndvi_block)
                    rgb_block[0, water_mask] = 0  # R
                    rgb_block[1, water_mask] = 0  # G
                    rgb_block[2, water_mask] = 255  # B

                    # Маска для почвы/городской застройки (коричневый/серый)
                    urban_mask = (ndvi_block >= 0) & (ndvi_block < 0.2) & np.isfinite(ndvi_block)
                    rgb_block[0, urban_mask] = 150  # R
                    rgb_block[1, urban_mask] = 150  # G
                    rgb_block[2, urban_mask] = 150  # B

                    # Маска для редкой растительности (светло-зеленый)
                    sparse_veg_mask = (ndvi_block >= 0.2) & (ndvi_block < 0.4) & np.isfinite(ndvi_block)
                    rgb_block[0, sparse_veg_mask] = 180  # R
                    rgb_block[1, sparse_veg_mask] = 230  # G
                    rgb_block[2, sparse_veg_mask] = 180  # B

                    # Маска для умеренной растительности (зеленый)
                    mod_veg_mask = (ndvi_block >= 0.4) & (ndvi_block < 0.6) & np.isfinite(ndvi_block)
                    rgb_block[0, mod_veg_mask] = 50  # R
                    rgb_block[1, mod_veg_mask] = 200  # G
                    rgb_block[2, mod_veg_mask] = 50  # B

                    # Маска для густой растительности (темно-зеленый)
                    dense_veg_mask = (ndvi_block >= 0.6) & np.isfinite(ndvi_block)
                    rgb_block[0, dense_veg_mask] = 0  # R
                    rgb_block[1, dense_veg_mask] = 130  # G
                    rgb_block[2, dense_veg_mask] = 0  # B

                    # Записываем блок в выходной файл
                    dst.write(rgb_block, window=((y_start, y_end), (x_start, x_end)))

                    # Освобождаем память
                    del ndvi_block, rgb_block, water_mask, urban_mask, sparse_veg_mask, mod_veg_mask, dense_veg_mask
                    gc.collect()

        logger.info("Цветная карта NDVI для всей сцены успешно создана.")
    except Exception as e:
        logger.error(f"Ошибка при создании цветной карты NDVI: {e}")

    # Анализ распределения значений NDVI
    logger.info("Анализ распределения значений NDVI...")

    # Берем несколько образцов из разных частей изображения
    ndvi_values = []
    for i in range(3):
        for j in range(3):
            y_pos = int(ndvi_masked.shape[0] * (i + 1) / 4)
            x_pos = int(ndvi_masked.shape[1] * (j + 1) / 4)

            # Берем образец 500x500 пикселей
            half_sample = 250
            y_slice = slice(max(0, y_pos - half_sample), min(ndvi_masked.shape[0], y_pos + half_sample))
            x_slice = slice(max(0, x_pos - half_sample), min(ndvi_masked.shape[1], x_pos + half_sample))

            # Загружаем образец NDVI
            ndvi_sample = ndvi_masked[y_slice, x_slice].compute().values
            ndvi_sample_flat = ndvi_sample[np.isfinite(ndvi_sample)].flatten()
            ndvi_values.extend(ndvi_sample_flat)

    # Преобразуем в numpy массив
    ndvi_values = np.array(ndvi_values)

    # Вычисляем статистику
    ndvi_min = np.min(ndvi_values)
    ndvi_max = np.max(ndvi_values)
    ndvi_mean = np.mean(ndvi_values)
    ndvi_median = np.median(ndvi_values)
    ndvi_std = np.std(ndvi_values)

    # Вычисляем процентили
    ndvi_percentiles = np.percentile(ndvi_values, [1, 5, 10, 25, 50, 75, 90, 95, 99])

    logger.info(f"Статистика NDVI:")
    logger.info(f"  Мин: {ndvi_min:.4f}, Макс: {ndvi_max:.4f}")
    logger.info(f"  Среднее: {ndvi_mean:.4f}, Медиана: {ndvi_median:.4f}, Стд. откл.: {ndvi_std:.4f}")
    logger.info(f"  Процентили:")
    logger.info(f"    1%: {ndvi_percentiles[0]:.4f}, 5%: {ndvi_percentiles[1]:.4f}, 10%: {ndvi_percentiles[2]:.4f}")
    logger.info(f"    25%: {ndvi_percentiles[3]:.4f}, 50%: {ndvi_percentiles[4]:.4f}, 75%: {ndvi_percentiles[5]:.4f}")
    logger.info(f"    90%: {ndvi_percentiles[6]:.4f}, 95%: {ndvi_percentiles[7]:.4f}, 99%: {ndvi_percentiles[8]:.4f}")

    # Вычисляем процент пикселей в каждом диапазоне NDVI
    water_percent = np.sum(ndvi_values < 0) / len(ndvi_values) * 100
    urban_percent = np.sum((ndvi_values >= 0) & (ndvi_values < 0.2)) / len(ndvi_values) * 100
    sparse_veg_percent = np.sum((ndvi_values >= 0.2) & (ndvi_values < 0.4)) / len(ndvi_values) * 100
    mod_veg_percent = np.sum((ndvi_values >= 0.4) & (ndvi_values < 0.6)) / len(ndvi_values) * 100
    dense_veg_percent = np.sum(ndvi_values >= 0.6) / len(ndvi_values) * 100

    logger.info(f"Распределение значений NDVI:")
    logger.info(f"  Вода/снег/облака (NDVI < 0): {water_percent:.2f}%")
    logger.info(f"  Городские территории/почва (0 <= NDVI < 0.2): {urban_percent:.2f}%")
    logger.info(f"  Редкая растительность (0.2 <= NDVI < 0.4): {sparse_veg_percent:.2f}%")
    logger.info(f"  Умеренная растительность (0.4 <= NDVI < 0.6): {mod_veg_percent:.2f}%")
    logger.info(f"  Густая растительность (NDVI >= 0.6): {dense_veg_percent:.2f}%")

    # Создаем гистограмму распределения NDVI
    plt.figure(figsize=(12, 8))
    plt.hist(ndvi_values, bins=100, range=(-0.6, 0.6), alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--', label='NDVI = 0')
    plt.axvline(x=0.2, color='orange', linestyle='--', label='NDVI = 0.2')
    plt.axvline(x=0.4, color='g', linestyle='--', label='NDVI = 0.4')
    plt.axvline(x=0.6, color='darkgreen', linestyle='--', label='NDVI = 0.6')
    plt.xlabel('NDVI')
    plt.ylabel('Количество пикселей')
    plt.title('Распределение значений NDVI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ndvi_histogram.png", dpi=300)
    plt.close()


    # Создаем визуализацию цветной карты (уменьшенная версия)
    logger.info("Создание превью цветной карты NDVI...")

    try:
        # Открываем созданную цветную карту
        with rasterio.open(output_path) as src:
            # Определяем коэффициент уменьшения
            scale_factor = max(1, int(src.width / target_width))

            # Читаем уменьшенную версию
            rgb_preview = src.read(
                out_shape=(3, int(src.height / scale_factor), int(src.width / scale_factor)),
                resampling=Resampling.nearest
            )

            # Переставляем оси для matplotlib (channels last)
            rgb_preview = np.transpose(rgb_preview, (1, 2, 0))

            # Создаем и сохраняем визуализацию
            plt.figure(figsize=(15, 15))
            plt.imshow(rgb_preview)
            plt.title('Цветная карта NDVI для всей сцены')
            plt.axis('off')

            # Сохраняем изображение
            plt.savefig("ndvi_colormap_preview.png", dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Превью цветной карты NDVI успешно создано.")
    except Exception as e:
        logger.error(f"Ошибка при создании превью цветной карты: {e}")

    logger.info("Обработка завершена!")

    # Завершение работы
    client.close()
    cluster.close()