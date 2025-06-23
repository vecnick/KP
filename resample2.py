import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import rioxarray
import os
from rasterio.enums import Resampling
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(f"Dashboard: {client.dashboard_link}")

    # Пути к объединенным файлам B4 и B5
    b4_merged_path = "b4/b4.tif"  # Путь к объединенному B4
    b5_merged_path = "b5/b5.tif"  # Путь к объединенному B5

    # Загрузка данных с использованием rioxarray
    logger.info("Загрузка объединенных данных B4 и B5...")
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
    sample_size = 500
    y_pos = b4_data.shape[0] // 2
    x_pos = b4_data.shape[1] // 2
    half_sample = sample_size // 2
    y_slice = slice(max(0, y_pos - half_sample), min(b4_data.shape[0], y_pos + half_sample))
    x_slice = slice(max(0, x_pos - half_sample), min(b4_data.shape[1], x_pos + half_sample))

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

    # Вместо автоматического определения порогов, используем фиксированные значения
    # на основе анализа логов предыдущего запуска

    # 1. Прямой расчет NDVI без предварительного маскирования
    logger.info("Прямой расчет NDVI...")
    ndvi_direct = (b5_resampled - b4_data) / (b5_resampled + b4_data).where((b5_resampled + b4_data) != 0)

    # 2. Создание простой маски для фильтрации невалидных значений NDVI
    logger.info("Создание маски для NDVI...")
    ndvi_mask = (ndvi_direct > -1) & (ndvi_direct < 1)  # Разумный диапазон для NDVI

    # Также создадим маску для фильтрации невалидных исходных данных
    # Используем более широкий диапазон, чтобы не отфильтровать слишком много
    data_mask = (b4_data > 0.01) & (b5_resampled > 0.01)  # Отфильтровываем только явно невалидные значения

    # Объединяем маски
    final_mask = ndvi_mask & data_mask

    # Проверка маски на образце
    mask_sample = final_mask[y_slice, x_slice].compute().values
    mask_percent = np.sum(mask_sample) / mask_sample.size * 100
    logger.info(f"Процент валидных пикселей в маске: {mask_percent:.2f}%")

    # Если маска слишком строгая, используем только базовую маску данных
    if mask_percent < 10:
        logger.warning("Маска слишком строгая. Используем только базовую маску данных.")
        final_mask = data_mask
        mask_sample = final_mask[y_slice, x_slice].compute().values
        mask_percent = np.sum(mask_sample) / mask_sample.size * 100
        logger.info(f"Новый процент валидных пикселей: {mask_percent:.2f}%")

    # Применяем маску к NDVI
    ndvi_masked = ndvi_direct.where(final_mask)

    # Также маскируем исходные данные для визуализации
    b4_masked = b4_data.where(final_mask)
    b5_masked = b5_resampled.where(final_mask)

    # Визуализация результатов
    logger.info("Подготовка визуализации...")
    plt.figure(figsize=(20, 15))

    # Отображение B4 (красный канал)
    plt.subplot(2, 3, 1)
    vmin, vmax = np.nanpercentile(b4_sample[b4_finite], [2, 98]) if np.any(b4_finite) else (0, 1)
    plt.imshow(b4_sample, cmap='Reds', vmin=vmin, vmax=vmax)
    plt.title("B4 (Red) - Образец")
    plt.colorbar()

    # Отображение B5 (NIR канал)
    plt.subplot(2, 3, 2)
    vmin, vmax = np.nanpercentile(b5_sample[b5_finite], [2, 98]) if np.any(b5_finite) else (0, 1)
    plt.imshow(b5_sample, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.title("B5 (NIR) - Образец")
    plt.colorbar()

    # Отображение маски
    plt.subplot(2, 3, 3)
    plt.imshow(mask_sample, cmap='gray')
    plt.title(f"Маска (белый = валидные данные) - {mask_percent:.1f}%")
    plt.colorbar()

    # Отображение NDVI без маски
    plt.subplot(2, 3, 4)
    ndvi_sample = ndvi_direct[y_slice, x_slice].compute().values
    ndvi_finite = np.isfinite(ndvi_sample)
    vmin, vmax = -0.2, 0.8  # Типичный диапазон для NDVI
    plt.imshow(ndvi_sample, cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.title("NDVI (без маски) - Образец")
    plt.colorbar()

    # Отображение маскированного NDVI
    plt.subplot(2, 3, 5)
    ndvi_masked_sample = ndvi_masked[y_slice, x_slice].compute().values
    plt.imshow(ndvi_masked_sample, cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.title("NDVI (с маской) - Образец")
    plt.colorbar()

    # Гистограмма NDVI
    plt.subplot(2, 3, 6)
    ndvi_values = ndvi_sample[ndvi_finite]
    if len(ndvi_values) > 0:
        plt.hist(ndvi_values, bins=50, range=(-0.5, 1.0))
        plt.title("Гистограмма NDVI")
        plt.xlabel("NDVI")
        plt.ylabel("Количество пикселей")
    else:
        plt.text(0.5, 0.5, "Нет данных для гистограммы",
                 horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig("ndvi_results.png", dpi=300)
    plt.show()

    # Визуализация отдельных масок для диагностики
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    data_mask_sample = data_mask[y_slice, x_slice].compute().values
    data_mask_percent = np.sum(data_mask_sample) / data_mask_sample.size * 100
    plt.imshow(data_mask_sample, cmap='gray')
    plt.title(f"Маска данных - {data_mask_percent:.1f}%")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    ndvi_mask_sample = ndvi_mask[y_slice, x_slice].compute().values
    ndvi_mask_percent = np.sum(ndvi_mask_sample) / ndvi_mask_sample.size * 100
    plt.imshow(ndvi_mask_sample, cmap='gray')
    plt.title(f"Маска NDVI - {ndvi_mask_percent:.1f}%")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    # Создаем маску для воды (NDVI < 0)
    water_mask = (ndvi_direct < 0)
    water_mask_sample = water_mask[y_slice, x_slice].compute().values
    water_mask_percent = np.sum(water_mask_sample) / water_mask_sample.size * 100
    plt.imshow(water_mask_sample, cmap='Blues')
    plt.title(f"Маска воды (NDVI < 0) - {water_mask_percent:.1f}%")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    # Создаем маску для растительности (NDVI > 0.3)
    veg_mask = (ndvi_direct > 0.3)
    veg_mask_sample = veg_mask[y_slice, x_slice].compute().values
    veg_mask_percent = np.sum(veg_mask_sample) / veg_mask_sample.size * 100
    plt.imshow(veg_mask_sample, cmap='Greens')
    plt.title(f"Маска растительности (NDVI > 0.3) - {veg_mask_percent:.1f}%")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("mask_diagnostics.png", dpi=300)
    plt.show()

    # Сохранение результатов
    logger.info("Сохранение результатов...")

    # Сохраняем NDVI (и с маской, и без)
    logger.info("Сохранение NDVI без маски...")
    ndvi_direct.rio.to_raster("ndvi_unmasked.tif", driver="GTiff")

    logger.info("Сохранение NDVI с маской...")
    ndvi_masked.rio.to_raster("ndvi_masked.tif", driver="GTiff")

    # Сохраняем маску
    logger.info("Сохранение маски...")
    final_mask.astype(np.uint8).rio.to_raster("mask.tif", driver="GTiff")

    # Создаем цветную карту NDVI для визуализации
    logger.info("Создание цветной карты NDVI...")


    # Функция для создания цветной карты NDVI
    def create_ndvi_colormap(ndvi_data, output_path):
        # Создаем новый датасет с 3 каналами (RGB)
        height, width = ndvi_data.shape
        rgb = np.zeros((3, height, width), dtype=np.uint8)

        # Загружаем данные NDVI
        ndvi_values = ndvi_data.values

        # Создаем цветовую карту:
        # < 0 (вода): оттенки синего
        # 0-0.2 (почва/городская застройка): оттенки коричневого/серого
        # 0.2-0.4 (редкая растительность): светло-зеленый
        # 0.4-0.6 (умеренная растительность): зеленый
        # > 0.6 (густая растительность): темно-зеленый

        # Маска для воды (синий)
        water_mask = (ndvi_values < 0) & np.isfinite(ndvi_values)
        rgb[0, water_mask] = 0  # R
        rgb[1, water_mask] = 0  # G
        rgb[2, water_mask] = 255  # B (максимальный синий)

        # Маска для почвы/городской застройки (коричневый/серый)
        urban_mask = (ndvi_values >= 0) & (ndvi_values < 0.2) & np.isfinite(ndvi_values)
        rgb[0, urban_mask] = 150  # R
        rgb[1, urban_mask] = 150  # G
        rgb[2, urban_mask] = 150  # B

        # Маска для редкой растительности (светло-зеленый)
        sparse_veg_mask = (ndvi_values >= 0.2) & (ndvi_values < 0.4) & np.isfinite(ndvi_values)
        rgb[0, sparse_veg_mask] = 180  # R
        rgb[1, sparse_veg_mask] = 230  # G
        rgb[2, sparse_veg_mask] = 180  # B

        # Маска для умеренной растительности (зеленый)
        mod_veg_mask = (ndvi_values >= 0.4) & (ndvi_values < 0.6) & np.isfinite(ndvi_values)
        rgb[0, mod_veg_mask] = 50  # R
        rgb[1, mod_veg_mask] = 200  # G
        rgb[2, mod_veg_mask] = 50  # B

        # Маска для густой растительности (темно-зеленый)
        dense_veg_mask = (ndvi_values >= 0.6) & np.isfinite(ndvi_values)
        rgb[0, dense_veg_mask] = 0  # R
        rgb[1, dense_veg_mask] = 130  # G
        rgb[2, dense_veg_mask] = 0  # B

        # Создаем датасет с RGB каналами
        rgb_data = xr.DataArray(
            rgb,
            dims=('band', 'y', 'x'),
            coords={
                'band': [1, 2, 3],
                'y': ndvi_data.y,
                'x': ndvi_data.x
            }
        )

        # Копируем метаданные
        rgb_data.rio.write_crs(ndvi_data.rio.crs, inplace=True)
        rgb_data.rio.write_transform(ndvi_data.rio.transform(), inplace=True)

        # Сохраняем как GeoTIFF
        rgb_data.rio.to_raster(output_path, driver="GTiff")

        return rgb_data


    # Создаем цветную карту для NDVI
    try:
        ndvi_rgb = create_ndvi_colormap(ndvi_masked, "ndvi_colormap.tif")
        logger.info("Цветная карта NDVI успешно создана.")
    except Exception as e:
        logger.error(f"Ошибка при создании цветной карты NDVI: {e}")

    logger.info("Обработка завершена!")

    # Завершение работы
    client.close()
    cluster.close()