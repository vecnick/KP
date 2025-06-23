import xarray as xr
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster

if __name__ == '__main__':
    # Исправление для Windows/macOS
    from multiprocessing import set_start_method

    set_start_method('spawn')  # Или 'forkserver' для Linux

    # Локальный кластер Dask
    cluster = LocalCluster(
        n_workers=2,
        processes=False,
        memory_limit='4GB'
    )
    client = Client(cluster)
    print("Dashboard:", client.dashboard_link)

    # Загрузка данных
    ds = xr.open_dataset(
        "input.tif",
        engine="rasterio",
        chunks={"x": 1024, "y": 1024}
    )

    # Пример обработки (нормализация)
    processed = ds / 255.0

    # Проверка и коррекция размерности
    if isinstance(processed, xr.Dataset):
        # Выбор первой переменной (если их несколько)
        processed = processed[list(processed.data_vars.keys())[0]]

    # Удаляем лишние измерения (например, размерность длины 1)
    processed = processed.squeeze()

    # Убедимся, что данные 2D или 3D
    if processed.ndim not in (2, 3):
        raise ValueError(f"Неподдерживаемая размерность: {processed.ndim}")

    # Сохранение с метаданными
    processed.rio.write_crs(ds.rio.crs, inplace=True)
    processed.rio.write_transform(ds.rio.transform(), inplace=True)
    processed.compute().rio.to_raster("output.tif", driver="GTiff")

    # Визуализация matplotlib
    plt.figure(figsize=(12, 6))

    if processed.ndim == 3:
        # Для 3D данных (например, RGB)
        # Берем первые 3 канала и нормализуем
        rgb = processed[:3, :, :].compute()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        plt.imshow(rgb.transpose('y', 'x', 'band'))
    else:
        # Для 2D данных
        plt.imshow(processed.compute(), cmap='viridis')

    plt.title("Processed GeoTIFF")
    plt.colorbar()
    plt.show()

    # Завершение работы
    client.close()
    cluster.close()