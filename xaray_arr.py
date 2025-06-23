import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from dask.distributed import Client, LocalCluster
from rioxarray.merge import merge_datasets

if __name__ == '__main__':
    # Исправление для Windows/macOS
    from multiprocessing import set_start_method

    set_start_method('spawn')  # Или 'forkserver' для Linux

    # Локальный кластер Dask
    cluster = LocalCluster(
        n_workers=2,
        processes=False,
        memory_limit='5GB'
    )
    client = Client(cluster)
    print("Dashboard:", client.dashboard_link)

    # Путь к директории с TIF файлами
    tif_dir = "b4"  # Измените на свой путь
    tif_files = glob.glob(os.path.join(tif_dir, "*.TIF"))

    if not tif_files:
        raise ValueError(f"TIF файлы не найдены в директории {tif_dir}")

    print(f"Найдено {len(tif_files)} TIF файлов")

    # Загрузка всех файлов в список датасетов
    datasets = []
    for file_path in tif_files:
        print(f"Загрузка файла: {os.path.basename(file_path)}")
        ds = xr.open_dataset(
            file_path,
            engine="rasterio",
            chunks={"x": 1024, "y": 1024}
        )

        # Добавляем имя файла как атрибут для идентификации
        filename = os.path.basename(file_path)
        ds.attrs['filename'] = filename

        # Убедимся, что данные имеют правильную форму для визуализации
        # Удалим лишние размерности длиной 1
        for var in ds.data_vars:
            if hasattr(ds[var], 'squeeze'):
                ds[var] = ds[var].squeeze()

        datasets.append(ds)

    # Проверим размерности данных для лучшего понимания структуры
    shapes = []
    for i, ds in enumerate(datasets):
        var_name = list(ds.data_vars)[0]  # Берем первую переменную
        shape = ds[var_name].shape
        shapes.append(shape)
        print(f"Файл {i + 1}: {ds.attrs.get('filename')} - Форма: {shape}, Размерности: {ds[var_name].dims}")

    # Выбираем стратегию объединения на основе размерностей и форм данных
    # Проверяем, одинаковы ли формы всех файлов
    same_shape = all(shape == shapes[0] for shape in shapes)

    if same_shape:
        print("Объединение файлов как каналы одного изображения")

        # Создаем новый датасет с переменными из каждого файла
        merged_ds = xr.Dataset()

        for i, ds in enumerate(datasets):
            var_name = list(ds.data_vars)[0]
            # Используем имя файла или индекс как имя переменной
            new_var_name = f"band_{i + 1}"
            merged_ds[new_var_name] = ds[var_name]

        # Преобразуем в DataArray для визуализации
        # Объединяем все переменные в одну 3D DataArray
        vars_to_stack = list(merged_ds.data_vars)
        if len(vars_to_stack) > 0:
            merged_data = merged_ds[vars_to_stack].to_array(dim='band')
        else:
            raise ValueError("Нет переменных для объединения")
    else:
        print("Объединение файлов как мозаику")
        # Убедимся, что все датасеты имеют CRS
        for ds in datasets:
            if not hasattr(ds.rio, 'crs') or ds.rio.crs is None:
                raise ValueError(f"Файл {ds.attrs.get('filename')} не имеет CRS")

        # Проверка, что все CRS одинаковые
        crs_list = [ds.rio.crs for ds in datasets]
        if not all(crs == crs_list[0] for crs in crs_list):
            print("Предупреждение: Разные CRS в файлах. Перепроецирование к первому CRS.")
            target_crs = crs_list[0]
            for i in range(1, len(datasets)):
                if datasets[i].rio.crs != target_crs:
                    datasets[i] = datasets[i].rio.reproject(target_crs)

        # Объединение по пространственным координатам
        merged_ds = merge_datasets(datasets)

        # Преобразуем в DataArray для дальнейшей обработки
        # Берем первую переменную, если их несколько
        first_var = list(merged_ds.data_vars)[0]
        merged_data = merged_ds[first_var]

    # Вывод информации о объединенных данных
    print(f"Объединенные данные: Форма: {merged_data.shape}, Размерности: {merged_data.dims}")

    # Обработка объединенных данных (например, нормализация)
    # Нормализуем данные для визуализации
    # Сначала вычисляем максимум
    data_max = merged_data.max().compute()
    if data_max != 0:
        processed = merged_data / data_max
    else:
        processed = merged_data  # Избегаем деления на ноль

    # Визуализация matplotlib
    plt.figure(figsize=(15, 10))

    # Проверяем размерность данных
    if len(processed.dims) == 3 and 'band' in processed.dims:
        # Для 3D данных с измерением 'band'
        num_bands = processed.sizes['band']

        # Определяем, в каком порядке идут размерности
        dim_order = list(processed.dims)
        band_idx = dim_order.index('band')

        print(f"3D данные с измерением 'band' на позиции {band_idx}")

        if num_bands >= 3:
            # RGB композит из первых трех каналов
            print("Создание RGB композита из первых трех каналов")
            # Выбираем первые 3 канала
            rgb_data = processed.isel(band=slice(0, 3))

            # Нормализация для визуализации
            rgb_data = rgb_data.compute()

            # Для каждого канала выполняем нормализацию
            for i in range(min(3, num_bands)):
                band = rgb_data.isel(band=i)
                vmin, vmax = np.nanpercentile(band.values, [2, 98])
                if vmax > vmin:  # Избегаем деления на ноль
                    rgb_data[{'band': i}] = (band - vmin) / (vmax - vmin)
                rgb_data[{'band': i}] = np.clip(rgb_data[{'band': i}], 0, 1)

            # Переставляем оси для matplotlib - нужно 'band' в конце
            if band_idx != 2:  # Если 'band' не последний
                # Переставляем размерности так, чтобы канал был последним
                new_order = [d for d in dim_order if d != 'band'] + ['band']
                rgb_data = rgb_data.transpose(*new_order)

            # Преобразуем в numpy массив для matplotlib
            rgb_array = rgb_data.values

            # Если первая размерность длины 1, удаляем её
            if rgb_array.shape[0] == 1:
                rgb_array = rgb_array[0]

            plt.imshow(rgb_array)
            plt.title("RGB композит из объединенных TIF файлов")
        else:
            # Отображение каждого канала отдельно
            print("Отображение каждого канала отдельно")
            for i in range(num_bands):
                plt.subplot(1, num_bands, i + 1)

                # Выбираем канал
                channel_data = processed.isel(band=i).compute()

                # Преобразуем в numpy массив
                channel_array = channel_data.values

                # Если массив имеет лишние размерности длины 1, удаляем их
                channel_array = np.squeeze(channel_array)

                plt.imshow(channel_array, cmap='viridis')
                plt.title(f"Канал {i + 1}")
                plt.colorbar()
    elif len(processed.dims) == 2:
        # Для 2D данных - просто отображаем
        print("Отображение 2D данных")
        plt.imshow(processed.compute(), cmap='viridis')
        plt.title("Объединенные TIF файлы")
        plt.colorbar()
    else:
        # Для других размерностей
        print(f"Данные имеют нестандартную размерность: {len(processed.dims)}")
        # Пытаемся преобразовать к 2D для отображения
        if len(processed.dims) > 2:
            # Оставляем только последние две размерности
            slice_dims = {dim: 0 for dim in list(processed.dims)[:-2]}
            processed_2d = processed.isel(slice_dims)
            plt.imshow(processed_2d.compute(), cmap='viridis')
        else:
            # Если меньше 2D, не можем отобразить как изображение
            print("Невозможно отобразить данные как изображение")

    plt.tight_layout()
    plt.savefig("merged_visualization.png", dpi=300)
    plt.show()

    # Сохранение объединенного результата
    print("Сохранение объединенного результата")
    # Убедимся, что у нас есть CRS и transform
    if hasattr(processed.rio, 'crs') and processed.rio.crs is not None:
        processed.rio.to_raster("merged_output.tif", driver="GTiff")
    else:
        print("Предупреждение: Отсутствует CRS, сохранение без геопривязки")
        # Сохраняем как обычный TIFF без геопривязки
        xr.Dataset({"data": processed}).to_netcdf("merged_output.nc")

    # Завершение работы
    client.close()
    cluster.close()