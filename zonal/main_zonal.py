from GeoTIFFToDaskConverter import GeoTIFFToDaskConverter
from zonal.ZonalOperations import ZonalOperations

# Пример использования
if __name__ == "__main__":
    # Создание конвертера
    converter = GeoTIFFToDaskConverter("../input.tif", chunk_size=(256, 256))

    # Инициализация зональных операций
    zonal = ZonalOperations(
        converter.dask_array,
        transform=converter.dataset.transform,
        crs=converter.dataset.crs
    )

    # Пример 1: Сумма значений в зоне (ветер > 3 м/с)
    wind_speed_mask = converter.dask_array > 3.0
    total = zonal.from_mask(wind_speed_mask).zonal_sum()

    # # Пример 2: Среднее значение в полигоне из Shapefile
    # field_mask = zonal.from_vector("fields.shp")
    # average = zonal.zonal_mean(field_mask)
    #
    # # Пример 3: Статистика по нескольким зонам
    # zones = zonal.from_vector("regions.shp", attribute="class_id")
    # stats = {
    #     "sum": zonal.zonal_sum(zones),
    #     "mean": zonal.zonal_mean(zones),
    #     "max": zonal.zonal_max(zones)
    # }

    # Результаты
    print("Сумма в зоне ветра:", total.compute())
    # print("Среднее по полям:", average.compute())
    # print("Статистика по регионам:", {k: v.compute() for k, v in stats.items()})

    converter.close()