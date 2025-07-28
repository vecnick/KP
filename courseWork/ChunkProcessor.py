import rasterio
import numpy as np
from pathlib import Path
import shutil
import warnings


class ChunkProcessor:
    """Обработчик чанков с оптимизированной работой с данными"""

    def __init__(self, b4_path: str, b5_path: str, crs: any, transform: any, output_dir: Path):
        self.b4_path = b4_path
        self.b5_path = b5_path
        self.crs = crs
        self.transform = transform
        self.output_dir = output_dir
        self.high_ndvi_dir = self.output_dir / "high_ndvi_chunks"
        self.high_ndvi_dir.mkdir(parents=True, exist_ok=True)

    def process(self, coords, chunk_id: str):
        """Основной метод обработки чанка"""
        try:
            y_start, y_end, x_start, x_end = coords
            width = x_end - x_start
            height = y_end - y_start
            window = rasterio.windows.Window(x_start, y_start, width, height)

            # Чтение только необходимой части данных
            with rasterio.open(self.b4_path) as src:
                b4_block = src.read(1, window=window).astype(np.float32)

            with rasterio.open(self.b5_path) as src:
                b5_block = src.read(1, window=window).astype(np.float32)

            # Подавление предупреждений о делении на ноль
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # Расчет NDVI с защитой от деления на ноль
                denominator = b4_block + b5_block

                # Создаем маску валидных пикселей
                valid_mask = (
                        (b4_block > 0.01) &
                        (b5_block > 0.01) &
                        (denominator > 0.01)
                )

                # Вычисляем NDVI только для валидных пикселей
                ndvi_array = np.full_like(b4_block, np.nan, dtype=np.float32)
                ndvi_array[valid_mask] = (
                        (b5_block[valid_mask] - b4_block[valid_mask]) /
                        denominator[valid_mask]
                )

            # Сохранение результата
            chunk_dir = self.output_dir / "ndvi_chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_filename = chunk_dir / f"{chunk_id}.tif"

            with rasterio.open(
                    chunk_filename,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=ndvi_array.dtype,
                    crs=self.crs,
                    transform=rasterio.windows.transform(window, self.transform),
                    nodata=np.nan
            ) as dst:
                dst.write(ndvi_array, 1)

            # Проверка на высокую растительность
            if not np.all(np.isnan(ndvi_array)):
                max_ndvi = np.nanmax(ndvi_array)
                if max_ndvi > 0.8:
                    high_ndvi_filename = self.high_ndvi_dir / f"high_{chunk_id}.tif"
                    shutil.copy(chunk_filename, high_ndvi_filename)

                    # Дополнительная информация о местоположении
                    center_y = y_start + height // 2
                    center_x = x_start + width // 2
                    lon, lat = rasterio.transform.xy(self.transform, center_y, center_x)
                    # Логируем информацию о высокой растительности
                    print(f"High NDVI found in chunk {chunk_id}: "
                          f"max={max_ndvi:.2f}, center=({lon:.4f}, {lat:.4f})")

            return chunk_id, str(chunk_filename)

        except Exception as e:
            raise RuntimeError(f"Error processing chunk {chunk_id}: {str(e)}")