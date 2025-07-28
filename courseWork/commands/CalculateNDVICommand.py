import gc
import rasterio
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from courseWork.Chunk import Chunk
from courseWork.commands.Command import Command


class CalculateNDVICommand(Command):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.high_ndvi_dir = self.output_dir / "high_ndvi_chunks"
        self.high_ndvi_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, chunk: Chunk):
        # Временное решение: получаем сцену из атрибута chunk
        scene = getattr(chunk, 'scene', None)
        if not scene:
            raise ValueError("Scene not available in chunk context")

        y_start, y_end, x_start, x_end = chunk.coords

        try:
            # Получение данных чанка
            b4_chunk = scene.b4_merged[y_start:y_end, x_start:x_end].compute()
            b5_chunk = scene.b5_merged[y_start:y_end, x_start:x_end].compute()

            # Расчет NDVI
            valid_mask = (b4_chunk > 0.01) & (b5_chunk > 0.01)
            sum_mask = (b4_chunk + b5_chunk) > 0.01
            valid_mask = valid_mask & sum_mask

            chunk_ndvi = np.where(
                valid_mask,
                (b5_chunk - b4_chunk) / (b5_chunk + b4_chunk),
                np.nan
            )

            # Сохранение результата
            chunk.data = xr.DataArray(
                chunk_ndvi,
                dims=['y', 'x'],
                coords={
                    'y': scene.b4_merged.y[y_start:y_end],
                    'x': scene.b4_merged.x[x_start:x_end]
                }
            )
            chunk.data.rio.write_crs(scene.crs, inplace=True)
            chunk.data.rio.write_transform(scene.transform, inplace=True)

            # Проверка на высокую растительность
            max_ndvi = float(chunk.data.max())
            if max_ndvi > 0.8:
                self.save_high_ndvi_chunk(chunk, scene)

            return True

        except Exception as e:
            raise RuntimeError(f"Error processing chunk: {str(e)}")
        finally:
            del b4_chunk, b5_chunk
            gc.collect()

    def save_high_ndvi_chunk(self, chunk: Chunk, scene):
        """Сохранение чанка с высокой растительностью"""
        y_start, y_end, x_start, x_end = chunk.coords
        center_y = (y_start + y_end) // 2
        center_x = (x_start + x_end) // 2
        lon, lat = rasterio.transform.xy(scene.transform, center_y, center_x)

        filename = f"high_ndvi_{chunk.id}_lon{lon:.4f}_lat{lat:.4f}.tif"
        filepath = self.high_ndvi_dir / filename
        chunk.data.rio.to_raster(filepath)

        # Анализ и визуализация
        high_ndvi_mask = chunk.data > 0.8
        high_ndvi_pixels = np.sum(high_ndvi_mask)
        total_pixels = np.prod(chunk.data.shape)
        high_ndvi_percentage = (high_ndvi_pixels / total_pixels) * 100

        # Визуализация
        plt.figure(figsize=(10, 8))
        plt.imshow(chunk.data.squeeze().values, cmap='RdYlGn', vmin=-0.1, vmax=1.0)
        plt.colorbar(label='NDVI')
        plt.title(f'High NDVI Chunk {chunk.id}')
        plt.savefig(self.high_ndvi_dir / f"{filename}.png")
        plt.close()