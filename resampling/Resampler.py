import dask.array as da
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine


class Resampler:
    def __init__(self, raster_converter):
        self.src_array = raster_converter.dask_array
        self.src_transform = raster_converter.dataset.transform
        self.src_crs = raster_converter.dataset.crs
        self.src_nodata = raster_converter.dataset.nodata
        self.src_res = (
            self.src_transform.a,  # Разрешение по X
            abs(self.src_transform.e)  # Разрешение по Y
        )
        self.src_shape = self.src_array.shape

    def _compute_new_shape(self, dst_res: tuple) -> tuple:
        """Рассчитывает новую форму данных после ресемплинга."""
        dst_width = int(self.src_shape[2] * (self.src_res[0] / dst_res[0]))
        dst_height = int(self.src_shape[1] * (self.src_res[1] / dst_res[1]))
        return (self.src_shape[0], dst_height, dst_width)

    def _compute_chunks(self, dim_size: int, chunk_size: int = 256) -> tuple:
        """Генерирует кортеж чанков для заданного измерения."""
        num_full = dim_size // chunk_size
        remainder = dim_size % chunk_size
        chunks = [chunk_size] * num_full
        if remainder > 0:
            chunks.append(remainder)
        return tuple(chunks)

    def resample(
            self,
            dst_res: tuple,
            resampling_method: Resampling = Resampling.bilinear,
            dst_chunks: tuple = None
    ) -> da.Array:
        # Расчет новой формы
        dst_shape = self._compute_new_shape(dst_res)
        print(f"Новая форма данных: {dst_shape}")  # Отладочный вывод

        # Автоматический расчет чанков
        if not dst_chunks:
            channel_chunks = self.src_array.chunks[0]
            height_chunks = self._compute_chunks(dst_shape[1])
            width_chunks = self._compute_chunks(dst_shape[2])
            dst_chunks = (channel_chunks, height_chunks, width_chunks)
            print(f"Автоматические чанки: {dst_chunks}")  # Отладочный вывод

        # Проверка соответствия чанков и формы
        assert sum(dst_chunks[1]) == dst_shape[1], f"Чанки по высоте {sum(dst_chunks[1])} ≠ {dst_shape[1]}"
        assert sum(dst_chunks[2]) == dst_shape[2], f"Чанки по ширине {sum(dst_chunks[2])} ≠ {dst_shape[2]}"

        # Создаем новый трансформ
        dst_transform = Affine(
            dst_res[0], 0, self.src_transform.c,
            0, -dst_res[1], self.src_transform.f
        )

        # Функция для обработки каждого блока
        def _resample_block(block):
            result = np.zeros((block.shape[0], dst_shape[1], dst_shape[2]), dtype=block.dtype)
            for c in range(block.shape[0]):
                reproject(
                    source=block[c],
                    destination=result[c],
                    src_transform=self.src_transform,
                    src_crs=self.src_crs,
                    dst_transform=dst_transform,
                    dst_crs=self.src_crs,
                    resampling=resampling_method
                )
            return result

        # Применяем ресемплинг и явно задаем чанки
        return da.map_blocks(
            _resample_block,
            self.src_array,
            dtype=self.src_array.dtype,
            chunks=dst_chunks
        ).rechunk(dst_chunks)

    def regrid(
            self,
            target_converter,
            resampling_method: Resampling = Resampling.bilinear,
            dst_chunks: tuple = None
    ) -> da.Array:
        # Получаем параметры целевого растра
        target_shape = target_converter.dask_array.shape
        target_transform = target_converter.dataset.transform

        # Рассчитываем чанки
        if not dst_chunks:
            dst_chunks = (
                self.src_array.chunks[0],  # Каналы
                *target_converter.dask_array.chunks[1:]  # Пространственные оси
            )

        # Проверяем геопривязку
        if not target_converter.dataset.crs:
            raise ValueError("Целевой растр не имеет CRS")

        # Функция для обработки каждого блока
        def _regrid_block(block):
            result = np.zeros((block.shape[0], *target_shape[1:]), dtype=block.dtype)
            for c in range(block.shape[0]):
                reproject(
                    source=block[c],
                    destination=result[c],
                    src_transform=self.src_transform,
                    src_crs=self.src_crs,
                    dst_transform=target_transform,
                    dst_crs=target_converter.dataset.crs,
                    resampling=resampling_method
                )
            return result

        return da.map_blocks(
            _regrid_block,
            self.src_array,
            dtype=self.src_array.dtype,
            chunks=dst_chunks
        ).rechunk(dst_chunks)