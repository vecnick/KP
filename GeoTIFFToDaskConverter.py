import rasterio
import dask.array as da
import numpy as np
from rasterio.windows import Window


class GeoTIFFToDaskConverter:
    def __init__(self, file_path, chunk_size=(256, 256)):
        self.dataset = rasterio.open(file_path)
        self.dtype = self.dataset.dtypes[0]
        self.num_channels = self.dataset.count
        self.height = self.dataset.height
        self.width = self.dataset.width
        self.chunk_size = chunk_size

        # Рассчитываем чанки
        self.chunks = self._init_chunks()

        self.dask_array = self._create_dask_array()

    @staticmethod
    def _calculate_chunks(total_size, chunk_size):
        """Рассчитывает размеры чанков с остатком."""
        num_full = total_size // chunk_size
        remainder = total_size % chunk_size
        chunks = [chunk_size] * num_full
        if remainder > 0:
            chunks.append(remainder)
        return tuple(chunks)

    def _init_chunks(self):
        """Генерирует структуру чанков для всех измерений."""
        height_chunks = self._calculate_chunks(self.height, self.chunk_size[0])
        width_chunks = self._calculate_chunks(self.width, self.chunk_size[1])
        return ((self.num_channels,), height_chunks, width_chunks)

    def _create_dask_array(self):
        def read_block(block_info=None):
            if block_info is None:
                return np.zeros((self.num_channels, self.chunk_size[0], self.chunk_size[1]),
                                dtype=self.dtype)

            # Получаем координаты текущего блока
            _, y_block, x_block = block_info[None]['chunk-location']

            # Рассчитываем стартовые координаты
            y_start = sum(self.chunks[1][:y_block])
            x_start = sum(self.chunks[2][:x_block])

            # Размеры текущего чанка
            chunk_height = self.chunks[1][y_block]
            chunk_width = self.chunks[2][x_block]

            # Создаем окно чтения
            window = Window(
                col_off=x_start,
                row_off=y_start,
                width=chunk_width,
                height=chunk_height
            )

            # Чтение данных
            data = self.dataset.read(
                window=window,
                boundless=True,
                fill_value=0
            )

            return data

        return da.map_blocks(
            read_block,
            chunks=self.chunks,
            dtype=self.dtype,
            meta=np.array([], dtype=self.dtype)
        )

    def close(self):
        self.dataset.close()