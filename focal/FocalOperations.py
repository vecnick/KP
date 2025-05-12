import numpy as np
from scipy.ndimage import generic_filter
import dask.array as da


class FocalOperations:
    def __init__(self, dask_array):
        self.dask_array = dask_array
        self._validate_shape()

    def _validate_shape(self):
        if len(self.dask_array.shape) != 3:
            raise ValueError("Массив должен быть 3D: (каналы, высота, ширина)")

    def apply_focal(self, func, window_size=3):
        """
        Применяет фокальную операцию ко всем чанкам.

        :param func: Функция для обработки окна (например, np.mean)
        :param window_size: Размер окна (только нечетные числа)
        :return: Новый Dask-массив с результатом
        """
        if window_size % 2 == 0:
            raise ValueError("Размер окна должен быть нечетным")

        pad = window_size // 2
        depth = (0, pad, pad)  # Перекрытие по высоте и ширине

        def wrapped_func(chunk):
            result = np.empty_like(chunk)
            for c in range(chunk.shape[0]):
                result[c] = generic_filter(
                    chunk[c],
                    func,
                    size=window_size,
                    mode='reflect'
                )
            return result

        return da.overlap.map_overlap(
            wrapped_func,
            self.dask_array,
            depth=depth,
            boundary='reflect',
            dtype=self.dask_array.dtype
        )

    def focal_mean(self, window_size=3):
        return self.apply_focal(np.mean, window_size)

    def focal_median(self, window_size=3):
        return self.apply_focal(np.median, window_size)

    def focal_sum(self, window_size=3):
        return self.apply_focal(np.sum, window_size)