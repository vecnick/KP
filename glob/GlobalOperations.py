import dask.array as da
import numpy as np

class GlobalOperations:
    def __init__(self, dask_array):
        """
        Инициализация глобальных операций

        :param dask_array: Dask-массив данных (формат: [каналы, высота, ширина])
        """
        self.dask_array = dask_array

    # Базовые агрегационные операции
    def sum(self, axis=None, skipna=True):
        """Сумма значений по осям"""
        return self._apply_agg(da.nansum if skipna else da.sum, axis)

    def mean(self, axis=None, skipna=True):
        """Среднее значение по осям"""
        return self._apply_agg(da.nanmean if skipna else da.mean, axis)

    def min(self, axis=None, skipna=True):
        """Минимальное значение по осям"""
        return self._apply_agg(da.nanmin if skipna else da.min, axis)

    def max(self, axis=None, skipna=True):
        """Максимальное значение по осям"""
        return self._apply_agg(da.nanmax if skipna else da.max, axis)

    # Статистические операции
    def std(self, axis=None, skipna=True):
        """Стандартное отклонение"""
        return self._apply_agg(da.nanstd if skipna else da.std, axis)

    def var(self, axis=None, skipna=True):
        """Дисперсия"""
        return self._apply_agg(da.nanvar if skipna else da.var, axis)

    def median(self, axis=None, skipna=True):
        """Медиана"""
        return self._apply_agg(da.nanmedian if skipna else da.median, axis)

    def mode(self, axis=None):
        """Вычисление моды с поддержкой Dask."""

        def _mode_chunk(chunk):
            values, counts = np.unique(chunk, return_counts=True)
            if len(values) == 0:
                return np.array([np.nan])
            return values[np.argmax(counts)]

        # Применяем к каждому блоку
        return da.map_blocks(
            _mode_chunk,
            self.dask_array,
            dtype=self.dask_array.dtype,
            drop_axis=axis  # Удаляем оси, если нужно
        )

    def histogram(self, bins=10, range=None):
        """Гистограмма (возвращает частоты и бины)"""
        return da.histogram(self.dask_array, bins=bins, range=range)

    # Специальные операции
    def percentile(self, q, axis=None):
        cleaned = da.where(da.isnan(self.dask_array), np.nan, self.dask_array)

        return da.percentile(cleaned.flatten(), q)

    def count(self, axis=None):
        """Количество не-NaN значений"""
        return da.count_nonzero(~da.isnan(self.dask_array), axis=axis)

    def any(self, axis=None):
        """Логическое ИЛИ (хотя бы одно True)"""
        return da.any(self.dask_array, axis=axis)

    def all(self, axis=None):
        """Логическое И (все True)"""
        return da.all(self.dask_array, axis=axis)

    # Вспомогательные методы
    def _apply_agg(self, func, axis):
        """Обертка для агрегационных функций"""
        if axis is None:
            return func(self.dask_array)
        return func(self.dask_array, axis=axis)

    def _apply_custom_agg(self, func):
        """Для пользовательских агрегаций"""
        return func(self.dask_array)

    def compute(self):
        """Выполнить вычисления и вернуть результат"""
        return self.dask_array.compute()