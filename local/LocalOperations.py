import dask.array as da
import numpy as np


class LocalOperations:
    def __init__(self, dask_array):
        self.dask_array = dask_array

    def select_band(self, band_index: int):
        """
        Выбор канала из многоканального изображения.

        :param band_index: Индекс канала (начиная с 0)
        :return: Новый объект LocalOperations с выбранным каналом
        """
        if band_index >= self.dask_array.shape[0]:
            raise IndexError(f"Канал {band_index} не существует. Всего каналов: {self.dask_array.shape[0]}")

        # Выбираем канал и сохраняем размерность (добавляем новую ось)
        selected = self.dask_array[band_index][np.newaxis, :, :]
        return LocalOperations(selected)

    # -------------------------------
    # Базовые арифметические операции
    # -------------------------------
    def add(self, other):
        """Покомпонентное сложение: A + B или A + const"""
        return LocalOperations(self.dask_array + other)

    def subtract(self, other):
        """Покомпонентное вычитание: A - B или A - const"""
        return LocalOperations(self.dask_array - other)

    def multiply(self, other):
        """Покомпонентное умножение: A * B или A * const"""
        return LocalOperations(self.dask_array * other)

    def divide(self, other):
        """Покомпонентное деление: A / B или A / const"""
        return LocalOperations(self.dask_array / other)

    # -------------------------------
    # Реляционные операции
    # -------------------------------
    def greater_than(self, other):
        """Покомпонентное сравнение: A > B или A > const"""
        return LocalOperations(self.dask_array > other)

    def less_than(self, other):
        """Покомпонентное сравнение: A < B или A < const"""
        return LocalOperations(self.dask_array < other)

    # -------------------------------
    # Математические функции
    # -------------------------------
    def abs(self):
        """Абсолютное значение"""
        return LocalOperations(da.abs(self.dask_array))

    def sqrt(self):
        """Квадратный корень"""
        return LocalOperations(da.sqrt(self.dask_array))

    def log(self, base=np.e):
        """Логарифм с произвольным основанием"""
        return LocalOperations(da.log(self.dask_array) / da.log(base))

    # -------------------------------
    # Условные операции
    # -------------------------------
    def conditional(self, condition, true_value, false_value):
        """
        Условная операция: if condition then true_value else false_value
        Пример: conditional(A > 50, 1, 0)
        """
        return LocalOperations(
            da.where(condition.dask_array, true_value, false_value)
        )

    # -------------------------------
    # Булевы/битовые операции
    # -------------------------------
    def logical_and(self, other):
        """Логическое И: A & B"""
        return LocalOperations(self.dask_array & other)

    def logical_or(self, other):
        """Логическое ИЛИ: A | B"""
        return LocalOperations(self.dask_array | other)

    # -------------------------------
    # Вспомогательные методы
    # -------------------------------
    def compute(self):
        """Выполнить вычисления и вернуть результат как массив NumPy"""
        return self.dask_array.compute()

    def persist(self):
        """Сохранить результаты в памяти"""
        self.dask_array = self.dask_array.persist()
        return self

    @property
    def result(self):
        """Возвращает Dask-массив с результатами"""
        return self.dask_array