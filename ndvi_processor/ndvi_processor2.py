from typing import List, Tuple, Dict, Optional, Any
import os
import xarray as xr
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
import matplotlib.pyplot as plt
import rioxarray
from rasterio.enums import Resampling
import logging
import gc
import glob
from pathlib import Path
from rioxarray.merge import merge_arrays
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import threading
import time
from datetime import datetime
from dask.distributed import as_completed
import psutil
import json
import random

class NDVIProcessor2:
    def __init__(self, b4_dir: str, b5_dir: str, output_dir: str, chunk_size: int = 512) -> None:
        """
        Инициализация процессора NDVI с обработкой по чанкам
        
        Args:
            b4_dir: Директория с файлами B4 (красный канал)
            b5_dir: Директория с файлами B5 (ближний ИК канал)
            output_dir: Директория для сохранения результатов
            chunk_size: Размер чанка для обработки (по умолчанию 512x512)
        """
        self.b4_dir = Path(b4_dir)
        self.b5_dir = Path(b5_dir)
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.chunk_size = chunk_size
        
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        
        # Инициализация Dask кластера с оптимизированными настройками
        memory_limit = psutil.virtual_memory().available * 0.8  # 80% от доступной памяти
        n_workers = max(1, psutil.cpu_count() - 1)  # Оставляем один поток для системы
        
        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit // n_workers,
            silence_logs=logging.ERROR
        )
        self.client = Client(self.cluster)
        
        # Настройка параметров чанков для Dask
        dask.config.set({
            'array.chunk-size': '32MiB',
            'distributed.worker.memory.target': 0.6,  # Целевое использование памяти
            'distributed.worker.memory.spill': 0.7,   # Порог для сброса на диск
            'distributed.worker.memory.pause': 0.8,   # Порог для приостановки
            'distributed.worker.memory.terminate': 0.95  # Порог для завершения
        })
        
        # Создаем директорию для чанков с высокой растительностью
        self.high_ndvi_dir = self.output_dir / "high_ndvi_chunks"
        self.high_ndvi_dir.mkdir(parents=True, exist_ok=True)
        
        # Словарь для отслеживания чанков
        self.chunk_status = {}
        
        # Флаг для остановки мониторинга
        self.monitoring_active = False
        
        # Словарь для хранения графа задач
        self.task_graph = {}
        
    def select_worker(self, chunk_id: str) -> str:
        """
        Выбор воркера для обработки чанка
        
        Args:
            chunk_id: Идентификатор чанка
            
        Returns:
            str: Адрес выбранного воркера
        """
        # Получаем список доступных воркеров
        workers = list(self.client.scheduler_info()['workers'].keys())
        
        if not workers:
            raise RuntimeError("No workers available")
            
        # Выбираем случайного воркера
        selected_worker = random.choice(workers)
        
        # Логируем выбор
        self.logger.info(f"Selected worker {selected_worker} for chunk {chunk_id}")
        
        return selected_worker
        
    def create_task_graph(self, chunk_coords: List[Tuple[int, int, int, int]]) -> Dict[str, Dict]:
        """
        Создание графа задач для обработки чанков
        
        Args:
            chunk_coords: Список координат чанков
            
        Returns:
            Dict: Граф задач
        """
        task_graph = {}
        
        for i, (y_start, y_end, x_start, x_end) in enumerate(chunk_coords):
            chunk_id = f"chunk_{i}"
            
            # Выбираем воркер для чанка
            worker = self.select_worker(chunk_id)
            
            # Создаем запись в графе задач
            task_graph[chunk_id] = {
                'worker': worker,
                'coords': (y_start, y_end, x_start, x_end),
                'status': 'pending',
                'start_time': None,
                'end_time': None
            }
            
        # Сохраняем граф задач
        self.task_graph = task_graph
        
        # Сохраняем граф в JSON для анализа
        with open(self.output_dir / 'task_graph.json', 'w') as f:
            json.dump(task_graph, f, indent=4, default=str)
            
        return task_graph
        
    def update_task_status(self, chunk_id: str, status: str):
        """
        Обновление статуса задачи в графе
        
        Args:
            chunk_id: Идентификатор чанка
            status: Новый статус задачи
        """
        if chunk_id in self.task_graph:
            self.task_graph[chunk_id]['status'] = status
            
            if status == 'started':
                self.task_graph[chunk_id]['start_time'] = datetime.now()
            elif status == 'completed':
                self.task_graph[chunk_id]['end_time'] = datetime.now()
                
            # Обновляем JSON файл
            with open(self.output_dir / 'task_graph.json', 'w') as f:
                json.dump(self.task_graph, f, indent=4, default=str)
                
    def monitor_chunks(self) -> None:
        """Мониторинг распределения чанков по воркерам"""
        try:
            # Получаем информацию о воркерах и их текущих задачах
            scheduler_info = self.client.scheduler_info()
            workers = scheduler_info['workers']
            worker_status = {worker: [] for worker in workers.keys()}
            
            # Получаем информацию о текущих активных задачах
            active_tasks = []
            for worker_addr, worker_info in workers.items():
                if 'processing' in worker_info:
                    for task in worker_info['processing']:
                        if task in self.chunk_status:
                            active_tasks.append((worker_addr, task))
                            worker_status[worker_addr].append(task)
            
            # Логируем статус
            self.logger.info("\nТекущее распределение чанков по воркерам:")
            for worker, chunks in worker_status.items():
                if chunks:  # Показываем только воркеров с активными чанками
                    self.logger.info(f"Воркер {worker}:")
                    for chunk_id in chunks:
                        coords = self.chunk_status[chunk_id]
                        self.logger.info(f"  Чанк {chunk_id}: y={coords[0]}:{coords[1]}, x={coords[2]}:{coords[3]}")
            
            # Создаем гистограмму только если есть активные чанки
            if active_tasks:
                plt.figure(figsize=(12, 6))
                workers_list = list(worker_status.keys())
                chunk_counts = [len(chunks) for chunks in worker_status.values()]
                
                plt.bar(range(len(workers_list)), chunk_counts)
                plt.title('Распределение чанков по воркерам')
                plt.xlabel('Воркер')
                plt.ylabel('Количество чанков')
                plt.xticks(range(len(workers_list)), [f'Worker {i+1}' for i in range(len(workers_list))], rotation=45)
                
                # Добавляем значения над столбцами
                for i, count in enumerate(chunk_counts):
                    if count > 0:  # Показываем только ненулевые значения
                        plt.text(i, count, str(count), ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'chunk_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.close()
                
                # Выводим статистику
                total_chunks = sum(chunk_counts)
                self.logger.info(f"\nСтатистика распределения:")
                self.logger.info(f"Всего активных чанков: {total_chunks}")
                for i, count in enumerate(chunk_counts):
                    if count > 0:  # Показываем только воркеров с активными чанками
                        self.logger.info(f"Воркер {i+1}: {count} чанков ({count/total_chunks*100:.1f}%)")
            else:
                self.logger.info("Нет активных чанков в данный момент")
                
        except Exception as e:
            self.logger.error(f"Ошибка при мониторинге чанков: {str(e)}")
            
    def save_high_ndvi_chunk(self, ndvi_chunk: xr.DataArray, chunk_coords: Tuple[int, int, int, int], transform: Any) -> None:
        """Сохранение чанка с высокой растительностью"""
        y_start, y_end, x_start, x_end = chunk_coords
        
        # Проверяем максимальное значение NDVI в чанке
        max_ndvi = float(ndvi_chunk.max())
        if max_ndvi > 0.8:
            # Получаем координаты центра чанка
            center_y = (y_start + y_end) // 2
            center_x = (x_start + x_end) // 2
            
            # Преобразуем координаты в географические
            lon, lat = rasterio.transform.xy(transform, center_y, center_x)
            
            # Создаем имя файла с координатами
            filename = f"high_ndvi_chunk_lon{lon:.4f}_lat{lat:.4f}.tif"
            filepath = self.high_ndvi_dir / filename
            
            # Гарантируем, что директория существует
            self.high_ndvi_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем чанк
            ndvi_chunk.rio.to_raster(filepath)
            
            # Анализируем чанк
            high_ndvi_mask = ndvi_chunk > 0.8
            high_ndvi_pixels = np.sum(high_ndvi_mask)
            total_pixels = np.prod(ndvi_chunk.shape)
            high_ndvi_percentage = (high_ndvi_pixels / total_pixels) * 100
            
            self.logger.info(f"Найден чанк с высокой растительностью:")
            self.logger.info(f"  Файл: {filename}")
            self.logger.info(f"  Координаты: lon={lon:.4f}, lat={lat:.4f}")
            self.logger.info(f"  Максимальное значение NDVI: {max_ndvi:.4f}")
            self.logger.info(f"  Процент пикселей с NDVI > 0.8: {high_ndvi_percentage:.2f}%")
            self.logger.info(f"  Размер чанка: {y_end-y_start}x{x_end-x_start}")
            
            # Создаем визуализацию чанка
            plt.figure(figsize=(10, 8))
            ndvi_data = ndvi_chunk.squeeze().values
            plt.imshow(ndvi_data, cmap='RdYlGn', vmin=-0.1, vmax=1.0)
            plt.colorbar(label='NDVI')
            plt.title(f'Чанк с высокой растительностью\nМакс. NDVI: {max_ndvi:.4f}')
            plt.savefig(self.high_ndvi_dir / f"preview_{filename.replace('.tif', '.png')}")
            plt.close()
            
    def merge_bands(self) -> None:
        """Объединение слоев B4 и B5 с оптимизированной обработкой памяти"""
        self.logger.info("Объединение слоев B4 и B5...")
        
        b4_files = sorted(glob.glob(os.path.join(self.b4_dir, "*.TIF")))
        b5_files = sorted(glob.glob(os.path.join(self.b5_dir, "*.TIF")))
        
        if not b4_files or not b5_files:
            raise ValueError("Не найдены файлы B4 или B5")
            
        self.logger.info(f"Найдено файлов B4: {len(b4_files)}, B5: {len(b5_files)}")
        
        # Используем меньший размер чанка для загрузки
        chunk_size = {'y': 2048, 'x': 2048}
        
        # Загружаем первый файл для получения метаданных
        with rioxarray.open_rasterio(b4_files[0], chunks=chunk_size) as template:
            self.crs = template.rio.crs
            self.transform = template.rio.transform()
        
        # Загружаем и объединяем B4
        b4_arrays = []
        for f in b4_files:
            with rioxarray.open_rasterio(f, chunks=chunk_size) as ds:
                b4_arrays.append(ds.squeeze().drop_vars('band'))
            gc.collect()
        
        self.b4_merged = merge_arrays(b4_arrays)
        del b4_arrays
        gc.collect()
        
        # Загружаем и объединяем B5
        b5_arrays = []
        for f in b5_files:
            with rioxarray.open_rasterio(f, chunks=chunk_size) as ds:
                b5_arrays.append(ds.squeeze().drop_vars('band'))
            gc.collect()
        
        self.b5_merged = merge_arrays(b5_arrays)
        del b5_arrays
        gc.collect()
        
        self.logger.info("Объединение слоев завершено")
        
    def calculate_ndvi(self) -> Optional[xr.DataArray]:
        """Calculate NDVI using optimized Dask processing"""
        self.logger.info("Расчет NDVI по чанкам...")
        
        # Получаем размеры данных
        height, width = self.b4_merged.shape
        
        # Создаем список координат чанков
        chunk_coords = []
        for y_start in range(0, height, self.chunk_size):
            y_end = min(y_start + self.chunk_size, height)
            for x_start in range(0, width, self.chunk_size):
                x_end = min(x_start + self.chunk_size, width)
                chunk_coords.append((y_start, y_end, x_start, x_end))
        
        # Создаем пустой массив для результата
        ndvi = np.full((height, width), np.nan)
        
        # Создаем граф задач
        task_graph = self.create_task_graph(chunk_coords)
        
        # Обрабатываем чанки последовательно с обновлением статусов
        for i, (y_start, y_end, x_start, x_end) in enumerate(chunk_coords):
            chunk_id = f"chunk_{i}"
            try:
                self.update_task_status(chunk_id, 'started')
                # Получаем чанки данных
                b4_chunk = self.b4_merged[y_start:y_end, x_start:x_end].compute()
                b5_chunk = self.b5_merged[y_start:y_end, x_start:x_end].compute()
                
                # Вычисляем NDVI для чанка
                valid_mask = (b4_chunk > 0.01) & (b5_chunk > 0.01)
                sum_mask = (b4_chunk + b5_chunk) > 0.01
                valid_mask = valid_mask & sum_mask
                
                chunk_ndvi = np.where(
                    valid_mask,
                    (b5_chunk - b4_chunk) / (b5_chunk + b4_chunk),
                    np.nan
                )
                
                # Сохраняем результат
                ndvi[y_start:y_end, x_start:x_end] = chunk_ndvi
                
                # Очищаем память
                del b4_chunk, b5_chunk, valid_mask, sum_mask, chunk_ndvi
                gc.collect()
                
                self.update_task_status(chunk_id, 'completed')
                self.logger.info(f"Обработан чанк {i+1}/{len(chunk_coords)} [{y_start}:{y_end}, {x_start}:{x_end}]")
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке чанка {i+1}: {str(e)}")
                self.update_task_status(chunk_id, 'failed')
                continue
        
        # Преобразуем в xarray.DataArray
        ndvi = xr.DataArray(
            ndvi,
            dims=['y', 'x'],
            coords={
                'y': self.b4_merged.y,
                'x': self.b4_merged.x
            }
        )
        
        # Добавляем метаданные
        ndvi.rio.write_crs(self.crs, inplace=True)
        ndvi.rio.write_transform(self.transform, inplace=True)
        
        self.logger.info("NDVI расчет завершен")
        return ndvi
        
    def create_visualizations(self, ndvi: Optional[xr.DataArray]) -> None:
        """Создание визуализаций"""
        if ndvi is None:
            self.logger.error("Нет данных NDVI для визуализации")
            return
            
        self.logger.info("Создание визуализаций...")
        
        try:
            # Создаем директорию для визуализаций
            vis_dir = self.output_dir / "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Создаем цветовую карту для NDVI
            colors = [
                (0.0, 0.0, 0.5),    # Темно-синий (вода, NDVI < 0)
                (0.0, 0.0, 1.0),    # Синий (вода, NDVI около 0)
                (0.5, 0.5, 0.5),    # Серый (голая почва, NDVI около 0)
                (0.8, 0.8, 0.0),    # Желтый (редкая растительность)
                (0.0, 0.8, 0.0),    # Зеленый (умеренная растительность)
                (0.0, 0.6, 0.0),    # Темно-зеленый (густая растительность)
                (0.0, 0.4, 0.0)     # Очень темно-зеленый (очень густая растительность)
            ]
            ndvi_cmap = LinearSegmentedColormap.from_list('ndvi_cmap', colors, N=256)
            
            # Создание уменьшенной версии для визуализации
            target_size = 1200  # Целевой размер для визуализации
            h, w = ndvi.shape
            scale_factor = max(1, int(max(h, w) / target_size))
            
            # Создаем уменьшенную версию
            ndvi_small = ndvi[::scale_factor, ::scale_factor].compute()
            
            # Создаем фигуру с фиксированным размером
            plt.figure(figsize=(15, 15))
            plt.imshow(ndvi_small, cmap=ndvi_cmap, vmin=-0.1, vmax=0.6)
            plt.colorbar(label='NDVI')
            plt.title('NDVI для всей сцены')
            plt.axis('off')
            plt.savefig(vis_dir / "ndvi_preview.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Сохраняем гистограмму NDVI
            plt.figure(figsize=(10, 6))
            valid_ndvi = ndvi_small.values.flatten() if hasattr(ndvi_small, 'values') else ndvi_small.flatten()
            valid_ndvi = valid_ndvi[~np.isnan(valid_ndvi)]
            plt.hist(valid_ndvi, bins=100, range=(-0.1, 1.0))
            plt.title('Распределение значений NDVI')
            plt.xlabel('NDVI')
            plt.ylabel('Частота')
            plt.savefig(vis_dir / "ndvi_histogram.png")
            plt.close()
            
            self.logger.info("Визуализации созданы")
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании визуализаций: {str(e)}")
        
    def analyze_ndvi(self, ndvi: Optional[xr.DataArray]) -> None:
        """Анализ распределения значений NDVI"""
        self.logger.info("Анализ распределения значений NDVI...")
        
        if ndvi is None:
            self.logger.error("Нет данных NDVI для анализа")
            return
            
        try:
            # Вычисляем базовую статистику
            ndvi_values = ndvi.values[~np.isnan(ndvi.values)]
            
            if len(ndvi_values) == 0:
                self.logger.error("Нет валидных значений NDVI для анализа")
                return
                
            stats = {
                'min': float(np.min(ndvi_values)),
                'max': float(np.max(ndvi_values)),
                'mean': float(np.mean(ndvi_values)),
                'median': float(np.median(ndvi_values)),
                'std': float(np.std(ndvi_values))
            }
            
            # Сохраняем статистику в JSON
            stats_file = self.output_dir / "ndvi_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
                
            self.logger.info(f"Статистика NDVI: {stats}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе NDVI: {str(e)}")

    def start_monitoring(self) -> None:
        """Запуск периодического мониторинга"""
        self.monitoring_active = True
        self._schedule_monitoring()
        
    def stop_monitoring(self) -> None:
        """Остановка периодического мониторинга"""
        self.monitoring_active = False
        
    def _schedule_monitoring(self) -> None:
        """Планирование следующего мониторинга"""
        if self.monitoring_active:
            self.monitor_chunks()
            threading.Timer(5.0, self._schedule_monitoring).start()
            
    def create_band_previews(self, band_files: list, out_dir: Path, band_name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in band_files:
            try:
                with rioxarray.open_rasterio(f) as arr:
                    arr = arr.squeeze()
                    # Уменьшаем размер для превью
                    scale = max(arr.shape[-2] // 512, 1)
                    arr_small = arr[::scale, ::scale]
                    plt.figure(figsize=(8, 8))
                    plt.imshow(arr_small, cmap='gray')
                    plt.title(f"{band_name} preview: {Path(f).name}")
                    plt.axis('off')
                    out_path = out_dir / f"{Path(f).stem}_preview.png"
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                self.logger.error(f"Ошибка при создании превью для {f}: {e}")

    def create_merged_band_preview(self, arr: xr.DataArray, out_path: Path, band_name: str):
        arr = arr.squeeze()
        scale = max(arr.shape[-2] // 1024, 1)
        arr_small = arr[::scale, ::scale].compute() if hasattr(arr, 'compute') else arr[::scale, ::scale]
        plt.figure(figsize=(10, 10))
        plt.imshow(arr_small, cmap='gray')
        plt.title(f"{band_name} merged preview")
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    def process(self) -> None:
        """Основной метод обработки"""
        import time
        timings = {}
        try:
            # 1. Мозайка
            t0 = time.time()
            self.merge_bands()
            t1 = time.time()
            timings['mosaic_seconds'] = t1 - t0
            
            # Превью исходных сцен
            self.create_band_previews(sorted(glob.glob(str(self.b4_dir / '*.TIF'))), self.output_dir / 'b4_previews', 'B4')
            self.create_band_previews(sorted(glob.glob(str(self.b5_dir / '*.TIF'))), self.output_dir / 'b5_previews', 'B5')
            
            # Превью объединённых слоёв
            self.create_merged_band_preview(self.b4_merged, self.output_dir / 'b4_merged_preview.png', 'B4')
            self.create_merged_band_preview(self.b5_merged, self.output_dir / 'b5_merged_preview.png', 'B5')
            
            # 2. Подсчёт NDVI
            t2 = time.time()
            ndvi = self.calculate_ndvi()
            t3 = time.time()
            timings['ndvi_seconds'] = t3 - t2
            
            # Сохраняем timings
            with open(self.output_dir / 'timings.json', 'w') as f:
                json.dump(timings, f, indent=4)
            
            # Сохраняем результат
            if ndvi is not None:
                output_file = self.output_dir / "ndvi_result.tif"
                ndvi.rio.to_raster(output_file)
                self.logger.info(f"Результат сохранен в {output_file}")
            
            # Создаем визуализации
            self.create_visualizations(ndvi)
            
            # Анализируем результаты
            self.analyze_ndvi(ndvi)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке: {str(e)}")
            raise
            
        finally:
            # Закрываем Dask клиент
            self.client.close()
            self.cluster.close()

if __name__ == '__main__':
    # Пример использования
    processor = NDVIProcessor2(
        b4_dir="b4",
        b5_dir="b5",
        output_dir="ndvi_processor/output",
        chunk_size=512
    )
    processor.process() 