import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import rioxarray
import os
from rasterio.enums import Resampling
import logging
import gc
import dask
import glob
from pathlib import Path
from rioxarray.merge import merge_datasets
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import threading
import time

class NDVIProcessor:
    def __init__(self, b4_dir: str, b5_dir: str, output_dir: str):
        """
        Инициализация процессора NDVI
        
        Args:
            b4_dir: Директория с файлами B4 (красный канал)
            b5_dir: Директория с файлами B5 (ближний ИК канал)
            output_dir: Директория для сохранения результатов
        """
        self.b4_dir = Path(b4_dir)
        self.b5_dir = Path(b5_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Инициализация Dask клиента
        self._setup_dask()
        
    def _setup_dask(self):
        """Настройка Dask клиента"""
        dask.config.set({"distributed.worker.memory.target": 0.85})
        dask.config.set({"distributed.worker.memory.spill": 0.9})
        dask.config.set({"distributed.worker.memory.pause": 0.95})
        
        self.cluster = LocalCluster(
            n_workers=4,
            processes=True,
            memory_limit='6GB',
            threads_per_worker=2,
            silence_logs=logging.WARNING
        )
        self.client = Client(self.cluster)
        self.logger.info(f"Dashboard: {self.client.dashboard_link}")
        
    def merge_bands(self):
        """Объединение слоев B4 и B5 с помощью мозаики (merge_datasets)"""
        self.logger.info("Объединение слоев B4 и B5 (мозаика)...")
        
        b4_files = sorted(glob.glob(os.path.join(self.b4_dir, "*.TIF")))
        b5_files = sorted(glob.glob(os.path.join(self.b5_dir, "*.TIF")))
        
        if not b4_files or not b5_files:
            raise ValueError("Не найдены файлы B4 или B5")
        self.logger.info(f"Найдено {len(b4_files)} файлов B4 и {len(b5_files)} файлов B5")
        
        # Загружаем датасеты как xarray.Dataset
        b4_datasets = [xr.open_dataset(f, engine="rasterio", chunks={"x": 1024, "y": 1024}) for f in b4_files]
        b5_datasets = [xr.open_dataset(f, engine="rasterio", chunks={"x": 1024, "y": 1024}) for f in b5_files]
        
        # Мозаика: используется первый непустой пиксель (по умолчанию)
        merged_b4_ds = merge_datasets(b4_datasets)
        merged_b5_ds = merge_datasets(b5_datasets)
        
        # Берём первую переменную (обычно это 'band_data' или аналогичная)
        self.b4_merged = merged_b4_ds[list(merged_b4_ds.data_vars)[0]]
        self.b5_merged = merged_b5_ds[list(merged_b5_ds.data_vars)[0]]
        
        # --- КАЛИБРОВКА В REFLECTANCE ---
        self.logger.info("Приведение значений к reflectance по формуле из MTL.xml...")
        # Значения из вашего MTL.xml
        REFLECTANCE_MULT = 2.75e-05
        REFLECTANCE_ADD = -0.2
        orig_b4_min, orig_b4_max = float(self.b4_merged.min()), float(self.b4_merged.max())
        orig_b5_min, orig_b5_max = float(self.b5_merged.min()), float(self.b5_merged.max())
        self.logger.info(f"B4 до калибровки: min={orig_b4_min}, max={orig_b4_max}")
        self.logger.info(f"B5 до калибровки: min={orig_b5_min}, max={orig_b5_max}")
        self.b4_merged = self.b4_merged * REFLECTANCE_MULT + REFLECTANCE_ADD
        self.b5_merged = self.b5_merged * REFLECTANCE_MULT + REFLECTANCE_ADD
        cal_b4_min, cal_b4_max = float(self.b4_merged.min()), float(self.b4_merged.max())
        cal_b5_min, cal_b5_max = float(self.b5_merged.min()), float(self.b5_merged.max())
        self.logger.info(f"B4 после калибровки: min={cal_b4_min}, max={cal_b4_max}")
        self.logger.info(f"B5 после калибровки: min={cal_b5_min}, max={cal_b5_max}")
        # ---
        
        # Сохраняем объединенные слои
        self.b4_merged.rio.to_raster(os.path.join(self.output_dir, "b4_merged.tif"))
        self.b5_merged.rio.to_raster(os.path.join(self.output_dir, "b5_merged.tif"))
        self.logger.info("Слои успешно объединены и сохранены (мозаика)")
        
    def calculate_ndvi(self):
        """Расчет NDVI"""
        self.logger.info("Расчет NDVI...")
        valid_mask = (self.b4_merged > 0.01) & (self.b5_merged > 0.01) & \
                    (self.b4_merged < 1.0) & (self.b5_merged < 1.0)
        # NDVI через persist (асинхронно)
        self.ndvi = xr.where(
            valid_mask,
            (self.b5_merged - self.b4_merged) / (self.b5_merged + self.b4_merged),
            np.nan
        ).persist()
        # Запускаем мониторинг в отдельном потоке
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self.monitor_dask_chunks, args=(stop_event,))
        monitor_thread.start()
        # Триггерим вычисления (например, сохраняем NDVI на диск)
        ndvi_path = os.path.join(self.output_dir, 'ndvi.tif')
        self.ndvi.rio.to_raster(ndvi_path)
        # После завершения — останавливаем мониторинг
        stop_event.set()
        monitor_thread.join()
        
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
        
        # Создаем RGB версию NDVI
        ndvi_values = self.ndvi.values.squeeze()
        if len(ndvi_values.shape) > 2:
            ndvi_values = ndvi_values[0]  # Берем первый слой, если есть дополнительные измерения
        
        # Создаем маску для валидных значений NDVI
        valid_mask = (ndvi_values >= -1) & (ndvi_values <= 1)
        
        # Создаем RGB массив
        rgb = np.zeros((*ndvi_values.shape, 3), dtype=np.uint8)
        
        # Применяем цветовую карту только к валидным значениям
        valid_indices = np.where(valid_mask)
        for i, j in zip(*valid_indices):
            ndvi_val = ndvi_values[i, j]
            if ndvi_val < 0:  # Вода/снег/облака
                rgb[i, j] = [0, 0, 128]  # Темно-синий
            elif ndvi_val < 0.2:  # Городские территории/почва
                rgb[i, j] = [128, 128, 128]  # Серый
            elif ndvi_val < 0.4:  # Редкая растительность
                rgb[i, j] = [255, 255, 0]  # Желтый
            elif ndvi_val < 0.6:  # Умеренная растительность
                rgb[i, j] = [0, 255, 0]  # Зеленый
            else:  # Густая растительность
                rgb[i, j] = [0, 128, 0]  # Темно-зеленый
        
        # Сохраняем RGB версию
        rgb_path = os.path.join(self.output_dir, 'ndvi_rgb.tif')
        with rasterio.open(ndvi_path) as src:
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 3,
                'nodata': None
            })
        
        with rasterio.open(rgb_path, 'w', **profile) as dst:
            dst.write(rgb.transpose(2, 0, 1))
        
        self.logger.info("NDVI рассчитан и сохранен")
        
        return self.ndvi
        
    def create_visualizations(self, ndvi):
        """Создание визуализаций"""
        self.logger.info("Создание визуализаций...")
        
        # Гистограммы B4 и B5
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        b4_values = self.b4_merged.values.flatten()
        b4_values = b4_values[np.isfinite(b4_values)]
        plt.hist(b4_values, bins=100, alpha=0.7, label='B4')
        plt.title('Гистограмма B4')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        b5_values = self.b5_merged.values.flatten()
        b5_values = b5_values[np.isfinite(b5_values)]
        plt.hist(b5_values, bins=100, alpha=0.7, label='B5')
        plt.title('Гистограмма B5')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'histograms_b4_b5.png')
        plt.close()
        
        # Цветная карта NDVI
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
        h, w = ndvi.shape[-2:]  # Берем последние два измерения
        scale_factor = max(1, int(max(h, w) / target_size))
        self.logger.info(f"Уменьшение разрешения для визуализации (коэффициент: {scale_factor})...")
        
        # Создаем уменьшенную версию
        ndvi_small = ndvi[..., ::scale_factor, ::scale_factor]
        ndvi_downsampled = ndvi_small.compute()
        ndvi_downsampled = ndvi_downsampled.squeeze()
        
        # Получаем размеры после downsampling
        h_small, w_small = ndvi_downsampled.shape[-2:]
        self.logger.info(f"Размеры после downsampling: {h_small}x{w_small}")
        
        # Создаем фигуру с фиксированным размером
        plt.figure(figsize=(15, 15))
        plt.imshow(ndvi_downsampled, cmap=ndvi_cmap, vmin=-0.1, vmax=0.6)  # Изменен диапазон отображения
        plt.colorbar(label='NDVI')
        plt.title('NDVI для всей сцены')
        plt.axis('off')
        plt.savefig(self.output_dir / "ndvi_preview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Визуализации созданы")
        
    def analyze_ndvi(self, ndvi):
        """Анализ распределения значений NDVI"""
        self.logger.info("Анализ распределения значений NDVI...")
        
        # Вычисление статистик
        ndvi_values = ndvi.values.flatten()
        ndvi_values = ndvi_values[np.isfinite(ndvi_values)]
        
        stats = {
            'min': np.min(ndvi_values),
            'max': np.max(ndvi_values),
            'mean': np.mean(ndvi_values),
            'median': np.median(ndvi_values),
            'std': np.std(ndvi_values),
            'percentiles': np.percentile(ndvi_values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        }
        
        # Распределение по категориям
        categories = {
            'water': np.sum(ndvi_values < 0) / len(ndvi_values) * 100,
            'urban': np.sum((ndvi_values >= 0) & (ndvi_values < 0.2)) / len(ndvi_values) * 100,
            'sparse_veg': np.sum((ndvi_values >= 0.2) & (ndvi_values < 0.4)) / len(ndvi_values) * 100,
            'mod_veg': np.sum((ndvi_values >= 0.4) & (ndvi_values < 0.6)) / len(ndvi_values) * 100,
            'dense_veg': np.sum(ndvi_values >= 0.6) / len(ndvi_values) * 100
        }
        
        # Логирование результатов
        self.logger.info("Статистика NDVI:")
        self.logger.info(f"  Мин: {stats['min']:.4f}, Макс: {stats['max']:.4f}")
        self.logger.info(f"  Среднее: {stats['mean']:.4f}, Медиана: {stats['median']:.4f}, Стд. откл.: {stats['std']:.4f}")
        self.logger.info("  Процентили:")
        self.logger.info(f"    1%: {stats['percentiles'][0]:.4f}, 5%: {stats['percentiles'][1]:.4f}, 10%: {stats['percentiles'][2]:.4f}")
        self.logger.info(f"    25%: {stats['percentiles'][3]:.4f}, 50%: {stats['percentiles'][4]:.4f}, 75%: {stats['percentiles'][5]:.4f}")
        self.logger.info(f"    90%: {stats['percentiles'][6]:.4f}, 95%: {stats['percentiles'][7]:.4f}, 99%: {stats['percentiles'][8]:.4f}")
        
        self.logger.info("Распределение значений NDVI:")
        self.logger.info(f"  Вода/снег/облака (NDVI < 0): {categories['water']:.2f}%")
        self.logger.info(f"  Городские территории/почва (0 <= NDVI < 0.2): {categories['urban']:.2f}%")
        self.logger.info(f"  Редкая растительность (0.2 <= NDVI < 0.4): {categories['sparse_veg']:.2f}%")
        self.logger.info(f"  Умеренная растительность (0.4 <= NDVI < 0.6): {categories['mod_veg']:.2f}%")
        self.logger.info(f"  Густая растительность (NDVI >= 0.6): {categories['dense_veg']:.2f}%")
        
    def monitor_dask_chunks(self, stop_event=None):
        """Мониторинг распределения чанков по воркерам во время вычислений (пока есть активные задачи)"""
        try:
            if not self.client or self.client.status == 'closed':
                return
            while True:
                if stop_event and stop_event.is_set():
                    break
                workers = self.client.scheduler_info()['workers']
                worker_chunks = {worker: [] for worker in workers}
                total_tasks = 0
                # Получаем информацию о текущих задачах
                for worker, info in workers.items():
                    # Активные задачи
                    for task_key in info.get('processing', []):
                        worker_chunks[worker].append({'key': task_key, 'status': 'processing'})
                        total_tasks += 1
                    # В очереди
                    for task_key in info.get('ready', []):
                        worker_chunks[worker].append({'key': task_key, 'status': 'ready'})
                        total_tasks += 1
                # Логируем
                self.logger.info("\nDASK: распределение задач по воркерам:")
                for worker, tasks in worker_chunks.items():
                    if tasks:
                        self.logger.info(f"Воркер {worker}:")
                        for t in tasks:
                            self.logger.info(f"  {t['key']} [{t['status']}]")
                # Визуализация
                plt.figure(figsize=(12, 6))
                worker_names = list(worker_chunks.keys())
                chunk_counts = [len(tasks) for tasks in worker_chunks.values()]
                plt.bar(range(len(worker_names)), chunk_counts)
                plt.title('Dask: распределение задач по воркерам')
                plt.xlabel('Воркер')
                plt.ylabel('Количество задач')
                plt.xticks(range(len(worker_names)), [f'Worker {i+1}' for i in range(len(worker_names))], rotation=45)
                plt.tight_layout()
                output_path = self.output_dir / 'dask_chunk_distribution.png'
                plt.savefig(output_path)
                plt.close()
                self.logger.info(f"DASK: гистограмма сохранена в {output_path}")
                # Если нет активных задач — выходим
                if total_tasks == 0:
                    self.logger.info("DASK: нет активных задач, мониторинг завершён.")
                    break
                time.sleep(2)
        except Exception as e:
            logging.error(f"Ошибка при мониторинге чанков Dask: {str(e)}")
        
    def process(self):
        """Полный процесс обработки"""
        try:
            self.merge_bands()
            ndvi = self.calculate_ndvi()
            self.create_visualizations(ndvi)
            self.analyze_ndvi(ndvi)
            self.logger.info("Обработка успешно завершена!")
        finally:
            self.client.close()
            self.cluster.close()
            
if __name__ == '__main__':
    # Пример использования
    processor = NDVIProcessor(
        b4_dir="b4",
        b5_dir="b5",
        output_dir="ndvi_processor/output"
    )
    processor.process() 