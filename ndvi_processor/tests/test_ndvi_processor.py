import unittest
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import shutil
from ..ndvi_processor2 import NDVIProcessor2

class TestNDVIProcessor(unittest.TestCase):
    def setUp(self):
        # Создаем временные директории для тестов
        self.temp_dir = tempfile.mkdtemp()
        self.b4_dir = Path(self.temp_dir) / "b4"
        self.b5_dir = Path(self.temp_dir) / "b5"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Создаем тестовые данные
        self.create_test_data()
        
        # Инициализируем процессор
        self.processor = NDVIProcessor2(
            b4_dir=str(self.b4_dir),
            b5_dir=str(self.b5_dir),
            output_dir=str(self.output_dir),
            chunk_size=256
        )
    
    def tearDown(self):
        # Удаляем временные директории
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Создание тестовых данных"""
        # Создаем директории
        self.b4_dir.mkdir(parents=True)
        self.b5_dir.mkdir(parents=True)
        
        # Создаем тестовые массивы
        data = np.random.rand(100, 100)
        
        # Сохраняем тестовые файлы
        for i in range(2):
            b4_data = xr.DataArray(data, dims=['y', 'x'])
            b5_data = xr.DataArray(data, dims=['y', 'x'])
            
            b4_data.to_netcdf(self.b4_dir / f"test_b4_{i}.nc")
            b5_data.to_netcdf(self.b5_dir / f"test_b5_{i}.nc")
    
    def test_merge_bands(self):
        """Тест объединения слоев"""
        self.processor.merge_bands()
        self.assertIsNotNone(self.processor.b4_merged)
        self.assertIsNotNone(self.processor.b5_merged)
    
    def test_calculate_ndvi(self):
        """Тест расчета NDVI"""
        self.processor.merge_bands()
        ndvi = self.processor.calculate_ndvi()
        self.assertIsNotNone(ndvi)
        self.assertTrue(np.all(ndvi.values >= -1) and np.all(ndvi.values <= 1))
    
    def test_process_chunk(self):
        """Тест обработки отдельного чанка"""
        # Создаем тестовые чанки
        b4_chunk = xr.DataArray(np.ones((10, 10)), dims=['y', 'x'])
        b5_chunk = xr.DataArray(np.ones((10, 10)), dims=['y', 'x'])
        
        # Обрабатываем чанк
        result = self.processor._process_chunk(0, 10, 0, 10, b4_chunk, b5_chunk)
        
        # Проверяем результат
        self.assertEqual(result.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_analyze_ndvi(self):
        """Тест анализа NDVI"""
        # Создаем тестовые данные NDVI
        ndvi_data = np.random.uniform(-1, 1, (100, 100))
        ndvi = xr.DataArray(ndvi_data, dims=['y', 'x'])
        
        # Анализируем данные
        self.processor.analyze_ndvi(ndvi)
        
        # Проверяем, что файл статистики создан
        stats_file = self.output_dir / "ndvi_stats.json"
        self.assertTrue(stats_file.exists())

if __name__ == '__main__':
    unittest.main() 