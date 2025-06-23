import numpy as np
import xarray as xr
from pathlib import Path
import rasterio
from rasterio.transform import from_origin

def create_test_data():
    # Создаем тестовые директории
    b4_dir = Path("ndvi_processor/b4")
    b5_dir = Path("ndvi_processor/b5")
    b4_dir.mkdir(parents=True, exist_ok=True)
    b5_dir.mkdir(parents=True, exist_ok=True)
    
    # Параметры тестовых данных
    height = 1000
    width = 1000
    transform = from_origin(0, height, 1, 1)
    
    # Создаем тестовые массивы
    for i in range(2):
        # Создаем случайные данные с нормальным распределением
        b4_data = np.random.normal(0.3, 0.1, (height, width))
        b5_data = np.random.normal(0.5, 0.1, (height, width))
        
        # Сохраняем B4
        with rasterio.open(
            b4_dir / f"test_b4_{i}.tif",
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=b4_data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(b4_data, 1)
        
        # Сохраняем B5
        with rasterio.open(
            b5_dir / f"test_b5_{i}.tif",
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=b5_data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(b5_data, 1)
        
        print(f"Created test data {i+1}/2")

if __name__ == "__main__":
    create_test_data() 