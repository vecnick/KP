from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NDVIConfig:
    # Размеры чанков
    CHUNK_SIZE: int = 512
    
    # Пороговые значения для NDVI
    HIGH_NDVI_THRESHOLD: float = 0.8
    VALID_PIXEL_THRESHOLD: float = 0.01
    
    # Настройки Dask
    DASK_CONFIG: Dict[str, Any] = {
        'array.chunk-size': '32MiB',
        'distributed.worker.memory.target': 0.6,
        'distributed.worker.memory.spill': 0.7,
        'distributed.worker.memory.pause': 0.8,
        'distributed.worker.memory.terminate': 0.95
    }
    
    # Настройки визуализации
    VISUALIZATION_CONFIG: Dict[str, Any] = {
        'target_size': 1200,
        'dpi': 300,
        'ndvi_range': (-0.1, 0.6)
    }
    
    # Настройки мониторинга
    MONITORING_INTERVAL: float = 5.0  # секунды
    
    # Цветовая карта для NDVI
    NDVI_COLORS: list = [
        (0.0, 0.0, 0.5),    # Темно-синий (вода, NDVI < 0)
        (0.0, 0.0, 1.0),    # Синий (вода, NDVI около 0)
        (0.5, 0.5, 0.5),    # Серый (голая почва, NDVI около 0)
        (0.8, 0.8, 0.0),    # Желтый (редкая растительность)
        (0.0, 0.8, 0.0),    # Зеленый (умеренная растительность)
        (0.0, 0.6, 0.0),    # Темно-зеленый (густая растительность)
        (0.0, 0.4, 0.0)     # Очень темно-зеленый (очень густая растительность)
    ] 