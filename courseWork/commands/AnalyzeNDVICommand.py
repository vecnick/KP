import json
import numpy as np
import rasterio
from pathlib import Path
from courseWork.commands.Command import Command


class AnalyzeNDVICommand(Command):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def execute(self, processor):
        if not processor.scene or not processor.scene.ndvi_path:
            raise ValueError("NDVI result not available")

        processor.logger.info("Analyzing NDVI data...")

        # Используем выборку для анализа больших данных
        sample_size = 10000
        with rasterio.open(processor.scene.ndvi_path) as src:
            # Случайная выборка пикселей
            data = src.read(1, out_shape=(1, int(src.height * 0.1), int(src.width * 0.1)))
            valid_data = data[~np.isnan(data) & (data != src.nodata)]

            if valid_data.size == 0:
                processor.logger.error("No valid NDVI values")
                return

            sample = np.random.choice(valid_data, size=min(sample_size, valid_data.size), replace=False)

            stats = {
                'min': float(np.min(sample)),
                'max': float(np.max(sample)),
                'mean': float(np.mean(sample)),
                'median': float(np.median(sample)),
                'std': float(np.std(sample))
            }

        # Сохранение статистики
        stats_file = self.output_dir / "ndvi_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)