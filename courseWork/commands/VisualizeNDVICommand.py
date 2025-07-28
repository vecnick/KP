import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from courseWork.commands.Command import Command


class VisualizeNDVICommand(Command):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def execute(self, processor):
        if not processor.scene or not processor.scene.ndvi_path:
            raise ValueError("NDVI result not available")

        processor.logger.info("Creating visualizations...")

        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Чтение уменьшенной версии
        with rasterio.open(processor.scene.ndvi_path) as src:
            data = src.read(1, out_shape=(1, int(src.height / 4), int(src.width / 4)))
            transform = src.transform * src.transform.scale(4, 4)

        # Визуализация
        colors = [
            (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.5, 0.5, 0.5),
            (0.8, 0.8, 0.0), (0.0, 0.8, 0.0), (0.0, 0.6, 0.0), (0.0, 0.4, 0.0)
        ]
        ndvi_cmap = LinearSegmentedColormap.from_list('ndvi_cmap', colors, N=256)

        plt.figure(figsize=(15, 15))
        plt.imshow(data, cmap=ndvi_cmap, vmin=-0.1, vmax=0.6)
        plt.colorbar(label='NDVI')
        plt.title('NDVI Map')
        plt.axis('off')
        plt.savefig(vis_dir / "ndvi_map.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Гистограмма
        valid_data = data[~np.isnan(data)]
        plt.figure(figsize=(10, 6))
        plt.hist(valid_data, bins=100, range=(-0.1, 1.0))
        plt.title('NDVI Distribution')
        plt.xlabel('NDVI')
        plt.ylabel('Frequency')
        plt.savefig(vis_dir / "ndvi_histogram.png")
        plt.close()