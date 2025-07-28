# MergeBandsCommand.py
import glob
import os
import subprocess
from pathlib import Path

import rasterio

from courseWork.Scene import Scene
from courseWork.commands.Command import Command

class MergeBandsCommand(Command):
    def __init__(self, b4_dir: str, b5_dir: str, output_dir: Path):
        self.b4_dir = Path(b4_dir)
        self.b5_dir = Path(b5_dir)
        self.output_dir = output_dir

    def execute(self, processor):
        processor.logger.info("Merging bands using VRT...")

        b4_files = sorted(glob.glob(os.path.join(self.b4_dir, "*.TIF")))
        b5_files = sorted(glob.glob(os.path.join(self.b5_dir, "*.TIF")))

        if not b4_files or not b5_files:
            raise ValueError("B4 or B5 files not found")

        # Создаем VRT-файлы
        b4_vrt_path = self.output_dir / "b4.vrt"
        b5_vrt_path = self.output_dir / "b5.vrt"

        try:
            subprocess.run(["gdalbuildvrt", str(b4_vrt_path)] + b4_files, check=True)
            subprocess.run(["gdalbuildvrt", str(b5_vrt_path)] + b5_files, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"VRT creation failed: {str(e)}")

        # Получаем метаданные
        with rasterio.open(b4_vrt_path) as src:
            crs = src.crs
            transform = src.transform

        # Создаем сцену
        scene = Scene(str(b4_vrt_path), str(b5_vrt_path), crs, transform)
        processor.scene = scene