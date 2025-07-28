import subprocess
from pathlib import Path
from courseWork.Scene import Scene
from courseWork.commands.Command import Command


class AssembleNDVICommand(Command):
    def __init__(self, output_dir: Path, chunk_files: dict):
        self.output_dir = output_dir
        self.chunk_files = chunk_files

    def execute(self, processor):
        processor.logger.info("Assembling final NDVI using VRT...")

        if not self.chunk_files:
            raise ValueError("No chunk files available for assembly")

        # Создаем VRT из всех чанков
        vrt_path = self.output_dir / "ndvi_result.vrt"
        chunk_files_list = list(self.chunk_files.values())

        try:
            subprocess.run(["gdalbuildvrt", str(vrt_path)] + chunk_files_list, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"NDVI assembly failed: {str(e)}")

        # Сохраняем путь к результату
        processor.scene.ndvi_path = str(vrt_path)
        processor.logger.info(f"NDVI result assembled at {vrt_path}")