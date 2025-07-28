import rasterio

from courseWork.Chunk import Chunk
from courseWork.commands.Command import Command

class CreateChunksCommand(Command):
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def execute(self, processor):
        processor.logger.info("Creating chunks...")

        if not processor.scene:
            raise ValueError("Scene not initialized")

        # Получаем размеры данных из VRT
        with rasterio.open(processor.scene.b4_path) as src:
            height, width = src.shape

        chunk_coords = []
        for y_start in range(0, height, self.chunk_size):
            y_end = min(y_start + self.chunk_size, height)
            for x_start in range(0, width, self.chunk_size):
                x_end = min(x_start + self.chunk_size, width)
                chunk_coords.append((y_start, y_end, x_start, x_end))

        for i, coords in enumerate(chunk_coords):
            chunk_id = f"chunk_{i}"
            worker = processor.worker_strategy.select_worker(processor, chunk_id)
            chunk = Chunk(
                id=chunk_id,
                coords=coords,
                worker=worker
            )
            processor.scene.add_chunk(chunk)
            processor.chunk_status[chunk_id] = {
                'status': 'pending',
                'worker': worker
            }