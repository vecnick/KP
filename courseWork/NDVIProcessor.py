# NDVIProcessor.py
import os
import logging
import psutil
import json
from dask.distributed import Client, LocalCluster
import dask
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from courseWork.Chunk import Chunk
from courseWork.ChunkProcessor import ChunkProcessor
from courseWork.LeastLoadedWorkerStrategy import LeastLoadedWorkerStrategy
from courseWork.RandomWorkerStrategy import RandomWorkerStrategy
from courseWork.SceneBuilder import SceneBuilder
from courseWork.WorkerSelectionStrategy import WorkerSelectionStrategy
from courseWork.commands.AnalyzeNDVICommand import AnalyzeNDVICommand
from courseWork.commands.AssembleNDVICommand import AssembleNDVICommand
from courseWork.commands.VisualizeNDVICommand import VisualizeNDVICommand
from courseWork.commands.CreateChunksCommand import CreateChunksCommand
from courseWork.commands.MergeBandsCommand import MergeBandsCommand


class NDVIProcessor:
    def __init__(self, output_dir: str, worker_strategy: WorkerSelectionStrategy = None):
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Инициализация логгера
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )

        # Инициализация Dask
        memory_limit = psutil.virtual_memory().available * 0.8
        n_workers = max(1, psutil.cpu_count() - 1)

        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit // n_workers,
            silence_logs=logging.ERROR
        )
        self.client = Client(self.cluster)

        # Настройка Dask
        dask.config.set({
            'array.chunk-size': '32MiB',
            'distributed.worker.memory.target': 0.6,
            'distributed.worker.memory.spill': 0.7,
            'distributed.worker.memory.pause': 0.8,
            'distributed.worker.memory.terminate': 0.95
        })

        # Состояние обработки
        self.scene: Optional['Scene'] = None
        self.chunk_status: Dict[str, Dict] = {}
        self.worker_strategy = worker_strategy or LeastLoadedWorkerStrategy()
        self.chunk_files: Dict[str, str] = {}

    def update_chunk_status(self, chunk_id: str, status: str):
        """Обновляет статус обработки чанка"""
        if chunk_id in self.chunk_status:
            self.chunk_status[chunk_id]['status'] = status
            if status == 'started':
                self.chunk_status[chunk_id]['start_time'] = datetime.now().isoformat()
            elif status in ['completed', 'failed']:
                self.chunk_status[chunk_id]['end_time'] = datetime.now().isoformat()

            # Сохранение статуса в файл
            status_file = self.output_dir / 'chunk_status.json'
            with status_file.open('w') as f:
                json.dump(self.chunk_status, f, indent=4)

    def process(self, builder: 'SceneBuilder'):
        """Выполняет обработку сцены с использованием построителя"""
        try:
            # Выполняем глобальные команды
            builder.execute_global_commands(self)

            # Асинхронная обработка чанков
            futures = []
            for chunk_id, chunk in self.scene.chunks.items():
                # Создаем обработчик для каждого чанка
                chunk_processor = ChunkProcessor(
                    self.scene.b4_path,
                    self.scene.b5_path,
                    self.scene.crs,
                    self.scene.transform,
                    self.output_dir
                )

                future = self.client.submit(
                    chunk_processor.process,
                    chunk.coords,
                    chunk_id,
                    workers=[chunk.worker],
                    pure=False
                )
                futures.append(future)
                self.update_chunk_status(chunk_id, 'started')

            # Собираем результаты
            for future in futures:
                chunk_id, filename = future.result()
                self.chunk_files[chunk_id] = filename
                self.update_chunk_status(chunk_id, 'completed')

            # Сборка финального результата
            final_builder = SceneBuilder()
            final_builder.add_global_command(AssembleNDVICommand(self.output_dir, self.chunk_files))
            final_builder.add_global_command(VisualizeNDVICommand(self.output_dir))
            final_builder.add_global_command(AnalyzeNDVICommand(self.output_dir))
            final_builder.execute_global_commands(self)

            self.logger.info("Processing completed successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Processing failed: {str(e)}")
            return False
        finally:
            self.client.close()
            self.cluster.close()


if __name__ == '__main__':
    processor = NDVIProcessor(
        output_dir="ndvi_processor/output",
        worker_strategy=RandomWorkerStrategy()
    )

    scene_builder = SceneBuilder()
    scene_builder.add_global_command(MergeBandsCommand("./b4", "./b5", processor.output_dir))
    scene_builder.add_global_command(CreateChunksCommand(chunk_size=512))

    processor.process(scene_builder)