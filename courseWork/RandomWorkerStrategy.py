from courseWork.WorkerSelectionStrategy import WorkerSelectionStrategy
import random

class RandomWorkerStrategy(WorkerSelectionStrategy):
    def select_worker(self, processor: 'NDVIProcessor', chunk_id: str) -> str:
        workers = list(processor.client.scheduler_info()['workers'].keys())
        if not workers:
            raise RuntimeError("No workers available")

        selected_worker = random.choice(workers)
        processor.logger.info(f"Randomly selected worker {selected_worker} for chunk {chunk_id}")
        return selected_worker