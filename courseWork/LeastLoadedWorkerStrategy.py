from courseWork.WorkerSelectionStrategy import WorkerSelectionStrategy


class LeastLoadedWorkerStrategy(WorkerSelectionStrategy):
    def select_worker(self, processor: 'NDVIProcessor', chunk_id: str) -> str:
        workers_info = processor.client.scheduler_info()['workers']
        if not workers_info:
            raise RuntimeError("No workers available")

        worker_load = {}
        for addr, info in workers_info.items():
            n_tasks = len(info.get('processing', {}))
            worker_load[addr] = n_tasks

        selected_worker = min(worker_load, key=worker_load.get)
        processor.logger.info(f"Selected worker {selected_worker} for chunk {chunk_id}")
        return selected_worker