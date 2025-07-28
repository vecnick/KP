from abc import abstractmethod, ABC

class WorkerSelectionStrategy(ABC):
    @abstractmethod
    def select_worker(self, processor: 'NDVIProcessor', chunk_id: str) -> str:
        pass