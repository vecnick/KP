# Scene.py
from typing import Dict, Optional

class Scene:
    def __init__(self, b4_path: str, b5_path: str, crs: any, transform: any):
        self.b4_path = b4_path
        self.b5_path = b5_path
        self.crs = crs
        self.transform = transform
        self.chunks: Dict[str, 'Chunk'] = {}
        self.ndvi_path: Optional[str] = None

    def add_chunk(self, chunk: 'Chunk'):
        self.chunks[chunk.id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional['Chunk']:
        return self.chunks.get(chunk_id)