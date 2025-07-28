from typing import List, Dict

from courseWork.commands.Command import Command


class SceneBuilder:
    def __init__(self):
        self.global_commands: List[Command] = []
        self.chunk_commands: Dict[str, List[Command]] = {}

    def add_global_command(self, command: Command):
        """Добавляет глобальную команду"""
        self.global_commands.append(command)
        return self

    def add_chunk_command(self, command: Command, chunk_id: str):
        """Добавляет команду для конкретного чанка"""
        if chunk_id not in self.chunk_commands:
            self.chunk_commands[chunk_id] = []
        self.chunk_commands[chunk_id].append(command)
        return self

    def execute_global_commands(self, processor):
        """Выполняет все глобальные команды"""
        for command in self.global_commands:
            command.execute(processor)

    def execute_chunk_commands(self, chunk):
        """Выполняет команды для конкретного чанка"""
        if chunk.id in self.chunk_commands:
            for command in self.chunk_commands[chunk.id]:
                command.execute(chunk)