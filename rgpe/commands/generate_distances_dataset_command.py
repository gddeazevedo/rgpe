from . import Command
from ..services import base_dataset_generator_service


class GenerateDistancesDatasetCommand(Command):
    def execute(self, *args):
        base_dataset_generator_service.generate_distances_dataset()
