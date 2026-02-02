from . import Command
from ..services import base_dataset_generator_service


class GenerateCogramPointsDatasetCommand(Command):
    def execute(self, *args):
        base_dataset_generator_service.generate_cogram_points_dataset()
