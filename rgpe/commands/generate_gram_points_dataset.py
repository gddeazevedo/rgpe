from . import Command
from ..services import base_dataset_generator_service


class GenerateGramPointsDatasetCommand(Command):
    def execute(self, *args):
        base_dataset_generator_service.generate_gram_points_dataset()
