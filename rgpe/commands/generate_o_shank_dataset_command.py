from . import Command
from ..services import o_shank_dataset_generator_service, base_dataset_generator_service


class GenerateOShankDatasetCommand(Command):
    def execute(self, *args):
        base_dataset_generator_service.generate_gram_points_dataset()
        o_shank_dataset_generator_service.generate_o_shank_dataset()
