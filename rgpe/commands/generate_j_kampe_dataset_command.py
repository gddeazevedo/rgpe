from multiprocessing import Process
from . import Command
from ..services import j_kampe_dataset_generator_service, base_dataset_generator_service


class GenerateJKampeDatasetCommand(Command):
    def execute(self, *args):
        p_gram = Process(target=base_dataset_generator_service.generate_gram_points_dataset)
        p_cogram = Process(target=base_dataset_generator_service.generate_cogram_points_dataset)
        p_gram.start()
        p_cogram.start()
        p_gram.join()
        p_cogram.join()
        base_dataset_generator_service.generate_distances_dataset()
        j_kampe_dataset_generator_service.generate_j_kampe_dataset()
