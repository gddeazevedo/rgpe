from ..commands import Command
from ..commands.demo_command import DemoCommand
from ..commands.generate_all_datasets_command import GenerateAllDatasetsCommand
from ..commands.generate_custom_dataset_command import GenerateCustomDatasetCommand
from ..commands.generate_o_shank_dataset_command import GenerateOShankDatasetCommand
from ..commands.generate_j_kampe_dataset_command import GenerateJKampeDatasetCommand
from ..commands.generate_gram_points_dataset import GenerateGramPointsDatasetCommand
from ..commands.generate_cogram_points_dataset_command import GenerateCogramPointsDatasetCommand
from ..commands.generate_distances_dataset_command import GenerateDistancesDatasetCommand


COMMANDS: dict[str, type[Command]] = {
    'demo': DemoCommand,
    'generate-datasets': GenerateAllDatasetsCommand,
    'generate-custom-dataset': GenerateCustomDatasetCommand,
    'generate-o-shank-dataset': GenerateOShankDatasetCommand,
    'generate-j-kampe-dataset': GenerateJKampeDatasetCommand,
    'generate-gram-points-dataset': GenerateGramPointsDatasetCommand,
    'generate-cogram-points-dataset': GenerateCogramPointsDatasetCommand,
    'generate-distances-dataset': GenerateDistancesDatasetCommand,
}
