
# generate hERG dataset
from utils import build_dataset
args={}
args['input_csv'] = '../data/Hepatotoxicity.csv'
args['output_bin'] = '../data/Hepatotoxicity.bin'
args['output_csv'] = '../data/Hepatotoxicity_group.csv'

build_dataset.built_data_and_save_for_splited(
        origin_path=args['input_csv'],
        save_path=args['output_bin'],
        group_path=args['output_csv'],
        task_list_selected=None
         )






