import os

base_dir = os.getenv('base_dir')

data_dir = f'{base_dir}/dataset'

models = f'{base_dir}/models'

def get_input_data_dir (folder : str , file : str) -> str :
    folder_str = os.path.join(data_dir , folder)
    return os.path.join(folder_str , file)


# print(get_input_data_dir('agnews' , 'label_names.txt'))
# print(data_dir)