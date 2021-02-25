import os
import pandas as pd
import torch
from torch.utils.data import RandomSampler, DataLoader
from data_prep import CustomDataset


def remove_invalid_inputs(dataset,text_column):
    'Simpel metode til at fjerne alle rækker fra en dataframe, baseret på om værdierne i en kolonne er af typen str'
    dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
    return dataset.loc[dataset.valid]

def save_model(model_to_save,save_directory,num_gpus=0):
    'Metode til at gemme en pytorch model, der tager højde for om modelen trænes i parallel eller ej'
    WEIGHTS_NAME = "pytorch_model.bin" # this comes from transformers.file_utils
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    if num_gpus > 1:
        model_to_save = model_to_save.module

    state_dict = model_to_save.state_dict()
    torch.save(state_dict, output_model_file)

def get_data_loader(path,tokenizer,max_len,batch_size,num_cpus):
    dataset = pd.read_csv(path, sep='\t', names = ['targets', 'text'])
    dataset = remove_invalid_inputs(dataset,'text')

    data = CustomDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                    )

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers = num_cpus, pin_memory=True)
    return dataloader,data