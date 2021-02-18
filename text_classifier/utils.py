import pandas as pd

def remove_invalid_inputs(dataset,text_column):
    dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
    return dataset.loc[dataset.valid]

def save_model(model_to_save,output_model_file):
    state_dict = model_to_save.state_dict()
    torch.save(state_dict, output_model_file)