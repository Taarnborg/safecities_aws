import pandas as pd

def remove_invalid_inputs(dataset,text_column):
    dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
    return dataset.loc[dataset.valid]