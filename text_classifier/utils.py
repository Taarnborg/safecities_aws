import pandas as pd
import logging

def remove_invalid_inputs(dataset,text_column):
    dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
    return dataset.loc[dataset.valid]

class BertEncoderFilter(logging.Filter):

    def filter(self, record):

        if "module_name:module.bert.encoder" in record:
            return True
        else:
            return False
