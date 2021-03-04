import pandas as pd
import torch
from torch.utils.data import Dataset,RandomSampler,DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self,text,targets,tokenizer,max_len):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        target = self.targets[item]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long),
        }

class WithContextDataset(Dataset):
    def __init__(self,text,context,targets,tokenizer,max_len):
        self.text = text
        self.context = context
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        context = self.context[item]
        target = self.targets[item]

        text_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
        context_encoding = self.tokenizer(
            context,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
    
        return {
          'text': text,
          'context':context,
          'input_ids_text': text_encoding['input_ids'].flatten(),
          'attention_mask_text': text_encoding['attention_mask'].flatten(),
          'input_ids_context': context_encoding['input_ids'].flatten(),
          'attention_mask_context': context_encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long),
        }

def get_data_loader(path,tokenizer,max_len,batch_size,num_workers):
    # dataset = pd.read_csv(path, sep='\t', names=['targets', 'text'])
    dataset = pd.read_csv(path, sep='\t', names = ['targets', 'text', 'origin', 'main_text', 'secondary_text'])
    dataset = remove_invalid_inputs(dataset,'text')

    data = CustomDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                    )

    sampler = RandomSampler(data)
    dataloader = DataLoader(data,batch_size=batch_size,sampler=sampler,num_workers=num_workers,pin_memory=True)
    return dataloader,data


def get_data_with_context_loader(path,tokenizer,max_len,batch_size,num_workers):
    dataset = pd.read_csv(path, sep='\t', names = ['targets', 'text', 'origin', 'main_text', 'secondary_text'])
    dataset = remove_invalid_inputs(dataset,'text')
    dataset = remove_invalid_inputs(dataset,'main_text')

    data = WithContextDataset(
                    text=dataset.text.to_numpy(),
                    context=dataset.main_text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                    )

    sampler = RandomSampler(data)
    dataloader = DataLoader(data,batch_size=batch_size,sampler=sampler,num_workers=num_workers,pin_memory=True)
    return dataloader,data

def remove_invalid_inputs(dataset,text_column):
    'Simpel metode til at fjerne alle rækker fra en dataframe, baseret på om værdierne i en kolonne er af typen str'
    dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
    return dataset.loc[dataset.valid]