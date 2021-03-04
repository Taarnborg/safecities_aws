from transformers import ElectraModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class ElectraClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(pretrained_model_name)
        self.dense = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra.config.hidden_size, self.num_labels)

    def classifier(self,sequence_output):
        x = sequence_output[:, 0, :]
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

    def forward(self, input_ids=None,attention_mask=None):
        discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        return logits

class ElectraWithContextClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(ElectraWithContextClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra_text = ElectraModel.from_pretrained(pretrained_model_name,output_hidden_states=True)
        self.electra_context = ElectraModel.from_pretrained(pretrained_model_name,output_hidden_states=True)

        self.avg_pool = nn.AvgPool1d(2, 2)
        self.dense = nn.Linear(self.electra_text.config.hidden_size, self.electra_text.config.hidden_size)
        self.dropout = nn.Dropout(self.electra_text.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra_text.config.hidden_size, self.num_labels)

    def process_hidden_states(self,hs_text,hs_context):

        # take mean of all hidden layers. unsqueeze it, and reduce the len of the embedding with AvgPool1d with kernel_size and stride as 2.

        hs_text = self.avg_pool(hs_text.mean(dim=1).unsqueeze(dim=1)).squeeze(dim=1)
        hs_context = self.avg_pool(hs_context.mean(dim=1).unsqueeze(dim=1)).squeeze(dim=1)
        # concat the layers
        return torch.cat((hs_text,hs_context),dim=1)

    def classifier(self,x):
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

    def forward(self, input_ids_text=None,attention_mask_text=None,input_ids_context=None,attention_mask_context=None):

        hs_text = self.electra_text(input_ids=input_ids_text,attention_mask=attention_mask_text)
        print(hs_text[0])


        raise(ValueError)
        hs_text = self.electra_text(input_ids=input_ids_text,attention_mask=attention_mask_text).hidden_states[0]
        hs_context = self.electra_text(input_ids=input_ids_context,attention_mask=attention_mask_context).hidden_states[0]
        hs_combined = self.process_hidden_states(hs_text,hs_context)
        logits = self.classifier(hs_combined)
        return logits