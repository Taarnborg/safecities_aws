from transformers import ElectraModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

def freeze(model,n_layers_to_freeze=10):

    modules = [model.embeddings, *model.encoder.layer[:n_layers_to_freeze]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

class ElectraClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(pretrained_model_name)

        self.dense = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra.config.hidden_size, self.num_labels)

    def forward(self, input_ids=None,attention_mask=None,labels=None):
        discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]
        x = sequence_output[:, 0, :]

        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        # x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        # x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        # x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        logits = self.out_proj(x)

        loss = None
        if labels is not None:
            if self.num_labels > 1:
                self.loss_fn = nn.CrossEntropyLoss()
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                # loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )