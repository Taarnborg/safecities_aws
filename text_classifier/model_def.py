from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

PRETRAINED_MODEL_NAME = 'KB/bert-base-swedish-cased'
class KimCNN(nn.Module):
    
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(KimCNN, self).__init__()        
        V = embed_num
        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes
        
        self.static = static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.static:
            x = Variable(x)        
        x = x.unsqueeze(1)  # (N, Ci, W, D)        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)        
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output


class TestClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TestClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.bert.config.hidden_size, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, n_classes),
            )
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        return self.classifier(output.pooler_output)

class CNNClassifier(nn.Module):
    def __init__(self, n_classes):
        super(CNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

        STATIC = True
        MAX_LEN = 512
        HIDDEN_SIZE = 768
        N_KERNELS = 3
        KERENL_SIZES = [2,3,4]
        DROPOUT = 0.2
        N_CLASSES = 2

        self.static = STATIC
        self.embed = nn.Embedding(MAX_LEN, HIDDEN_SIZE)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, N_KERNELS, (KS, HIDDEN_SIZE)) for KS in KERENL_SIZES])
        self.dropout = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(len(KERENL_SIZES) * N_KERNELS, N_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def classifier(self,x):

        if self.static:
            x = Variable(x)        
        x = x.unsqueeze(1)  # (N, Ci, W, D)        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)        
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output
       
    def forward(self, input_ids, attention_mask):
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        return self.classifier(output.pooler_output)
