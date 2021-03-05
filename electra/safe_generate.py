import os
import json
import torch
import csv

from io import StringIO
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model_def import ElectraClassifier

MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = "KB/electra-base-swedish-cased-discriminator"

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=True)

JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElectraClassifier(PRE_TRAINED_MODEL_NAME, 2)
    model.load_state_dict(torch.load(model_dir + '/pytorch_model.bin', map_location=torch.device('cpu')))    
    return model.to(device)

def input_fn(serialized_input_data, request_content_type):
    #print('STARTED input_fn')
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(serialized_input_data)                        
        tokenized_text = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=False, max_length=MAX_LEN)
        input_ids_text = tokenized_text['input_ids']
        attention_mask_text = tokenized_text['attention_mask']
        return input_ids_text, attention_mask_text
    elif request_content_type == 'text/csv':
        data_list = serialized_input_data.split('\t')
        id_val = data_list[0]
        text = data_list[1]
        tokenized_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)        
        input_ids_text = tokenized_text['input_ids']
        attention_mask_text = tokenized_text['attention_mask']

        return id_val, input_ids_text, attention_mask_text
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    #print('STARTED output_fn')
    #logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    #print('STARTED predict_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    id_val, input_ids_text, attention_mask_text = input_data
    input_ids_text = input_ids_text.to(device)
    attention_mask_text = attention_mask_text.to(device)
    with torch.no_grad():
        y = model(input_ids_text,attention_mask_text)
        probs = y.softmax(1).tolist()
    return id_val, probs

