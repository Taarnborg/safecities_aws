import os
import json
import torch
import csv

from io import StringIO
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model_def import ElectraWithContextClassifier

MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = "Maltehb/-l-ctra-danish-electra-small-uncased"
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=True)
JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    print('model_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElectraWithContextClassifier(PRE_TRAINED_MODEL_NAME, 2)
    model.load_state_dict(torch.load(model_dir + '/pytorch_model.bin', map_location=torch.device('cpu')))
    return model.to(device)

def input_fn(serialized_input_data, request_content_type):
    print('input_fn')

    if request_content_type == "application/json":
        data = json.loads(serialized_input_data)     
        print(data)           
        tokenized_text = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=False, max_length=MAX_LEN)
        tokenized_context = tokenizer(data['context'], return_tensors='pt', padding=True, truncation=False, max_length=MAX_LEN)
        input_ids_text = tokenized_text['input_ids']
        attention_mask_text = tokenized_text['attention_mask']
        input_ids_context = tokenized_context['input_ids']
        attention_mask_context = tokenized_context['attention_mask']
        return input_ids_text, attention_mask_text, input_ids_context, attention_mask_context
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    print('output_fn')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('predict_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    id_val, input_ids_text, attention_mask_text, input_ids_context, attention_mask_context = input_data
    input_ids_text = input_ids_text.to(device)
    attention_mask_text = attention_mask_text.to(device)
    input_ids_context = input_ids_context.to(device)
    attention_mask_context = attention_mask_context.to(device)
    print(id_val)
    with torch.no_grad():
        y = model(input_ids_text, attention_mask_text=attention_mask_text, input_ids_context=input_ids_context, attention_mask_context=attention_mask_context)
        probs = y.softmax(1).tolist()
    return id_val, probs

