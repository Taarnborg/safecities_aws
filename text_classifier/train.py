import argparse
import logging
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch_optimizer as optim
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
# from datasets import load_dataset, load_metric
from torch.utils.data import RandomSampler, DataLoader

# Network definition
from data_prep import CustomDataset
from model_def import CNNClassifier

# Utils
from utils import remove_invalid_inputs,BertEncoderFilter

f = BertEncoderFilter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addFilter(f)

TRAIN = 'hateful_70.csv'
VALID = 'hateful_10.csv'
TEST = 'hateful_20.csv'
WEIGHTS_NAME = "pytorch_model.bin" # this comes from transformers.file_utils
MAX_LEN = 512
PRETRAINED_MODEL_NAME = 'KB/bert-base-swedish-cased'
# PRETRAINED_MODEL_NAME = 'prajjwal1/bert-tiny'

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)

def _get_train_data_loader(batch_size, data_dir):
    dataset = pd.read_csv(os.path.join(args.data_dir, TRAIN), sep='\t', names = ['targets', 'text'])
    dataset = remove_invalid_inputs(dataset,'text')

    train_data = CustomDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=MAX_LEN
                    )

    #Maybe use different sampler???
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers = args.num_cpus, pin_memory=True)
    return(train_dataloader)

def _get_eval_data_loader(batch_size, data_dir):
    dataset = pd.read_csv(os.path.join(args.data_dir, VALID), sep='\t', names = ['targets', 'text'])
    dataset = remove_invalid_inputs(dataset,'text')
    train_data = CustomDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=MAX_LEN
                    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers = args.num_cpus, pin_memory=True)
    return(train_dataloader)

def save_model(model_to_save,save_directory):
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    if args.num_gpus > 1:
        model_to_save = model_to_save.module

    state_dict = model_to_save.state_dict()
    torch.save(state_dict, output_model_file)

def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    print(device)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    model = CNNClassifier(PRETRAINED_MODEL_NAME,args.num_labels,MAX_LEN)
    
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

        if args.num_gpus > 1:
            model = torch.nn.DataParallel(model)
            print('data parallel model')

        model.cuda()

    # Maybe use different optimizer????
    optimizer = optim.Lamb(
            model.parameters(), 
            lr = args.lr, 
            betas=(0.9, 0.999), 
            eps=args.epsilon, 
            weight_decay=args.weight_decay)

    # Maybe use different loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()

        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            outputs = model(b_input_ids,attention_mask=b_input_mask)

            loss = loss_fn(outputs, b_labels)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ?????
            optimizer.step()
            optimizer.zero_grad()

            if args.verbose:
                if step % 100 == 0:
                    print('Batch', step)


    eval_loader = _get_eval_data_loader(args.test_batch_size, args.data_dir)        
    test(model, eval_loader, device)

    save_model(model, args.model_dir)

def test(model, eval_loader, device):
    model.eval()
    predicted_classes = torch.empty(0).to(device)
    labels = torch.empty(0).to(device)

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            outputs = model(b_input_ids,attention_mask=b_input_mask)
            _, preds = torch.max(outputs, dim=1)


            predicted_classes = torch.cat((predicted_classes, preds))
            labels = torch.cat((labels, b_labels))

    predicted_classes = predicted_classes.to('cpu')
    labels = labels.to('cpu')

    print("confusion matrix:")
    print(confusion_matrix(labels, predicted_classes))
    print('F1 score:', f1_score(labels, predicted_classes))
    print('Precision score:', precision_score(labels, predicted_classes))
    print('Recall score:', recall_score(labels, predicted_classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--num-labels", type=int, default=2, metavar="N", help="Number of labels."
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, metavar="N", help="input batch size for training (default: 16)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=8, metavar="N", help="input batch size for testing (default: 8)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="M", help="weight_decay (default: 0.01)")
    parser.add_argument("--seed", type=int, default=43, metavar="S", help="random seed (default: 43)")
    parser.add_argument("--epsilon", type=int, default=1e-8, metavar="EP", help="random seed (default: 1e-8)")
    parser.add_argument("--frozen_layers", type=int, default=10, metavar="NL", help="number of frozen layers(default: 10)")
    parser.add_argument('--verbose', default=True,help='For displaying logs')

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])
    # parser.add_argument("--num-gpus", type=int, default=False)
    # parser.add_argument("--num-cpus", type=int, default=False)

    args = parser.parse_args()
    
    train(args)
