import argparse
import logging
import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from model_def import ElectraClassifier
from utils import save_model
from data_prep import get_data_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(gpu,args):

    rank = args.num_nodes * args.num_gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
    	world_size=args.world_size,                              
    	rank=rank                                               
    )    

    model = ElectraClassifier(args.model_checkpoint,args.num_labels)


    torch.manual_seed(args.seed)
    model = ElectraClassifier(args.model_checkpoint,args.num_labels)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

    # tokenizer,dataloader and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    train_path = os.path.join(args.data_dir,args.train)
    num_workers = args.num_gpus * 4
    train_loader,train_data = get_data_loader(train_path,tokenizer,args.max_len,args.batch_size,num_workers)

    # Setting the optimizer (Important that this is done after, and not before, moving the model to cuda)
    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = args.lr, 
            eps = args.epsilon,
            weight_decay=args.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss().cuda(gpu)

    # Train
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0
        correct = 0
        print('Epoch', epoch)
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            logits = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(logits.view(-1, args.num_labels), b_labels.view(-1))
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            running_loss += loss.item() * b_input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == b_labels).sum().item()

        running_loss = running_loss/train_data.__len__()
        running_accuracy = 100*(correct/train_data.__len__())
        print('Running loss', running_loss)
        print('Running accuracy', running_accuracy)

    # Test on eval data
    eval_path = os.path.join(args.data_dir,args.valid)
    eval_loader,valid_data = get_data_loader(eval_path,tokenizer,args.max_len,args.test_batch_size,num_workers)
    test(model, eval_loader, device)

    ## save model
    if args.save_model:
        save_model(model, args.model_dir,args.num_gpus)

def test(model, eval_loader, device):
    model.eval()
    predicted_classes = torch.empty(0).to(device)
    labels = torch.empty(0).to(device)

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            logits = model(b_input_ids,b_input_mask)
            _,preds = torch.max(logits, dim=1)

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
    parser.add_argument("--model-checkpoint", type=str, default='Maltehb/-l-ctra-danish-electra-small-cased', help="name of pretrained model from huggingface model hub")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--train", type=str, default='train.csv')
    parser.add_argument("--valid", type=str, default='valid.csv')
    parser.add_argument("--test", type=str, default='test.csv')
    # Hyperparams
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epsilon", type=int, default=1e-8)  
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--save-model", type=int, default=1)

    ## RUN
    args = parser.parse_args()

    args.world_size = args.num_gpus * args.num_nodes
    os.environ['MASTER_ADDR'] = '10.57.23.164'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.num_gpus, args=(args,))
