import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    # Note: This script trains from scratch, so finetune is always False
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--eval_generation_freq', type=int, default=1,
                        help="How often to run full generation and metrics (every N epochs). Set to 0 to disable generation during training.")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    # Always set finetune to False for scratch training
    args.finetune = False
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'scr'  # Always scratch for this script
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'scratch_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    # Load tokenizer once for efficiency
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Only run full generation/metrics if it's time (or always if freq=1)
        should_generate = (args.eval_generation_freq > 0 and epoch % args.eval_generation_freq == 0)
        
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path,
                                                                         tokenizer, should_generate)
        if should_generate:
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        else:
            print(f"Epoch {epoch}: Dev loss: {eval_loss} (skipping generation/metrics this epoch)")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Use T5's built-in loss computation with labels parameter
        # This handles the proper alignment automatically
        # Labels should be the target sequence with -100 for padding tokens (which are ignored in loss)
        labels = decoder_targets.clone()
        labels[labels == PAD_IDX] = -100  # -100 is ignored in CrossEntropyLoss
        
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            # Count non-padding tokens for averaging
            non_pad = decoder_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path, tokenizer, should_generate=True):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    generated_sql_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss using T5's built-in loss
            labels = decoder_targets.clone()
            labels[labels == PAD_IDX] = -100  # -100 is ignored in loss
            
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Count non-padding tokens for averaging
            non_pad = decoder_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Only generate if requested (expensive operation)
            if should_generate:
                generation_config = GenerationConfig(
                    max_length=512,
                    num_beams=1,  # Greedy decoding
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=tokenizer.pad_token_id,  # T5 decoder starts with pad_token
                )
                
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    generation_config=generation_config,
                )
                
                # Decode generated SQL queries
                for gen_ids in generated_ids:
                    sql_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    generated_sql_queries.append(sql_query)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    # Only compute metrics if we generated queries
    if should_generate and len(generated_sql_queries) > 0:
        # Save generated queries and records
        save_queries_and_records(generated_sql_queries, model_sql_path, model_record_path)
        
        # Compute metrics
        sql_em, record_em, record_f1, error_msgs = compute_metrics(
            gt_sql_pth, model_sql_path, gt_record_path, model_record_path
        )
        
        # Compute error rate
        error_rate = sum(1 for msg in error_msgs if msg != "") / len(error_msgs) if error_msgs else 0
    else:
        # Return dummy values if not generating
        sql_em, record_em, record_f1, error_rate = 0.0, 0.0, 0.0, 0.0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    generated_sql_queries = []
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            generation_config = GenerationConfig(
                max_length=512,
                num_beams=1,  # Greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=tokenizer.pad_token_id,  # T5 decoder starts with pad_token
            )
            
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=generation_config,
            )
            
            # Decode generated SQL queries
            for gen_ids in generated_ids:
                sql_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_sql_queries.append(sql_query)
    
    # Save generated queries and records
    save_queries_and_records(generated_sql_queries, model_sql_path, model_record_path)
    print(f"Saved {len(generated_sql_queries)} test predictions to {model_sql_path}")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    print(f"Initializing model from scratch (finetune=False)")
    model = initialize_model(args)
    print(f"Model initialized: {model is not None}, type: {type(model)}")
    if model is None:
        raise ValueError("Model is None after initialization!")
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'scratch_experiment'
    model_type = 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path,
                                                                                    tokenizer, should_generate=True)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()

