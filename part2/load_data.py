import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Process natural language and SQL data files.
        For test set, only load natural language queries.
        '''
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        
        if split == 'test':
            # Test set only has natural language queries
            return [(nl, None) for nl in nl_lines]
        else:
            # Train and dev sets have both NL and SQL
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            return list(zip(nl_lines, sql_lines))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns:
        - For train/dev: (encoder_input_ids, decoder_input_ids, decoder_target_ids, initial_decoder_token)
        - For test: (encoder_input_ids, initial_decoder_token)
        '''
        nl_query, sql_query = self.data[idx]
        
        # Tokenize encoder input (natural language)
        encoder_encoded = self.tokenizer(
            nl_query,
            max_length=512,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        encoder_input_ids = encoder_encoded['input_ids']
        
        # Get initial decoder token (use extra_id_0 as BOS token)
        initial_decoder_token = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        
        if self.split == 'test':
            return {
                'encoder_input_ids': encoder_input_ids,
                'initial_decoder_token': initial_decoder_token
            }
        else:
            # Tokenize decoder input and target (SQL)
            # Decoder input should start with <extra_id_0>
            # Decoder target should be aligned with logits (shifted by one position)
            decoder_input_text = f'<extra_id_0>{sql_query}'
            
            decoder_input_encoded = self.tokenizer(
                decoder_input_text,
                max_length=512,
                padding=False,
                truncation=True,
                return_tensors=None
            )
            
            decoder_input_ids = decoder_input_encoded['input_ids']
            
            # Extract the extra_id_0 token ID
            extra_id_0_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
            
            # Target should be the tokens after <extra_id_0>, aligned with logits
            # If decoder_input_ids = [extra_id_0, token1, token2, ...], 
            # then decoder_targets should be [token1, token2, ..., eos] with same length
            # We need to find where extra_id_0 ends and SQL tokens begin
            if decoder_input_ids[0] == extra_id_0_id:
                # Remove the first token (<extra_id_0>) and add EOS at the end
                decoder_target_ids = decoder_input_ids[1:] + [self.tokenizer.eos_token_id]
                # Pad or truncate to match decoder_input_ids length
                target_len = len(decoder_input_ids)
                if len(decoder_target_ids) < target_len:
                    # Pad with PAD_IDX
                    decoder_target_ids = decoder_target_ids + [PAD_IDX] * (target_len - len(decoder_target_ids))
                elif len(decoder_target_ids) > target_len:
                    # Truncate
                    decoder_target_ids = decoder_target_ids[:target_len]
            else:
                # Fallback: tokenize SQL separately
                sql_encoded = self.tokenizer(
                    sql_query,
                    max_length=511,  # Leave room for EOS
                    padding=False,
                    truncation=True,
                    return_tensors=None
                )
                decoder_target_ids = sql_encoded['input_ids'] + [self.tokenizer.eos_token_id]
                # Ensure same length as decoder_input_ids
                target_len = len(decoder_input_ids)
                if len(decoder_target_ids) < target_len:
                    decoder_target_ids = decoder_target_ids + [PAD_IDX] * (target_len - len(decoder_target_ids))
                elif len(decoder_target_ids) > target_len:
                    decoder_target_ids = decoder_target_ids[:target_len]
            
            return {
                'encoder_input_ids': encoder_input_ids,
                'decoder_input_ids': decoder_input_ids,
                'decoder_target_ids': decoder_target_ids,
                'initial_decoder_token': initial_decoder_token
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input_ids_list = [torch.tensor(item['encoder_input_ids'], dtype=torch.long) for item in batch]
    decoder_input_ids_list = [torch.tensor(item['decoder_input_ids'], dtype=torch.long) for item in batch]
    decoder_target_ids_list = [torch.tensor(item['decoder_target_ids'], dtype=torch.long) for item in batch]
    initial_decoder_tokens = [item['initial_decoder_token'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_input_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_target_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs (for evaluation/generation)
    initial_decoder_inputs = torch.tensor(initial_decoder_tokens, dtype=torch.long)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input_ids_list = [torch.tensor(item['encoder_input_ids'], dtype=torch.long) for item in batch]
    initial_decoder_tokens = [item['initial_decoder_token'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_input_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs (for evaluation/generation)
    initial_decoder_inputs = torch.tensor(initial_decoder_tokens, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x