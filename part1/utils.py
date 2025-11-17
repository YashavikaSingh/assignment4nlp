import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # QWERTY keyboard adjacency map for typo simulation
    qwerty_adjacent = {
        'a': ['s', 'q', 'w'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'e': ['w', 'r', 'd', 's'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'], 'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k', 'l'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'], 'q': ['w', 'a'], 'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'], 't': ['r', 'y', 'g', 'f'], 'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'], 'w': ['q', 'e', 's', 'a'], 'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'], 'z': ['x', 's', 'a']
    }
    
    def introduce_typo(word):
        """Introduce a typo by replacing a random character with an adjacent QWERTY key."""
        if len(word) < 2:
            return word
        
        # Find alphabetic characters in the word
        char_positions = [(i, c.lower()) for i, c in enumerate(word) if c.isalpha()]
        if not char_positions:
            return word
        
        # Randomly select a character to replace (with 30% probability per word)
        if random.random() < 0.3:
            pos, char = random.choice(char_positions)
            if char in qwerty_adjacent:
                replacement = random.choice(qwerty_adjacent[char])
                # Preserve case
                if word[pos].isupper():
                    replacement = replacement.upper()
                word = word[:pos] + replacement + word[pos+1:]
        
        return word
    
    def get_synonym(word):
        """Get a synonym for a word using WordNet."""
        # Get synsets for the word
        synsets = wordnet.synsets(word)
        if not synsets:
            return None
        
        # Get all synonyms from synsets
        synonyms = []
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower().strip()
                # Only use single-word synonyms (no spaces, hyphens, or special chars)
                if (synonym != word.lower() and 
                    ' ' not in synonym and 
                    '-' not in synonym and
                    synonym.isalpha() and
                    len(synonym) > 1):
                    synonyms.append(synonym)
        
        if synonyms:
            return random.choice(synonyms)
        return None
    
    # Tokenize the text
    tokens = word_tokenize(example["text"])
    transformed_tokens = []
    
    for token in tokens:
        # Skip punctuation and special characters
        if not token.isalnum():
            transformed_tokens.append(token)
            continue
        
        # Apply synonym replacement with 20% probability
        if random.random() < 0.2:
            synonym = get_synonym(token.lower())
            if synonym:
                # Preserve original case pattern
                if token[0].isupper():
                    synonym = synonym.capitalize()
                transformed_tokens.append(synonym)
                continue
        
        # Apply typo introduction
        transformed_token = introduce_typo(token)
        transformed_tokens.append(transformed_token)
    
    # Detokenize back to text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
