from collections import defaultdict
from random import choices, choice
import pickle
from time import perf_counter as time
import os
from data.dataset import pairs

class KeyListDictionary(dict):
    """
    Stores keys in a list as well as a dictionary for faster random selection
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keylist = list(self.keys())

        self.keyidx = {}
        for i, k in enumerate(self.keylist):
            self.keyidx[k] = i

    def __setitem__(self, key, value):
        if key not in self:
            self.keylist.append(key)
            self.keyidx[key] = len(self.keylist) - 1
        super().__setitem__(key, value)

    def __delitem__(self, key):
        idx = self.keyidx[key]
        last_element = self.keylist[-1]

        self.keylist[idx] = self.keylist[-1]
        self.keyidx[last_element] = idx

        self.keylist.pop()
        del self.keyidx[key]

        super().__delitem__(key)

    def pop(self, key, default=None):
        if key in self:
            idx = self.keyidx[key]
            last_element = self.keylist[-1]

            self.keylist[idx] = self.keylist[-1]
            self.keyidx[last_element] = idx

            self.keylist.pop()
            del self.keyidx[key]

        return super().pop(key, default)

    def clear(self):
        super().clear()
        self.keylist.clear()
        self.keyidx.clear()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.keylist = list(self.keys())
        self.keyidx = {}
        for i, k in enumerate(self.keylist):
            self.keyidx[k] = i

    def get_random_key(self):
        return choice(self.keylist)

    def __repr__(self):
        return f"KeyListDictionary({super().__repr__()})"


def corpus_to_dict(corpus: str) -> dict:
    return dict((i, dict(j)) for i, j in corpus.items())

def save_dict(dictionary: dict, file_path: str) -> None:
    assert type(dictionary) == dict, f"Dictionary must specifically be a dict. (not {type(dictionary)})"
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file, -1)

def load_dict(file_path: str) -> dict:
    with open(file_path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def create_corpus(ngram: int=2) -> KeyListDictionary:
    corpus = defaultdict(lambda: defaultdict(int))
    
    # get all states
    for _, phon in pairs:
        for i in range(len(phon) - ngram - 1):
            prev_state = tuple(phon[i:i + ngram])
            next_state = phon[i + ngram]
            corpus[prev_state][next_state] += 1
    
    # calculate probabilities 
    for state, transitions in corpus.items():
        total = sum(transitions.values())
        for s, c in transitions.items():
            corpus[state][s] = c / total

    return KeyListDictionary(corpus)


def generate(corpus: KeyListDictionary, starting_keys: list|str, ngram=2, max_len=100) -> list:
    if not isinstance(starting_keys, list):
        starting_keys = list(starting_keys)
        
    for _ in range(max_len - len(starting_keys)):
        cur_state = tuple(starting_keys[-ngram:])
        if cur_state not in corpus or not corpus[cur_state]:
            break
        next_state = choices(list(corpus[cur_state].keys()), list(corpus[cur_state].values()))
        starting_keys.append(next_state[0])

    return starting_keys

def load_corpus(ngram: int=2) -> KeyListDictionary:
    corpus_name = f'data/corpus/corpus_{ngram}.pkl'
    if os.path.exists(corpus_name):
        print('Loading corpus...')
        start = time()
        corpus = load_dict(corpus_name)
        print(f'Corpus loaded. ({round(time() - start, 2)}s)')
    else:
        print('Creating corpus... (This might take a while the first run)')
        start = time()
        corpus = create_corpus(ngram)
        print(f'Corpus created. ({round(time() - start, 2)}s)')
        print('Saving corpus...')
        save_dict(corpus_to_dict(corpus), corpus_name)
    return KeyListDictionary(corpus)
