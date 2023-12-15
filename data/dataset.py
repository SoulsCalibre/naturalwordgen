import pickle

MAX_LENGTH = 13
SOS_token = 0
EOS_token = 1

with open('./data/mappings.pkl', 'rb') as handle:
    mappings = pickle.load(handle)

with open('./data/data.pkl', 'rb') as handle:
    pairs = pickle.load(handle)