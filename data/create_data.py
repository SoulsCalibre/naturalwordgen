"""
Script used to create data.plk and mappings.plk.
"""

import nltk
from nltk.corpus import cmudict
import pickle

if __name__ == '__main__':
    nltk.download('cmudict')
    cmudict = cmudict.dict()

    phonemes = set()
    characters = set() 
    pairs = []
    for i, j in cmudict.items():
        if not i.isalpha():
            continue

        for pronounciation in j:
            for phoneme in pronounciation:
                phonemes.add(phoneme)
        for c in i:
            characters.add(c)
        pairs.append((list(i), j[0]))  # pair word with it's first pronunciation

    phonemes = sorted(phonemes)
    characters = sorted(characters)

    MAX_LENGTH = 13
    pairs = [p for p in pairs if len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH]

    # mappings
    SOS_token = 0
    EOS_token = 1
    phon2int = dict(zip(phonemes, range(2, len(phonemes)+2)))
    char2int = dict(zip(characters, range(2, len(characters)+2)))
    int2phon = dict((j, i) for i, j in phon2int.items())
    int2char = dict((j, i) for i, j in char2int.items())

    phon2int['<SOS>'], phon2int['<EOS>'] = SOS_token, EOS_token
    char2int['<SOS>'], char2int['<EOS>'] = SOS_token, EOS_token
    int2phon[SOS_token], int2phon[EOS_token] = '<SOS>', '<EOS>'
    int2char[SOS_token], int2char[EOS_token] = '<SOS>', '<EOS>'

    map = {'phon2int': phon2int, 'char2int': char2int, 'int2phon': int2phon, 'int2char': int2char}

    with open('data/mappings.pkl', 'wb') as handle:
        pickle.dump(map, handle, -1)

    with open('data/data.pkl', 'wb') as handle:
        pickle.dump(pairs, handle, -1)
