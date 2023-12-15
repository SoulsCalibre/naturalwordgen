from training.model import *
from data.mappings import *
from time import time

class Translator:
    def __init__(self, encode_model, decode_model, toWord: bool = False) -> None:
        """
        Set toWord to True if you are loading the phonetics to word model
        """
        self.encode_model = encode_model
        self.decode_model = decode_model
        self.toWord = toWord

    def translate_raw(self, seq):
        seq = seq[::-1]
        if isinstance(seq, str):
            seq = seq.lower()
        elif isinstance(seq, list):
            if seq[-1] == '<EOS>':
                seq.pop()
        
        return translateSentence(self.encode_model, self.decode_model, seq, self.toWord)


    def translate(self, seq):
        seq = seq[::-1]
        if isinstance(seq, str):
            seq = seq.lower()
        elif isinstance(seq, list):
            if seq[-1] == '<EOS>':
                seq.pop()
        else:
            raise TypeError(f'seq must be str or list, got {type(seq)}')

        if self.toWord:
            return phonetics_to_words(self.encode_model, self.decode_model, seq)
        return word_to_phonetics(self.encode_model, self.decode_model, seq)

def load_models():
    """
    Loads models and returns a model that converts phonetics to words and words to phonetics in that order.

    Ex Usage:
        toPhon, toChar = load_models()
        phonetics = toPhon.translate('hello')
        print(phonetics)
    """
    print('loading models...')
    start = time()
    char2int = mappings['char2int']
    phon2int = mappings['phon2int']

    # to phonetics
    encoder = EncoderRNN(len(char2int) + 2, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, len(phon2int) + 2).to(device)
    encoder_dict = torch.load('models/encoder_c2p.pth')
    decoder_dict = torch.load('models/decoder_c2p.pth')
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

    # to character
    encoder2c = EncoderRNN(len(phon2int) + 2, hidden_size).to(device)
    decoder2c = AttnDecoderRNN(hidden_size, len(char2int) + 2).to(device)
    encoder2c_dict = torch.load('models/encoder_p2c.pth')
    decoder2c_dict = torch.load('models/decoder_p2c.pth')
    encoder2c.load_state_dict(encoder2c_dict)
    decoder2c.load_state_dict(decoder2c_dict)

    toPhonetics = Translator(encoder, decoder)
    toChar = Translator(encoder2c, decoder2c, toWord=True)
    print(f'models loaded. ({round(time() - start, 2)}s)')
    return toPhonetics, toChar
