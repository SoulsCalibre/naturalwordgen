from model_loader import load_models

if __name__ == '__main__':
    toPhon, toChar = load_models()

    while True:
        inp = '1'
        while not inp.isalpha():
            inp = input('Type a word to reconstuct (alpha only): ')

        print()

        phonemes = toPhon.translate(inp)
        phonemes.pop()

        print(f"{inp} >> ({' '.join(phonemes)}) >>", end=' ')

        word = toChar.translate(phonemes)
        word.pop()
        print(''.join(word), end='\n\n')  