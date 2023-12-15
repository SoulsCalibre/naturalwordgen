from model_loader import load_models
import data.markov as markov

if __name__ == '__main__':
    toPhon, toChar = load_models()
    ngram = 3
    corpus = markov.load_corpus(ngram)

    # pick a random key from the dictionary
    starting_words = corpus.get_random_key()
    max_len = 6

    print(f'\n***\nPress Enter to generate. Type "exit" to exit. Type a number greater than {ngram} to change the number of phonetics.\n***')

    while True:
        seq = markov.generate(corpus, starting_words, ngram, max_len=max_len)
        string = toChar.translate(seq)
        string.pop()
        string = ''.join(string)
        print(string.replace('<SOS>', ''), f"({' '.join(seq)})", end=' ')
        inp = input('')
        if inp == 'exit':
            break
        elif inp.isdigit():
            max_len = max(ngram, int(inp))
        starting_words = corpus.get_random_key()
            