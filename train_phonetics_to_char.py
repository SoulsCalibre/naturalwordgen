from training.phonetics_to_char import *

if __name__ == '__main__':
    train_dataloader = get_dataloader(batch_size)

    encoder = EncoderRNN(len(phon2int) + 2, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, len(char2int) + 2).to(device)

    print('training...')
    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder)

    torch.save(encoder.state_dict(), 'models/encoder_p2c.pth')
    torch.save(decoder.state_dict(), 'models/decoder_p2c.pth')