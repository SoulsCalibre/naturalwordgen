from training.char_to_phonetics import *

train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(len(char2int) + 2, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(phon2int) + 2).to(device)

print('training...')
train(train_dataloader, encoder, decoder, 80, print_every=1, plot_every=5)

encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)

torch.save(encoder.state_dict(), 'models/encoder_c2p.pth')
torch.save(decoder.state_dict(), 'models/decoder_c2p.pth')