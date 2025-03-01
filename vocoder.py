import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cuda", 0)
model = build_model().to(device)

checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    print(c)
    print(c.shape)
    waveform = wavegen(model, c=c)   
    librosa.output.write_wav(name+'.wav', waveform, sr=16000)