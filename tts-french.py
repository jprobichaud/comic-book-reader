import torch
import sys
from omegaconf import OmegaConf
import torchaudio

# torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
#                                'latest_silero_models.yml',
#                                progress=False)
# models = OmegaConf.load('latest_silero_models.yml')
# 
# # see latest avaiable models
# available_languages = list(models.tts_models.keys())
# print(f'Available languages {available_languages}')
# 
# for lang in available_languages:
#     _models = list(models.tts_models.get(lang).keys())
#     print(f'Available models for {lang}: {_models}')


language = 'fr'

speaker = "gilles_v2"
speaker = "gilles_16khz"
speaker = "v3_fr"

device = torch.device('cpu')
#model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
model, example = torch.hub.load(repo_or_dir='snakers4/silero-models',
                      model='silero_tts',
                      language=language,
                      speaker=speaker)


print(f"type model : {type(model)}")

print(model.speakers)

speaker = model.speakers[0]

if len(sys.argv) == 1:
    example_text = "salut!  c'est spécial comme le monde est petit!"
else:
    example_text = sys.argv[1]

print(f"text : {example_text}")

sample_rate = 48000
put_accent=True
put_yo=True
#model = model.to(device)  # gpu or cpu
audio = model.apply_tts(text=example_text,
                  speaker=speaker,
                  sample_rate=sample_rate,
                  put_accent=put_accent,
                  put_yo=put_yo)
audio = audio.unsqueeze(0)
print(f"audio type {type(audio)}, {audio.shape}")
torchaudio.save( "tmp.wav", audio, sample_rate, encoding="PCM_S", bits_per_sample=16)

