import json
import os
import random

import torch
import torchaudio
from lhotse import CutSet
from torchaudio.io import AudioEffector


def write_result(path, result_):
    with open(path, "w") as f:
        f.write(json.dumps(result_))


def get_result(path):
    with open(path, "r") as f:
        result = f.read()
    return json.loads(result)


def audio_lowpass(waveform, fs):
    waveform = waveform.permute(1, 0)
    effector = AudioEffector(effect="lowpass=2000")
    applied = effector.apply(waveform, fs)
    return applied.permute(1, 0)


def normalize(waveform):
    max_value = torch.max(torch.abs(waveform))
    if max_value == 0:
        return waveform
    normalized_waveform = waveform / max_value

    return normalized_waveform


cut_set = CutSet.from_jsonl("/root/public/dev8T/jtang/ASR/datasets/libriheavy-master/libriheavy_cuts_medium_filter_audio.jsonl.gz")
PATH_8k = r"/root/public/dev8T/jtang/ASR/datasets/libriheavy_test/clean8k/"
JSON_PTAH = r"/root/public/dev8T/jtang/ASR/datasets/libriheavy_test/5w_medium_8k_noised_data.json"

NOISE_PATH = r"/root/public/dev8T/jtang/fmcw/process/TARGET/NOISE_8K/NOISE/"
noise_list = os.listdir(NOISE_PATH)
data_length = 50000
MAX_LENGTH = 926066
save_json_data = []
try:
    for index in range(data_length):
        cut = cut_set[index]
        if index % 500 == 0:
            print("step", index)
        # cut = cut.trim_to_supervisions()
        audio_samples = cut.load_audio()
        for s_index, item in enumerate(cut.supervisions):
            audio_name = str(index).zfill(5) + ".wav"
            audio_8k_path = os.path.join(PATH_8k, audio_name)
            audio_sample = audio_samples[s_index]
            audio_id = item.id
            audio_sample = torch.from_numpy(audio_sample)
            audio_sample = audio_sample.unsqueeze(0)
            # audio_sample, sample_rate = torchaudio.from_numpy(audio_sample)
            audio_sample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)(audio_sample)
            noisy_audio_file = random.choice(noise_list)
            noisy_audio, noisy_rate = torchaudio.load(os.path.join(NOISE_PATH, noisy_audio_file))
            noisy_audio = normalize(noisy_audio)
            audio_sample = audio_lowpass(audio_sample, 8000)
            mixing_ratio = random.uniform(0.02, 0.48)
            if noisy_audio.size(1) > audio_sample.size(1):
                noisy_audio = noisy_audio[:, :audio_sample.size(1)]
            else:
                if audio_sample.size(1) / noisy_audio.size(1) < 2:
                    noisy_audio = torch.nn.functional.pad(noisy_audio, pad=(0, audio_sample.size(1) - noisy_audio.size(1)),
                                                          mode='reflect')
                else:
                    bei_size = int(audio_sample.size(1) / noisy_audio.size(1))
                    noisy_audio = noisy_audio.repeat(1, bei_size)
                    noisy_audio = torch.nn.functional.pad(noisy_audio, pad=(0, audio_sample.size(1) - noisy_audio.size(1)),
                                                          mode='reflect')
            mixed_audio = audio_sample * (1 - mixing_ratio) + noisy_audio[1] * mixing_ratio
            mixed_audio = 0.98 * mixed_audio + 0.02 * torch.randn(mixed_audio.size())
            torchaudio.save(audio_8k_path, src=mixed_audio, sample_rate=8000)
            save_json_data.append({"text_origin": item.custom['texts'][0], "text_upper": item.custom['texts'][1],
                                   "text_pre": item.custom['pre_texts'][0],
                                  "id": audio_id, "data_path": audio_name})
finally:
    write_result(JSON_PTAH, save_json_data)