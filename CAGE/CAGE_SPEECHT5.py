from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
import torch
import librosa

processor = SpeechT5Processor.from_pretrained("/models/T5_out3/checkpoint-2700")
model = SpeechT5ForSpeechToText.from_pretrained("/models/T5_out3/checkpoint-2700")

audio_file = "/distance/60/audio01_Raw_0.wav"
audio, sr = librosa.load(audio_file, sr=processor.feature_extractor.sampling_rate)

inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")

predicted_ids = model.generate(**inputs, max_length=100)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print("Transcription:", transcription[0])

