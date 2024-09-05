from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor
import torch
import librosa

# Load the saved model and processor
model_dir = "/mode_hu-bart2"
model = SpeechEncoderDecoderModel.from_pretrained(model_dir)
processor = Wav2Vec2Processor.from_pretrained(model_dir)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)

# Load and preprocess the input audio
def preprocess_audio(file_path):
    # Load the audio file
    audio_input, _ = librosa.load(file_path, sr=16000)  # 16 kHz is the standard sampling rate for HuBERT
    # Process the audio input
    # inputs = processor(audio_input, return_tensors="pt", padding=True)
    inputs = feature_extractor(audio_input, return_tensors="pt", padding=True)
    return inputs


# Evaluate the model
def evaluate_model(file_path):
    # Preprocess the input audio
    inputs = preprocess_audio(file_path)
    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        # Perform inference
        outputs = model.generate(inputs["input_values"])

    # Decode the output sequence
    decoded_output = processor.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


# Test with a sample WAV file
wav_file = "/test.wav"
result = evaluate_model(wav_file)
print("Generated Text:", result)
