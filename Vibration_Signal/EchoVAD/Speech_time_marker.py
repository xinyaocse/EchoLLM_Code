import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import io
import librosa.display

# Load audio file
audio_path = r'test.wav'
y, sr = librosa.load(audio_path, sr=None)

# Parameters
window_duration = 0.05  # 50ms window
hop_duration = 0.01  # 10ms step
window_samples = int(sr * window_duration)
hop_samples = int(sr * hop_duration)
n_fft = 512  # Increased for better frequency resolution
img_size = (128, 128)

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])


# Model definition (same as training)
class SpeechCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpeechCNN().to(device)
model.load_state_dict(torch.load('speech_detection_cnn1.pth', map_location=device))
model.eval()

# Process audio with sliding window
predictions = []
time_stamps = []

for i in range(0, len(y) - window_samples, hop_samples):
    # Extract audio segment
    start = i
    end = start + window_samples
    segment = y[start:end]

    # Generate spectrogram
    D = librosa.stft(segment, n_fft=n_fft, hop_length=n_fft // 2)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create figure
    fig = plt.figure(figsize=(4, 4), dpi=50)
    plt.axis('off')
    librosa.display.specshow(D_db, sr=sr, hop_length=n_fft // 2,
                             x_axis='time', y_axis='log')
    plt.tight_layout(pad=0)

    # Convert to tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    plt.close(fig)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = (output > 0.5).float().item()

    predictions.append(pred)
    time_stamps.append(i / sr)  # Start time of the window

# Find first and last class 1 segments
class1_times = [t for t, p in zip(time_stamps, predictions) if p == 1]
first_class1 = round(class1_times[0], 3) if class1_times else None
last_class1 = round(class1_times[-1], 3) if class1_times else None

# Print results
print(f"Total predictions: {predictions}")
print(f"speech count: {sum(predictions)}")
print(f"First speech at: {first_class1}s")
print(f"Last speech at: {last_class1}s")
