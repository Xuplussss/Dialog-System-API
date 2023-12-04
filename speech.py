import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import soundfile as sf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

model_path="./weight_file"
mask_prob=0.0
mask_length=10

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.float()
model.eval()
# device = 'cuda'
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 256)
        self.linear3 = nn.Linear(256, 192)
        self.linear4 = nn.Linear(192, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        self.linear7 = nn.Linear(64, 32)
        self.linear8 = nn.Linear(32, 5)
        self.linear9 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        xs = F.relu(self.linear6(x))
        xa =  F.relu(self.linear7(x))
        xs = self.linear8(xs)
        xa = self.linear9(xa)
        return xs, xa

PATH = './Hybrid.pth'
clf = Network().to(device)
clf.load_state_dict(torch.load(PATH))
clf.eval()

def hybrid_detection(save_file):   
    wav, sr = sf.read(save_file)
    input_values = feature_extractor(wav, return_tensors="pt").input_values
    input_values = input_values.float()
    input_values = input_values.to(device)
    input_values = input_values.view(1,-1)
    with torch.no_grad():
        outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state
    inputs = torch.mean(last_hidden_state, 1)

    with torch.no_grad():
        output_speed, output_angry = clf(inputs.to(device))
        __, predicted_speed = torch.max(output_speed, 1)
        __, predicted_angry = torch.max(output_angry, 1)
        return predicted_speed, predicted_angry


