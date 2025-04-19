import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractionBlock(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = adaptive_filter

        # trunc_normal_(self.complex_weight_high, std=.02)
        # trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        # print("feb in: ",x_in.shape)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        # print("feb xfft: ",x_fft.shape)
        weight = torch.view_as_complex(self.complex_weight)
        # print("feb weight: ", x_fft.shape)
        x_weighted = x_fft * weight
        # print("feb xweight: ", x_fft.shape)
        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, 64)  # Reshape back to original shape
        # print("feb out: ", x_in.shape)
        return x


class IntraChannelBlock(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)

        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class MultiModalFeatureBlock(nn.Module):
    def __init__(self, feature_dim, transformer_dim):
        super(MultiModalFeatureBlock, self).__init__()
        self.fc1 = nn.Linear(feature_dim, transformer_dim)
        self.fc2 = nn.Linear(feature_dim, transformer_dim)
        self.fc3 = nn.Linear(feature_dim, transformer_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.attention = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, z1, z2, z3):
        # FC layer projection
        f1 = self.fc1(z1)
        f2 = self.fc2(z2)
        f3 = self.fc3(z3)

        # Cat & Add positional encoding (simplified here)
        fused = torch.stack([f1, f2, f3], dim=1)  # (B, 3, D)
        fused = torch.mean(fused, dim=2).squeeze()

        # Transformer encoder
        transformer_out = self.transformer_encoder(fused)

        # Attention-based fusion
        weights = self.attention(transformer_out)  # (B, 3, 1)
        output = (transformer_out * weights).sum(dim=1)  # (B, D)

        return output


class SleepFIM(nn.Module):
    def __init__(self, input_dim=1, feature_dim=64, transformer_dim=128, num_classes=5):
        super(SleepFIM, self).__init__()

        # Feature Extraction Network
        self.feb = FeatureExtractionBlock(feature_dim)
        self.icb = IntraChannelBlock(feature_dim, feature_dim)
        self.fc1 = nn.Linear(feature_dim, transformer_dim)
        self.pool = nn.AdaptiveAvgPool1d(80)

        # Multimodal Feature Fusion
        # self.mffb = MultiModalFeatureBlock(feature_dim, transformer_dim)

        # Classifier
        '''
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, num_classes),
            nn.Softmax(dim=1)
        )
        '''

    def forward(self, eeg1, eeg2=None, eog=None):
        eeg1 = eeg1.transpose(1,2)

        # Pass through FEB and ICB
        z1 = self.icb(self.feb(eeg1))
        # z2 = self.icb(self.feb(eeg2))
        # z3 = self.icb(self.feb(eog))
        f1 = self.fc1(z1).transpose(1, 2)

        out = self.pool(f1).transpose(1, 2)
        # Feature Fusion
        # fused = self.mffb(z1, z2, z3)

        # Classification
        # output = self.classifier(fused)
        return [out]


'''
model = SleepFIM().cuda()
x = torch.rand(2, 30000, 1).cuda()
# y = torch.rand(2, 30000, 1).cuda()
# z = torch.rand(2, 30000, 1).cuda()
print(model(x).shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
'''