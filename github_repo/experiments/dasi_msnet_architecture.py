"""
Module 02: DASI-MSNet PyTorch Architecture & Baseline Models
Doctoral Novelties Implemented:
1. Multi-Scale Temporal Convolutional Block (Micro, Meso, Macro Temporal Dynamics)
2. Gradient Reversal Layer (GRL) for Subject-Identity Invariance
3. Cost-Sensitive Ordinal Focal Loss for 3-Level Engagement (Disengaged < Engaged < Highly Engaged)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. GRADIENT REVERSAL LAYER (GRL) FOR ADVERSARIAL SUBJECT DEBIASING
# ---------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient and scale by lambda
        return grad_output.neg() * ctx.lambda_val, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


# ---------------------------------------------------------------------------
# 2. MULTI-SCALE TEMPORAL CONVOLUTIONAL BLOCK (MS-TCN)
# ---------------------------------------------------------------------------
class MultiScaleTemporalBlock(nn.Module):
    """
    Captures multi-rate facial kinematics:
    - Micro branch (k=3, d=1): rapid blinks, eyelid twitches, micro-expressions
    - Meso branch (k=5, d=1): head nods, smile onset, gaze shifts
    - Macro branch (k=7, d=2): sustained attention, persistent posture
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 3
        rem = out_channels - branch_channels * 2

        self.conv_micro = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_meso  = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2, dilation=1)
        self.conv_macro = nn.Conv1d(in_channels, rem,             kernel_size=7, padding=6, dilation=2)

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x shape: (B, C, T)
        b_micro = self.conv_micro(x)
        b_meso  = self.conv_meso(x)
        b_macro = self.conv_macro(x)

        out = torch.cat([b_micro, b_meso, b_macro], dim=1) # (B, out_channels, T)
        out = out + self.residual(x)
        
        # Apply LayerNorm along channels: (B, T, C)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.act(out).transpose(1, 2)
        return out


# ---------------------------------------------------------------------------
# 3. PROPOSED DOCTORAL ARCHITECTURE: DASI-MSNet
# ---------------------------------------------------------------------------
class DASI_MSNet(nn.Module):
    def __init__(self, in_features=44, hidden_dim=64, num_classes=3, num_subjects=22):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input Feature Projector
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Multi-Scale Temporal Feature Extraction
        self.ms_tcn1 = MultiScaleTemporalBlock(hidden_dim, hidden_dim)
        self.ms_tcn2 = MultiScaleTemporalBlock(hidden_dim, hidden_dim)

        # Temporal Attention Pooling
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Head 1: Engagement Classifier (3 Classes)
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        # Head 2: Subject Adversarial Discriminator (via GRL)
        self.grl = GradientReversalLayer(lambda_val=1.0)
        self.subject_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_subjects)
        )

    def set_lambda(self, lambda_val):
        self.grl.lambda_val = lambda_val

    def extract_latent(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        proj = self.input_proj(x) # (B, T, hidden)
        
        # Convolutions over time: (B, hidden, T)
        tcn_in = proj.transpose(1, 2)
        tcn_out = self.ms_tcn1(tcn_in)
        tcn_out = self.ms_tcn2(tcn_out).transpose(1, 2) # (B, T, hidden)

        # Attention Pooling
        attn = F.softmax(self.attn_weights(tcn_out), dim=1) # (B, T, 1)
        z = torch.sum(attn * tcn_out, dim=1) # (B, hidden)
        return z

    def forward(self, x):
        z = self.extract_latent(x)
        
        # 1. Predict Engagement Level
        eng_logits = self.engagement_head(z)
        
        # 2. Predict Subject Identity through GRL
        z_rev = self.grl(z)
        subj_logits = self.subject_head(z_rev)

        return eng_logits, subj_logits, z


# ---------------------------------------------------------------------------
# 4. BASELINE ARCHITECTURES FOR HONEST ABLATION BENCHMARKING
# ---------------------------------------------------------------------------
class Standard_BiLSTM(nn.Module):
    """Conventional 2-layer Bi-LSTM (Das & Dev 2025, Selim et al. 2022) without GRL"""
    def __init__(self, in_features=44, hidden_dim=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)


class Standard_TCN(nn.Module):
    """Standard Single-Scale TCN without Adversarial Invariance (Copur et al. 2022)"""
    def __init__(self, in_features=44, hidden_dim=64, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        proj = self.input_proj(x).transpose(1, 2)
        out = F.gelu(self.conv1(proj))
        out = F.gelu(self.conv2(out)).transpose(1, 2)
        pooled = torch.mean(self.norm(out), dim=1)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# 5. COST-SENSITIVE ORDINAL FOCAL LOSS
# ---------------------------------------------------------------------------
class CostSensitiveOrdinalFocalLoss(nn.Module):
    """
    Penalizes cross-polar errors more heavily than adjacent errors:
    Penalty matrix C(i, j) = |i - j|^1.5
    Combines Focal Loss (gamma=2.0) with Class-Frequency Weighting
    """
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)
        
        # Ordinal distance penalty matrix (3x3):
        # C[0, 2] = 2.0^1.5 = 2.83, C[0, 1] = 1.0, C[i, i] = 0.0
        cost_mat = torch.tensor([
            [1.0,  1.5,  3.0],
            [1.5,  1.0,  1.5],
            [3.0,  1.5,  1.0]
        ], dtype=torch.float32)
        self.register_buffer('cost_matrix', cost_mat)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # One-hot representation
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        
        # Pt: probability assigned to correct class
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Ordinal cost multiplier based on predicted expectation vs target
        pred_classes = torch.argmax(probs, dim=1)
        ordinal_costs = self.cost_matrix[targets, pred_classes]

        # Base cross-entropy per sample
        ce = -(targets_one_hot * log_probs).sum(dim=1)
        
        if self.class_weights is not None:
            w = self.class_weights[targets]
            loss = w * focal_weight * ordinal_costs * ce
        else:
            loss = focal_weight * ordinal_costs * ce

        return loss.mean()
