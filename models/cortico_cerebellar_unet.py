import torch
import torch.nn as nn
from models.unet_parts import UNetEncoder, UNetDecoder
from models.cerebellar_module import CerebellarModule
from models.cortical_rnn import CorticalRNN
from config import cfg

class CorticoCerebellarUNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = cfg
        self.cfg = config
        self.feedback_type = config.FEEDBACK_TYPE
        self.tau_values = config.TAU_VALUES
        self.cereb_loss_max_weight = config.CEREB_LOSS_MAX_WEIGHT
        self.normalize_features = config.NORMALIZE_FEATURES
        self.bottleneck_dim = config.BOTTLENECK_DIM

        self.encoder = UNetEncoder(in_channels=1)
        if config.FREEZE_ENCODER:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.normalize_features:
            self.feature_norm = nn.LayerNorm(self.bottleneck_dim)

        if self.feedback_type == 'cerebellar':
            self.cerebellum = CerebellarModule(
                input_dim=self.bottleneck_dim,
                hidden_size=config.HIDDEN_SIZE,
                bottleneck_dim=self.bottleneck_dim,
                granule_expansion=config.GRANULE_EXPANSION,
                tau_values=config.TAU_VALUES,
                freeze_cerebellar_input=config.FREEZE_CEREBELLAR_INPUT,
                clip_feedback=config.CLIP_FEEDBACK,
                feedback_act=config.FEEDBACK_ACT,
                dropout=config.GRANULE_DROPOUT
            )
        else:
            self.cerebellum = None

        self.cortex = CorticalRNN(
            input_dim=self.bottleneck_dim,
            hidden_size=config.HIDDEN_SIZE,
            alpha=config.CORTICAL_ALPHA,
            freeze_recurrent=config.FREEZE_RECURRENT,
            freeze_input=config.FREEZE_INPUT,
            feedback_type=self.feedback_type
        )

        self.project_back = nn.Linear(config.HIDDEN_SIZE, self.bottleneck_dim)
        self.fc_vec_to_channels = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.conv_mix = nn.Conv2d(self.bottleneck_dim * 2, self.bottleneck_dim, kernel_size=1)
        self.decoder = UNetDecoder(out_channels=1)

    def forward(self, x, ablate: bool = False, compute_cereb_loss: bool = False,
                detach_cereb_input: bool = True, current_epoch: int = None):
        B, T, C, H, W = x.shape
        device = x.device

        skips_list = []
        bottleneck_features = []
        for t in range(T):
            e1, e2, e3, pooled = self.encoder(x[:, t])
            skips_list.append((e1, e2, e3))
            bottleneck_features.append(pooled)

        features_seq = torch.stack(bottleneck_features, dim=1)

        if self.normalize_features and hasattr(self, 'feature_norm'):
            features_seq = self.feature_norm(features_seq)

        if self.feedback_type == 'cerebellar' and not ablate:
            f_h_seq, cereb_preds_per_step = self.cortex(
                features_seq,
                cerebellum=self.cerebellum,
                ablate=False,
                return_cereb_preds=compute_cereb_loss,
                detach_cereb_input=detach_cereb_input
            )
        elif self.feedback_type == 'readout':
            f_h_seq, _ = self.cortex(features_seq, cerebellum=None, ablate=False, return_cereb_preds=False)
            cereb_preds_per_step = None
        else:
            f_h_seq, _ = self.cortex(features_seq, return_cereb_preds=False)
            cereb_preds_per_step = None

        cereb_loss = torch.tensor(0.0, device=device)
        if compute_cereb_loss and self.feedback_type == 'cerebellar' and not ablate and cereb_preds_per_step is not None:
            loss_cereb = 0.0
            count = 0
            for t in range(T):
                preds_t = cereb_preds_per_step[t]
                for tau, pred_feat in preds_t.items():
                    future_t = t + tau
                    if future_t < T:
                        future_feat = features_seq[:, future_t, :].detach()
                        loss_cereb += nn.functional.smooth_l1_loss(pred_feat, future_feat)
                        count += 1
            if count > 0:
                cereb_loss = loss_cereb / count

            if current_epoch is not None:
                if current_epoch < self.cfg.CEREB_LOSS_WARMUP_EPOCHS:
                    weight = 0.0
                else:
                    ramp = min(1.0, (current_epoch - self.cfg.CEREB_LOSS_WARMUP_EPOCHS) / self.cfg.CEREB_LOSS_RAMPUP_EPOCHS)
                    weight = ramp * self.cereb_loss_max_weight
                cereb_loss = cereb_loss * weight

        proj_seq = self.project_back(f_h_seq)

        outputs = []
        for t in range(T):
            e3 = skips_list[t][2]
            vec = proj_seq[:, t]
            vec_proj = self.fc_vec_to_channels(vec)
            vec_expanded = vec_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, e3.size(2), e3.size(3))
            combined = torch.cat([e3, vec_expanded], dim=1)
            feat = self.conv_mix(combined)
            mask = self.decoder(feat, skips_list[t][0], skips_list[t][1])
            outputs.append(mask)

        out_masks = torch.stack(outputs, dim=1)
        return out_masks, cereb_loss