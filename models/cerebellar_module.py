import torch
import torch.nn as nn
from typing import List, Optional

class CerebellarModule(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, bottleneck_dim: int,
                 granule_expansion: int = 20, tau_values: List[int] = [1,2,3,4],
                 freeze_cerebellar_input: bool = False,
                 clip_feedback: bool = False, feedback_act: Optional[str] = 'tanh',
                 dropout: float = 0.2):
        super().__init__()
        self.tau_values = tau_values
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.granule_dim = hidden_size * granule_expansion
        self.clip_feedback = clip_feedback
        self.feedback_act = feedback_act

        self.cerebellar_heads = nn.ModuleDict()
        for tau in tau_values:
            # Mossy fibers (MF) – fixed
            mf = nn.Linear(hidden_size, self.granule_dim, bias=False)
            nn.init.kaiming_normal_(mf.weight, mode='fan_in', nonlinearity='relu')
            mf.weight.requires_grad = False

            # Parallel fibers → Purkinje cells (PF) – trainable
            pf = nn.Linear(self.granule_dim, bottleneck_dim)
            nn.init.kaiming_uniform_(pf.weight, a=0.1)

            self.cerebellar_heads[str(tau)] = nn.ModuleDict({
                'MF': mf,
                'PF': pf,
                'activation': nn.ReLU(),
                'dropout': nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            })

        # Cerebellum → cortex projection (W_Ch)
        self.W_Ch = nn.Linear(bottleneck_dim * len(tau_values), hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.W_Ch.weight)
        if freeze_cerebellar_input:
            self.W_Ch.weight.requires_grad = False

        print(f"Cerebellum: {len(tau_values)} heads, granule dim: {self.granule_dim}, "
              f"W_Ch frozen={freeze_cerebellar_input}, clip={clip_feedback}, act={feedback_act}")

    def forward(self, f_h_current: torch.Tensor, detach_input: bool = False):
        if detach_input:
            f_h_current = f_h_current.detach()

        pred_features = {}
        all_preds = []
        for tau in self.tau_values:
            head = self.cerebellar_heads[str(tau)]
            granule = head['activation'](head['MF'](f_h_current))
            granule = head['dropout'](granule)
            pred = head['PF'](granule)
            pred_features[tau] = pred
            all_preds.append(pred)

        concat_preds = torch.cat(all_preds, dim=1)
        correction = self.W_Ch(concat_preds)

        if self.feedback_act == 'tanh':
            correction = torch.tanh(correction)
        elif self.feedback_act == 'sigmoid':
            correction = torch.sigmoid(correction)

        return correction, pred_features

    def reset_buffer(self, batch_size: int):
        pass