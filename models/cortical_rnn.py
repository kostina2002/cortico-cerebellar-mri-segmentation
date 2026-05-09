import torch
import torch.nn as nn
from typing import Optional
from models.cerebellar_module import CerebellarModule

class CorticalRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, alpha: float = 0.1,
                 freeze_recurrent: bool = True, freeze_input: bool = False,
                 feedback_type: str = 'none'):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.feedback_type = feedback_type

        self.W_hh = nn.Linear(hidden_size, hidden_size)
        if freeze_recurrent:
            self.W_hh.requires_grad_(False)
            nn.init.orthogonal_(self.W_hh.weight)

        self.W_ih = nn.Linear(input_dim, hidden_size)
        if freeze_input:
            self.W_ih.requires_grad_(False)

        if feedback_type == 'readout':
            self.W_rd = nn.Linear(hidden_size, hidden_size)
            self.W_zh = nn.Linear(hidden_size, hidden_size)
            nn.init.kaiming_uniform_(self.W_rd.weight)
            nn.init.kaiming_uniform_(self.W_zh.weight)

        self.activation = nn.Tanh()

    def forward(self, inputs: torch.Tensor,
                cerebellum: Optional[CerebellarModule] = None,
                ablate: bool = False,
                return_cereb_preds: bool = False,
                detach_cereb_input: bool = False):
        B, T, _ = inputs.shape
        device = inputs.device

        h_lin = torch.zeros(B, self.hidden_size, device=device)

        if cerebellum is not None and not ablate:
            cerebellum.reset_buffer(B)

        outputs = []
        cereb_preds_list = [] if return_cereb_preds else None

        for t in range(T):
            x_t = inputs[:, t]
            f_h_prev = self.activation(h_lin)

            if self.feedback_type == 'cerebellar' and cerebellum is not None and not ablate:
                correction, pred_features = cerebellum(f_h_prev, detach_input=detach_cereb_input)
                if return_cereb_preds:
                    cereb_preds_list.append(pred_features)
            elif self.feedback_type == 'readout':
                readout_t = self.W_rd(f_h_prev)
                correction = self.W_zh(readout_t)
                if return_cereb_preds:
                    cereb_preds_list.append({})
            else:
                correction = 0.0
                if return_cereb_preds:
                    cereb_preds_list.append({})

            recurrent = self.W_hh(f_h_prev)
            h_lin_new = self.alpha * h_lin + recurrent + self.W_ih(x_t) + correction
            f_h = self.activation(h_lin_new)
            outputs.append(f_h)
            h_lin = h_lin_new

        f_h_seq = torch.stack(outputs, dim=1)
        return f_h_seq, cereb_preds_list