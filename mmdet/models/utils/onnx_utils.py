import torch


def fmod(x: torch.Tensor, value: float):
    return x - (x / value).int().float() * value


def obbox2corners(rboxs: torch.Tensor):
    cx, cy, w, h, alpha = torch.split(rboxs, 1, 1)
    cos_a_half = torch.cos(alpha) * 0.5
    sin_a_half = torch.sin(alpha) * 0.5
    w_x = cos_a_half * w
    w_y = sin_a_half * w
    h_x = -sin_a_half * h
    h_y = cos_a_half * h
    return torch.cat([cx + w_x + h_x, cy + w_y + h_y,
                      cx + w_x - h_x, cy + w_y - h_y,
                      cx - w_x - h_x, cy - w_y - h_y,
                      cx - w_x + h_x, cy - w_y + h_y], dim=1)
