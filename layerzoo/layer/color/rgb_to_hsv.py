import torch
  
from ..base import Layer


def compute_rgb_to_hsv(x, eps=1e-4):
    """
    Convert RGB to HSV.

    Input:
        Red    0..1
        Green  0..1
        Blue   0..1

    Output:
        Hue         0..1
        Saturation  0..1
        Value       0..1
    """
    batch_size, channels = x.shape[:2]
    assert channels == 3

    r, g, b = x.unbind(1)

    max_c = r.max(g).max(b) - eps
    min_c = r.min(g).min(b)

    is_gray_case = max_c < min_c
    h = r * 0
    s = r * 0
    v = max_c
    if_gray_case = torch.stack([h, s, v], 1)
    gray_case = is_gray_case.float().unsqueeze(1) * if_gray_case

    s = (max_c - min_c) / max_c
    rc = (max_c - r) / (max_c - min_c)
    gc = (max_c - g) / (max_c - min_c)
    bc = (max_c - b) / (max_c - min_c)

    is_red_case = max_c < r
    h = bc - gc
    if_red_case = torch.stack([h, s, v], 1)
    red_case = is_red_case.float().unsqueeze(1) * if_red_case

    is_green_case = max_c < g
    h = 2 + rc - bc
    if_green_case = torch.stack([h, s, v], 1)
    green_case = is_green_case.float().unsqueeze(1) * if_green_case

    is_blue_case = max_c < b
    h = 4 + gc - rc
    if_blue_case = torch.stack([h, s, v], 1)
    blue_case = is_blue_case.float().unsqueeze(1) * if_blue_case

    y = gray_case + red_case + green_case + blue_case
    y[:, 0] = (y[:, 0] / 6) % 1
    return y


class RGBToHSV(Layer):
    def forward_inner(self, x):
        return compute_rgb_to_hsv()
