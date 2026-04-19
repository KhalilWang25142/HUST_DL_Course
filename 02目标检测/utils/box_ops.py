from __future__ import annotations

import torch


def encode_boxes(proposals: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    proposal_widths = proposals[:, 2] - proposals[:, 0]
    proposal_heights = proposals[:, 3] - proposals[:, 1]
    proposal_ctr_x = proposals[:, 0] + 0.5 * proposal_widths
    proposal_ctr_y = proposals[:, 1] + 0.5 * proposal_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - proposal_ctr_x) / proposal_widths.clamp(min=1e-6)
    dy = (gt_ctr_y - proposal_ctr_y) / proposal_heights.clamp(min=1e-6)
    dw = torch.log(gt_widths.clamp(min=1e-6) / proposal_widths.clamp(min=1e-6))
    dh = torch.log(gt_heights.clamp(min=1e-6) / proposal_heights.clamp(min=1e-6))
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    proposal_widths = proposals[:, 2] - proposals[:, 0]
    proposal_heights = proposals[:, 3] - proposals[:, 1]
    proposal_ctr_x = proposals[:, 0] + 0.5 * proposal_widths
    proposal_ctr_y = proposals[:, 1] + 0.5 * proposal_heights

    dx, dy, dw, dh = deltas.unbind(dim=1)
    pred_ctr_x = dx * proposal_widths + proposal_ctr_x
    pred_ctr_y = dy * proposal_heights + proposal_ctr_y
    pred_w = torch.exp(dw) * proposal_widths
    pred_h = torch.exp(dh) * proposal_heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=1)


def clamp_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
    return boxes

