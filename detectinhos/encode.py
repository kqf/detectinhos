from typing import List, Tuple, Union

import torch


def point_form(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            boxes[..., :2] - boxes[..., 2:] / 2,
            boxes[..., :2] + boxes[..., 2:] / 2,
        ),
        dim=-1,
    )


def encode(
    boxes: torch.Tensor,
    priors: torch.Tensor,
    variances: List[float],
) -> torch.Tensor:
    # dist b/t match center and prior's center
    g_cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - priors[..., :2]
    # encode variance
    g_cxcy /= variances[0] * priors[..., 2:]
    # match wh / prior wh
    g_wh = (boxes[:, 2:] - boxes[:, :2]) / priors[..., 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(
    boxes: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[..., :2] + boxes[..., :2] * variances[0] * priors[..., 2:],
            priors[..., 2:] * torch.exp(boxes[..., 2:] * variances[1]),
        ),
        dim=-1,
    )
    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]

    return boxes


def to_shape(x, matched) -> torch.Tensor:
    return x.unsqueeze(1).expand(matched.shape[0], 5).unsqueeze(2)


def encode_landmarks(
    landmarks: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    # dist b/t match center and prior's center

    # Reshape matched to [num_priors, 5, 2]
    landmarks = torch.reshape(landmarks, (landmarks.shape[0], 5, 2))

    # Expand priors to match the shape of matched using broadcasting
    # Change shape from [num_priors, 4] to [num_priors, 1, 4]

    priors = priors[:, None]

    # Calculate the distance between the match center and the prior's center
    g_cxcy = landmarks[:, :, :2] - priors[:, :, :2]

    # Encode variance
    g_cxcy /= variances[0] * priors[:, :, 2:]

    # Return target for smooth_l1_loss
    return g_cxcy.reshape(-1, 10)


def decode_landmarks(
    landmarks: torch.Tensor,
    priors: torch.Tensor,
    variances: Union[List[float], Tuple[float, float]],
) -> torch.Tensor:
    x = landmarks
    return torch.cat(
        (
            priors[..., :2] + x[..., :2] * variances[0] * priors[..., 2:],
            priors[..., :2] + x[..., 2:4] * variances[0] * priors[..., 2:],
            priors[..., :2] + x[..., 4:6] * variances[0] * priors[..., 2:],
            priors[..., :2] + x[..., 6:8] * variances[0] * priors[..., 2:],
            priors[..., :2] + x[..., 8:10] * variances[0] * priors[..., 2:],
        ),
        dim=-1,
    )
