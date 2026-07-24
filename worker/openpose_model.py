from __future__ import annotations

import math
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import List, NamedTuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

# Vendored from the CMU OpenPose body network (via Hzzone/pytorch-openpose and
# lllyasviel/ControlNet's controlnet_aux). The pretrained checkpoint below is
# CMU-licensed for non-commercial use only -- see the project README.
CHECKPOINT_URL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
CHECKPOINT_NAME = "openpose_body_pose_model.pth"
DETECT_RESOLUTION = 512

LIMB_SEQUENCE = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18],
]
MAP_IDX = [
    [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
    [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
    [55, 56], [37, 38], [45, 46],
]
LIMB_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
]


class Keypoint(NamedTuple):
    x: float
    y: float


class BodyResult(NamedTuple):
    keypoints: List[Union[Keypoint, None]]


def _make_layers(block: OrderedDict, no_relu_layers: list[str]) -> nn.Sequential:
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layers.append((layer_name, nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])))
        else:
            layers.append((layer_name, nn.Conv2d(v[0], v[1], kernel_size=v[2], stride=v[3], padding=v[4])))
            if layer_name not in no_relu_layers:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


class BodyPoseNet(nn.Module):
    """VGG19 backbone + 6-stage PAF/heatmap refinement (attribute names must match
    the pretrained checkpoint's structure, remapped via `transfer_state_dict`)."""

    def __init__(self) -> None:
        super().__init__()
        no_relu_layers = [
            "conv5_5_CPM_L1", "conv5_5_CPM_L2", "Mconv7_stage2_L1", "Mconv7_stage2_L2",
            "Mconv7_stage3_L1", "Mconv7_stage3_L2", "Mconv7_stage4_L1", "Mconv7_stage4_L2",
            "Mconv7_stage5_L1", "Mconv7_stage5_L2", "Mconv7_stage6_L1", "Mconv7_stage6_L1",
        ]
        blocks: dict[str, OrderedDict] = {}
        block0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]), ("conv1_2", [64, 64, 3, 1, 1]), ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]), ("conv2_2", [128, 128, 3, 1, 1]), ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]), ("conv3_2", [256, 256, 3, 1, 1]), ("conv3_3", [256, 256, 3, 1, 1]),
            ("conv3_4", [256, 256, 3, 1, 1]), ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]), ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3_CPM", [512, 256, 3, 1, 1]), ("conv4_4_CPM", [256, 128, 3, 1, 1]),
        ])
        blocks["block1_1"] = OrderedDict([
            ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]), ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]), ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
            ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
        ])
        blocks["block1_2"] = OrderedDict([
            ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]), ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]), ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
            ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
        ])
        self.model0 = _make_layers(block0, no_relu_layers)

        for i in range(2, 7):
            blocks["block%d_1" % i] = OrderedDict([
                ("Mconv1_stage%d_L1" % i, [185, 128, 7, 1, 3]), ("Mconv2_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d_L1" % i, [128, 128, 7, 1, 3]), ("Mconv4_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d_L1" % i, [128, 128, 7, 1, 3]), ("Mconv6_stage%d_L1" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d_L1" % i, [128, 38, 1, 1, 0]),
            ])
            blocks["block%d_2" % i] = OrderedDict([
                ("Mconv1_stage%d_L2" % i, [185, 128, 7, 1, 3]), ("Mconv2_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d_L2" % i, [128, 128, 7, 1, 3]), ("Mconv4_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d_L2" % i, [128, 128, 7, 1, 3]), ("Mconv6_stage%d_L2" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d_L2" % i, [128, 19, 1, 1, 0]),
            ])

        for name in blocks:
            blocks[name] = _make_layers(blocks[name], no_relu_layers)

        self.model1_1, self.model2_1, self.model3_1 = blocks["block1_1"], blocks["block2_1"], blocks["block3_1"]
        self.model4_1, self.model5_1, self.model6_1 = blocks["block4_1"], blocks["block5_1"], blocks["block6_1"]
        self.model1_2, self.model2_2, self.model3_2 = blocks["block1_2"], blocks["block2_2"], blocks["block3_2"]
        self.model4_2, self.model5_2, self.model6_2 = blocks["block4_2"], blocks["block5_2"], blocks["block6_2"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out1 = self.model0(x)
        out1_1, out1_2 = self.model1_1(out1), self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2_1, out2_2 = self.model2_1(out2), self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1, out3_2 = self.model3_1(out3), self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1, out4_2 = self.model4_1(out4), self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1, out5_2 = self.model5_1(out5), self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1, out6_2 = self.model6_1(out6), self.model6_2(out6)
        return out6_1, out6_2


def transfer_state_dict(model: nn.Module, checkpoint_weights: dict) -> dict:
    # The checkpoint's keys drop the top-level submodule name (e.g. "model0.").
    remapped = {}
    for name in model.state_dict().keys():
        remapped[name] = checkpoint_weights[".".join(name.split(".")[1:])]
    return remapped


def load_pose_model(cache_dir: Path, device: torch.device) -> BodyPoseNet:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / CHECKPOINT_NAME
    if not checkpoint.exists():
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    model = BodyPoseNet()
    model.load_state_dict(transfer_state_dict(model, torch.load(checkpoint, map_location="cpu")))
    return model.to(device).eval()


def _resize_planes(array: np.ndarray, size: tuple[int, int] | None = None, fx: float = 1.0, fy: float = 1.0) -> np.ndarray:
    # cv2.resize only supports up to 4 channels; heatmap (19ch) and PAF (38ch) need a per-channel loop.
    kwargs = {"dsize": size} if size is not None else {"dsize": None, "fx": fx, "fy": fy}
    return np.stack([cv2.resize(array[:, :, i], interpolation=cv2.INTER_CUBIC, **kwargs) for i in range(array.shape[2])], axis=2)


def _pad_right_down(img: np.ndarray, stride: int, pad_value: int) -> tuple[np.ndarray, list[int]]:
    h, w = img.shape[:2]
    pad = [0, 0, (0 if h % stride == 0 else stride - h % stride), (0 if w % stride == 0 else stride - w % stride)]
    padded = np.pad(img, ((0, pad[2]), (0, pad[3]), (0, 0)), mode="constant", constant_values=pad_value)
    return padded, pad


def _estimate(model: BodyPoseNet, frame_bgr: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Port of pytorch-openpose's Body.__call__: single-scale PAF/heatmap inference
    + greedy limb assembly. gaussian_filter (scipy) is replaced with cv2.GaussianBlur."""
    box_size, stride, pad_value, thre1, thre2 = 368, 8, 128, 0.1, 0.05
    scale = 0.5 * box_size / frame_bgr.shape[0]
    height, width = frame_bgr.shape[:2]
    heatmap_avg = np.zeros((height, width, 19))
    paf_avg = np.zeros((height, width, 38))

    resized = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    padded, pad = _pad_right_down(resized, stride, pad_value)
    tensor = np.transpose(np.float32(padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
    tensor = torch.from_numpy(np.ascontiguousarray(tensor)).float().to(device)

    with torch.no_grad():
        paf_out, heatmap_out = model(tensor)
    heatmap_out = heatmap_out.cpu().numpy()
    paf_out = paf_out.cpu().numpy()

    heatmap = np.transpose(np.squeeze(heatmap_out), (1, 2, 0))
    heatmap = _resize_planes(heatmap, fx=stride, fy=stride)
    heatmap = heatmap[: padded.shape[0] - pad[2], : padded.shape[1] - pad[3], :]
    heatmap = _resize_planes(heatmap, size=(width, height))

    paf = np.transpose(np.squeeze(paf_out), (1, 2, 0))
    paf = _resize_planes(paf, fx=stride, fy=stride)
    paf = paf[: padded.shape[0] - pad[2], : padded.shape[1] - pad[3], :]
    paf = _resize_planes(paf, size=(width, height))

    heatmap_avg += heatmap
    paf_avg += paf

    all_peaks = []
    peak_counter = 0
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        smoothed = cv2.GaussianBlur(map_ori.astype(np.float32), (0, 0), sigmaX=3)

        map_left, map_right = np.zeros_like(smoothed), np.zeros_like(smoothed)
        map_up, map_down = np.zeros_like(smoothed), np.zeros_like(smoothed)
        map_left[1:, :] = smoothed[:-1, :]
        map_right[:-1, :] = smoothed[1:, :]
        map_up[:, 1:] = smoothed[:, :-1]
        map_down[:, :-1] = smoothed[:, 1:]

        peaks_binary = np.logical_and.reduce((
            smoothed >= map_left, smoothed >= map_right, smoothed >= map_up, smoothed >= map_down, smoothed > thre1,
        ))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_ids = range(peak_counter, peak_counter + len(peaks))
        all_peaks.append([peaks_with_score[i] + (peak_ids[i],) for i in range(len(peak_ids))])
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(len(MAP_IDX)):
        score_mid = paf_avg[:, :, [x - 19 for x in MAP_IDX[k]]]
        cand_a = all_peaks[LIMB_SEQUENCE[k][0] - 1]
        cand_b = all_peaks[LIMB_SEQUENCE[k][1] - 1]
        n_a, n_b = len(cand_a), len(cand_b)
        if n_a != 0 and n_b != 0:
            candidates = []
            for i in range(n_a):
                for j in range(n_b):
                    vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = max(0.001, math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]))
                    vec = np.divide(vec, norm)
                    startend = list(zip(
                        np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                        np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num),
                    ))
                    vec_x = np.array([score_mid[int(round(startend[t][1])), int(round(startend[t][0])), 0] for t in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[t][1])), int(round(startend[t][0])), 1] for t in range(len(startend))])
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        candidates.append([i, j, score_with_dist_prior, score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

            candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in candidates:
                i, j, s = c[0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(connection) >= min(n_a, n_b):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    for k in range(len(MAP_IDX)):
        if k in special_k:
            continue
        part_as, part_bs = connection_all[k][:, 0], connection_all[k][:, 1]
        index_a, index_b = np.array(LIMB_SEQUENCE[k]) - 1
        for i in range(len(connection_all[k])):
            found = 0
            subset_idx = [-1, -1]
            for j in range(len(subset)):
                if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                    subset_idx[found] = j
                    found += 1
            if found == 1:
                j = subset_idx[0]
                if subset[j][index_b] != part_bs[i]:
                    subset[j][index_b] = part_bs[i]
                    subset[j][-1] += 1
                    subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
            elif found == 2:
                j1, j2 = subset_idx
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                if len(np.nonzero(membership == 2)[0]) == 0:
                    subset[j1][:-2] += subset[j2][:-2] + 1
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, 0)
                else:
                    subset[j1][index_b] = part_bs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
            elif not found and k < 17:
                row = -1 * np.ones(20)
                row[index_a] = part_as[i]
                row[index_b] = part_bs[i]
                row[-1] = 2
                row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                subset = np.vstack([subset, row])

    delete_idx = [i for i in range(len(subset)) if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4]
    subset = np.delete(subset, delete_idx, axis=0)
    return candidate, subset


def _format_bodies(candidate: np.ndarray, subset: np.ndarray) -> List[BodyResult]:
    return [
        BodyResult(
            keypoints=[
                Keypoint(x=candidate[idx][0], y=candidate[idx][1]) if idx != -1 else None
                for idx in person[:18].astype(int)
            ],
        )
        for person in subset
    ]


def draw_bodypose(canvas: np.ndarray, keypoints: List[Union[Keypoint, None]]) -> np.ndarray:
    height, width = canvas.shape[:2]
    stick_width = 4
    for (a, b), color in zip(LIMB_SEQUENCE, LIMB_COLORS):
        k1, k2 = keypoints[a - 1], keypoints[b - 1]
        if k1 is None or k2 is None:
            continue
        y = np.array([k1.x, k2.x]) * float(width)
        x = np.array([k1.y, k2.y]) * float(height)
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
        polygon = cv2.ellipse2Poly((int(y.mean()), int(x.mean())), (int(length / 2), stick_width), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(c * 0.6) for c in color])
    for keypoint, color in zip(keypoints, LIMB_COLORS):
        if keypoint is None:
            continue
        cv2.circle(canvas, (int(keypoint.x * width), int(keypoint.y * height)), 4, color, thickness=-1)
    return canvas


def detect_pose(model: BodyPoseNet, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Returns a BGR uint8 frame sized to match frame_bgr: colored OpenPose skeleton on black."""
    height, width = frame_bgr.shape[:2]
    detect_scale = DETECT_RESOLUTION / min(height, width)
    detect_h = max(64, round(height * detect_scale / 64) * 64)
    detect_w = max(64, round(width * detect_scale / 64) * 64)
    interpolation = cv2.INTER_LANCZOS4 if detect_scale > 1 else cv2.INTER_AREA
    small = cv2.resize(frame_bgr, (detect_w, detect_h), interpolation=interpolation)

    candidate, subset = _estimate(model, small, device)
    bodies = _format_bodies(candidate, subset)

    canvas = np.zeros((detect_h, detect_w, 3), dtype=np.uint8)
    for body in bodies:
        normalized = [
            Keypoint(x=k.x / float(detect_w), y=k.y / float(detect_h)) if k is not None else None
            for k in body.keypoints
        ]
        canvas = draw_bodypose(canvas, normalized)

    canvas_bgr = canvas[:, :, ::-1]  # draw_bodypose's colors are RGB-ordered
    return cv2.resize(canvas_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
