import typing as t

import cv2 as cv
import numpy as np
import pytest
import torch

import rectiou


def devices() -> t.Sequence[str]:
    dev = ["cpu"]
    if torch.cuda.is_available():
        dev.append("cuda")

    return dev


@pytest.fixture(params=devices())
def device(request) -> str:
    return request.param


@pytest.mark.parametrize(
    ["rect_a", "rect_b", "expected"],
    [
        [
            # Same box
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([1.0]),
        ],
        [
            # Octagon
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, torch.pi / 4]]),
            torch.tensor([0.7071067811865477]),
        ],
        [
            # Touching sides
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([0.0]),
        ],
        [
            # Touching corners
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]]),
            torch.tensor([0.0]),
        ],
    ],
)
def test_rectiou_explicit(
    rect_a: torch.Tensor,
    rect_b: torch.Tensor,
    expected: torch.Tensor,
    device: str,
):
    rect_a = rect_a.to(device)
    rect_b = rect_b.to(device)
    expected = expected.to(device)
    iou = rectiou.compute_iou(rect_a, rect_b)
    assert torch.allclose(iou, expected, atol=1e-6)


@pytest.mark.parametrize(
    "shape", [[], [1], [10], [1, 1], [1, 10], [3, 10], [10, 3], [2, 3, 2, 3]]
)
def test_rectiou_variable_shape(shape: t.Sequence[int], device: str):
    fshape = list(shape) + [5]
    rect_a = torch.rand(fshape, device=device)
    rect_b = torch.rand(fshape, device=device)
    iou = rectiou.compute_iou(rect_a, rect_b)
    rect_a_ = rect_a.view(-1, 5)
    rect_b_ = rect_b.view(-1, 5)
    iou_ = rectiou.compute_iou(rect_a_, rect_b_)
    assert torch.allclose(iou, iou_.view(fshape[:-1]))


@pytest.mark.parametrize(
    ["rect_a", "rect_b"],
    [
        [
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.5, 1.0, 1.0, 0.0]]),
        ],
        [
            torch.rand([100000, 5]).clip(0.1, 1.0),
            torch.rand([100000, 5]).clip(0.1, 1.0),
        ],
    ],
)
def test_rectiou_against_reference(rect_a: torch.Tensor, rect_b: torch.Tensor):
    def compute_iou_opencv(rect_a: torch.Tensor, rect_b: torch.Tensor):
        batchdims = rect_a.shape[:-1]
        rect_a_np = rect_a.view(-1, 5).detach().cpu().numpy()
        rect_b_np = rect_b.view(-1, 5).detach().cpu().numpy()

        ious = []
        for rect_a_, rect_b_ in zip(rect_a_np, rect_b_np):
            rect_a_[4] = -rect_a_[4] * 180.0 / np.pi
            rect_b_[4] = -rect_b_[4] * 180.0 / np.pi
            x1, y1, w1, h1, a1 = rect_a_.tolist()
            x2, y2, w2, h2, a2 = rect_b_.tolist()
            retval, cvi = cv.rotatedRectangleIntersection(
                ((x1, y1), (w1, h1), a1),
                ((x2, y2), (w2, h2), a2),
            )
            if retval == 0:
                ious.append(0.0)
            else:
                area = cv.contourArea(cvi)
                ious.append(area / (w1 * h1 + w2 * h2 - area))

        ious = np.array(ious).reshape(batchdims)
        return torch.from_numpy(ious).to(rect_a.device).to(rect_a.dtype)

    iou = rectiou.compute_iou(rect_a, rect_b)
    iou_ = compute_iou_opencv(rect_a, rect_b)
    assert torch.allclose(iou, iou_, rtol=0.0, atol=1e-3)


@pytest.mark.parametrize(
    ["rect_a", "rect_b"],
    [
        [
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.5, 1.0, 1.0, 0.0]]),
        ],
        [
            torch.rand([10, 5]).clip(0.1, 1.0),
            torch.rand([10, 5]).clip(0.1, 1.0),
        ],
    ],
)
def test_backprop(rect_a: torch.Tensor, rect_b: torch.Tensor, device: str):
    rect_a = rect_a.to(device).clone().detach().requires_grad_()
    rect_b = rect_b.to(device).clone().detach().requires_grad_()
    iou = rectiou.compute_iou(rect_a, rect_b)
    iou.sum().backward()
    assert rect_a.grad is not None
    assert rect_b.grad is not None
