import cv2 as cv
import numpy as np
import torch

from rectiou.rectiou import get_rect_vert, rect_intersection, compute_iou


def main():
    b = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rect_ab = [
        torch.tensor([[256.0, 256.0, 100.0, 100.0, 0.5]], device=device),
        torch.tensor([[300.0, 200.0, 150.0, 100.0, 0.0]], device=device),
    ]

    cv.namedWindow("canvas", cv.WINDOW_NORMAL)

    while True:
        iou = compute_iou(rect_ab[0], rect_ab[1])
        vert_a = get_rect_vert(rect_ab[0])
        vert_b = get_rect_vert(rect_ab[1])

        inter = rect_intersection(vert_a, vert_b)

        canvas = np.zeros([512, 512, 3], dtype=np.uint8)

        for i in range(2):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            vert = vert_a if i == 0 else vert_b
            vert_np = vert[0].detach().cpu().numpy().astype(np.int32)

            for j in range(4):
                cv.line(canvas, vert_np[j], vert_np[(j + 1) % 4], color, 4)

        inter_np = inter[0].detach().cpu().numpy().astype(np.int32)
        for i in range(9):
            cv.line(canvas, inter_np[i], inter_np[(i + 1) % 9], (255, 255, 255), 2)

        cv.putText(
            canvas,
            "iou: {:.2f}".format(iou[0].item()),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
        )

        cv.imshow("canvas", canvas)

        k = cv.waitKey(1)
        if k == ord("p"):
            break
        elif k == ord(" "):
            b = 1 - b
        elif k == ord("w"):
            rect_ab[b][0, 1] -= 1
        elif k == ord("s"):
            rect_ab[b][0, 1] += 1
        elif k == ord("a"):
            rect_ab[b][0, 0] -= 1
        elif k == ord("d"):
            rect_ab[b][0, 0] += 1
        elif k == ord("e"):
            rect_ab[b][0, 4] += 0.02
        elif k == ord("q"):
            rect_ab[b][0, 4] -= 0.02
        elif k == ord("z"):
            rect_ab[b][0, 2] -= 1
        elif k == ord("x"):
            rect_ab[b][0, 2] += 1
        elif k == ord("r"):
            rect_ab[b][0, 3] += 1
        elif k == ord("f"):
            rect_ab[b][0, 3] -= 1


if __name__ == "__main__":
    main()
