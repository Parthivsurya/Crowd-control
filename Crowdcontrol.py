#!/usr/bin/env python3
"""
Tile-based dense-crowd detector + heat-map + tracking + alert (batched for speed)
Now with timing breakdown and tile toggle
"""
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
import itertools
import random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="crowd.mp4", help="Path to video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    p.add_argument("--tile", type=int, default=320, help="Tile size")
    p.add_argument("--overlap", type=float, default=0.25, help="Tile overlap")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--alert", type=int, default=60, help="Alert count")
    p.add_argument("--gridh", type=int, default=18, help="Heatmap rows")
    p.add_argument("--gridw", type=int, default=32, help="Heatmap cols")
    p.add_argument("--no_tile", action="store_true", help="Disable tiling (full frame detection)")
    return p.parse_args()

def choose_device():
    print("Torch MPS available:", torch.backends.mps.is_available())
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_color(idx):
    random.seed(idx)
    return tuple(random.randint(64, 255) for _ in range(3))

def draw_tracks(frame, tracks):
    for tr in tracks:
        if not tr.is_confirmed(): continue
        x1, y1, x2, y2 = map(int, tr.to_ltrb())
        color = get_color(tr.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tr.track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    args = parse_args()
    device = choose_device()
    print(f"[INFO] Using device: {device}")

    model = YOLO(args.model)
    model.to(device)
    model.fuse()

    tracker = DeepSort(max_age=15)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not args.no_tile:
        stride = int(args.tile * (1 - args.overlap))
        xs = list(range(0, W, stride))
        ys = list(range(0, H, stride))

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret: break

        det_t0 = time.time()
        dets = []

        # ✅ Option 1: Use Tiling
        if not args.no_tile:
            tiles = []
            tile_positions = []
            for y0, x0 in itertools.product(ys, xs):
                y1 = min(y0 + args.tile, H)
                x1 = min(x0 + args.tile, W)
                tile = frame[y0:y1, x0:x1]
                if tile.size == 0: continue
                tiles.append(tile)
                tile_positions.append((x0, y0))

            results = model(tiles, conf=args.conf, iou=args.iou, device=device, verbose=False)

            for res, (x0, y0) in zip(results, tile_positions):
                for b in res.boxes:
                    if int(b.cls[0]) != 0: continue
                    xy = b.xyxy[0].cpu().numpy()
                    xy[[0,2]] += x0
                    xy[[1,3]] += y0
                    conf = b.conf.item()
                    dets.append([*xy, conf])

        # ✅ Option 2: Full Frame
        else:
            results = model(frame, conf=args.conf, iou=args.iou, device=device, verbose=False)
            for b in results[0].boxes:
                if int(b.cls[0]) != 0: continue
                xy = b.xyxy[0].cpu().numpy()
                conf = b.conf.item()
                dets.append([*xy, conf])

        detect_done = time.time()

        # ✅ Tracking
        det_arr = np.array(dets)
        tracks = tracker.update_tracks(
            [[list(map(float, d[:4])), float(d[4]), 'person'] for d in det_arr] if len(det_arr) else [],
            frame=frame
        )

        track_done = time.time()

        # ✅ Heatmap
        dh, dw = args.gridh, args.gridw
        density = np.zeros((dh, dw), dtype=np.float32)
        for d in det_arr:
            cx = int(((d[0]+d[2])/2) / W * dw)
            cy = int(((d[1]+d[3])/2) / H * dh)
            cx, cy = np.clip([cx, cy], 0, [dw-1, dh-1])
            density[cy, cx] += 1
        density = cv2.GaussianBlur(density, (0, 0), 1.3)
        count = int(density.sum())

        draw_tracks(frame, tracks)

        # ✅ Overlay heatmap
        heat = cv2.resize(density, (W, H))
        if heat.max() > 0:
            hn = (heat / heat.max() * 255).astype(np.uint8)
            hc = cv2.applyColorMap(hn, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.75, hc, 0.25, 0)

        # ✅ Alert and display
        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        if count > args.alert:
            cv2.putText(frame, "!!! ALERT CROWD !!!", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Crowd Detector", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        end = time.time()
        print(f"[TIMING] Detection: {(detect_done - det_t0):.2f}s | Tracking: {(track_done - detect_done):.2f}s | Total: {(end - start):.2f}s")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()