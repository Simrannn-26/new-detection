"""
Real-Time Object Detection System using YOLOv8 + OpenCV
Author: Computer Vision Engineer
Requirements: ultralytics, opencv-python, torch
"""

import cv2
import torch
import time
import argparse
import numpy as np
from ultralytics import YOLO
from collections import deque


# ─────────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────────

def load_model(model_size: str = "n", device: str = "auto") -> tuple[YOLO, str]:
    """
    Load a pretrained YOLOv8 model.

    Args:
        model_size: One of 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        device:     'auto' | 'cuda' | 'cpu' | '0' (GPU index)

    Returns:
        (model, resolved_device)
    """
    model_map = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
    }
    if model_size not in model_map:
        raise ValueError(f"Invalid model_size '{model_size}'. Choose from: {list(model_map)}")

    weights = model_map[model_size]
    print(f"[INFO] Loading YOLOv8-{model_size.upper()} weights: {weights}")

    model = YOLO(weights)  # auto-downloads on first run

    # Resolve device
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device

    model.to(resolved)
    print(f"[INFO] Running on: {resolved.upper()}")

    if resolved == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU: {gpu_name}")

    return model, resolved


def load_custom_model(weights_path: str, device: str = "auto") -> tuple[YOLO, str]:
    """
    Load a custom fine-tuned YOLOv8 model from a local .pt file.

    Args:
        weights_path: Path to your fine-tuned .pt file
        device:       'auto' | 'cuda' | 'cpu'
    """
    print(f"[INFO] Loading custom weights: {weights_path}")
    model = YOLO(weights_path)

    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(resolved)
    print(f"[INFO] Running on: {resolved.upper()}")
    return model, resolved


# ─────────────────────────────────────────────
# 2. INFERENCE
# ─────────────────────────────────────────────

def run_inference(
    model: YOLO,
    frame: np.ndarray,
    input_size: int = 640,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    device: str = "cpu",
) -> list[dict]:
    """
    Run YOLOv8 inference on a single frame.

    Args:
        model:          Loaded YOLO model
        frame:          BGR frame from OpenCV
        input_size:     Resize input to this (smaller = faster)
        conf_threshold: Minimum confidence to keep a detection
        iou_threshold:  NMS IoU threshold
        device:         'cuda' or 'cpu'

    Returns:
        List of dicts: {label, confidence, bbox: (x1, y1, x2, y2)}
    """
    results = model.predict(
        source=frame,
        imgsz=input_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False,
        half=(device == "cuda"),   # FP16 on GPU for speed
        augment=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
            })

    return detections


# ─────────────────────────────────────────────
# 3. VISUALIZATION
# ─────────────────────────────────────────────

# Precompute a color palette per class (up to 80 COCO classes)
PALETTE = np.random.default_rng(42).integers(80, 255, size=(80, 3), dtype=np.uint8).tolist()


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    fps: float,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, confidence scores, and FPS on the frame.

    Args:
        frame:      BGR frame
        detections: Output from run_inference()
        fps:        Current FPS to overlay
        show_conf:  Whether to include confidence % in label

    Returns:
        Annotated BGR frame
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        cls_id = det["class_id"] % len(PALETTE)
        color = tuple(PALETTE[cls_id])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {conf:.0%}" if show_conf else label
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            frame, text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2, cv2.LINE_AA,
        )

    # FPS counter (top-left)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # Object count
    count_text = f"Objects: {len(detections)}"
    cv2.putText(frame, count_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 255), 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# 4. MAIN DETECTION LOOP
# ─────────────────────────────────────────────

def run_webcam(
    model: YOLO,
    device: str,
    source: int | str = 0,
    input_size: int = 640,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    skip_frames: int = 0,
    window_name: str = "YOLOv8 Real-Time Detection",
):
    """
    Main loop: capture frames, run inference, display results.

    Args:
        model:          Loaded YOLO model
        device:         'cuda' or 'cpu'
        source:         Webcam index (0) or video file path
        input_size:     YOLO inference input resolution
        conf_threshold: Detection confidence threshold
        iou_threshold:  NMS IoU threshold
        skip_frames:    Process every N+1 frames (0 = every frame)
        window_name:    OpenCV window title
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Try to set webcam to max resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize capture lag

    fps_buffer = deque(maxlen=30)          # rolling FPS average
    frame_count = 0
    last_detections = []

    print("\n[INFO] Starting detection. Press 'q' to quit, 's' to save screenshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed — retrying...")
            continue

        t0 = time.perf_counter()
        frame_count += 1

        # Frame-skip: reuse last detections for intermediate frames
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            detections = last_detections
        else:
            detections = run_inference(
                model, frame,
                input_size=input_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )
            last_detections = detections

        # FPS calculation
        elapsed = time.perf_counter() - t0
        fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = sum(fps_buffer) / len(fps_buffer)

        # Draw and show
        annotated = draw_detections(frame, detections, fps)
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit signal received.")
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[INFO] Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")


# ─────────────────────────────────────────────
# 5. FINE-TUNING HELPER
# ─────────────────────────────────────────────

def fine_tune(
    base_model_size: str = "n",
    data_yaml: str = "data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project: str = "runs/finetune",
    name: str = "exp",
):
    """
    Fine-tune a pretrained YOLOv8 on your custom dataset.

    Args:
        base_model_size: 'n' | 's' | 'm' | 'l' | 'x'
        data_yaml:       Path to your dataset YAML (Ultralytics format)
        epochs:          Training epochs
        imgsz:           Training image size
        batch:           Batch size (-1 = auto)
        device:          'auto' | 'cuda' | 'cpu'
        project:         Output directory
        name:            Run name
    """
    model_map = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt",
                 "l": "yolov8l.pt", "x": "yolov8x.pt"}
    model = YOLO(model_map[base_model_size])

    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved,
        project=project,
        name=name,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        cache=True,           # cache images in RAM for speed
        workers=4,
    )
    print(f"[INFO] Training complete. Best weights: {results.save_dir}/weights/best.pt")
    return results


# ─────────────────────────────────────────────
# 6. CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Detector")

    parser.add_argument("--model",      default="n",    choices=["n","s","m","l","x"],
                        help="Model size: n=nano, s=small, m=medium, l=large, x=xlarge")
    parser.add_argument("--weights",    default=None,
                        help="Path to custom .pt weights (overrides --model)")
    parser.add_argument("--source",     default="0",
                        help="Webcam index (0) or video file path")
    parser.add_argument("--device",     default="auto",
                        help="Device: auto | cuda | cpu | 0")
    parser.add_argument("--imgsz",      default=640,    type=int,
                        help="Inference input size (smaller = faster)")
    parser.add_argument("--conf",       default=0.4,    type=float,
                        help="Confidence threshold")
    parser.add_argument("--iou",        default=0.45,   type=float,
                        help="NMS IoU threshold")
    parser.add_argument("--skip",       default=0,      type=int,
                        help="Skip N frames between inference (0 = process all)")
    parser.add_argument("--finetune",   action="store_true",
                        help="Run fine-tuning instead of detection")
    parser.add_argument("--data",       default="data.yaml",
                        help="Dataset YAML for fine-tuning")
    parser.add_argument("--epochs",     default=50,     type=int)
    parser.add_argument("--batch",      default=16,     type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.finetune:
        # ── Fine-tuning mode ──────────────────────────
        fine_tune(
            base_model_size=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    else:
        # ── Detection mode ────────────────────────────
        if args.weights:
            model, device = load_custom_model(args.weights, args.device)
        else:
            model, device = load_model(args.model, args.device)

        # Parse source (int if digit, else string path)
        source = int(args.source) if args.source.isdigit() else args.source

        run_webcam(
            model=model,
            device=device,
            source=source,
            input_size=args.imgsz,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            skip_frames=args.skip,
        )
