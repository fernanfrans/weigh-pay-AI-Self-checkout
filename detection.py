import cv2
import torch
import numpy as np
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
import time
import pandas as pd

# YOLOv7 specific imports
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    MODEL_PATH: str = r"C:\Users\Administrator\OneDrive\Documents\Self-ChickOut\yolov7.pt"
    FRAME_WIDTH: int = 1024
    FRAME_HEIGHT: int = 576
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.3
    FONT_SCALE: float = 1
    FONT_THICKNESS: int = 2

config = Config()

@dataclass
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

class Detections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = None

    def filter(self, mask, inplace=False):
        if inplace:
            self.xyxy = self.xyxy[mask]
            self.confidence = self.confidence[mask]
            self.class_id = self.class_id[mask]
            if self.tracker_id is not None:
                self.tracker_id = self.tracker_id[mask]
        else:
            return Detections(
                self.xyxy[mask],
                self.confidence[mask],
                self.class_id[mask]
            )

    def __iter__(self):
        for xyxy, confidence, class_id, tracker_id in zip(
            self.xyxy, self.confidence, self.class_id, self.tracker_id if self.tracker_id.size > 0 else [None] * len(self.xyxy)
        ):
            yield xyxy, confidence, class_id, tracker_id

class ObjectDetector:
    def __init__(self, config: Config):
        self.config = config
        self.device = select_device('')
        self.model = self._load_model()
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        
        # Correct handling of self.names assignment
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Ensure self.names is a dictionary
        if isinstance(self.names, list):
            self.names = {i: name for i, name in enumerate(self.names)}
        
        self.imgsz = check_img_size(736, s=self.model.stride.max())
        logger.info(f"Input image size: {self.imgsz}")

    def _load_model(self) -> torch.nn.Module:
        model = attempt_load(self.config.MODEL_PATH, map_location=self.device)
        logger.info(f"Model loaded. Number of classes: {len(model.names)}")
        logger.info(f"Class names: {model.names}")
        return model

    @staticmethod
    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]
        ))

    @staticmethod
    def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
        return np.array([
            track.tlbr
            for track
            in tracks
        ], dtype=float)  # Use np.float64 for NumPy scalar type

    @staticmethod
    def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]
    ) -> List[int]:
        if not np.any(detections.xyxy) or len(tracks) == 0:
            return [None] * len(detections.xyxy)

        tracks_boxes = ObjectDetector.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detections.xyxy)
        track2detection = np.argmax(iou, axis=1)

        tracker_ids = [None] * len(detections.xyxy)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id

        return tracker_ids

    @staticmethod
    def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, size=3).tolist())

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        class_counts = {}
        track_ids = []
        chick_counts = {}

        
        # dict mapping class_id to class_name
        CLASS_NAMES_DICT = self.names
        # class_ids of interest - assuming all classes are of interest
        CLASS_ID = list(CLASS_NAMES_DICT.keys())

        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        start_time = time.time()

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        pred = non_max_suppression(pred, self.config.CONFIDENCE_THRESHOLD, self.config.IOU_THRESHOLD)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if len(pred[0]) > 0:
            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            detections = Detections(
                xyxy=det[:, :4].cpu().numpy(),
                confidence=det[:, 4].cpu().numpy(),
                class_id=det[:, 5].cpu().numpy().astype(int)
            )

            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # tracking detections
            tracks = self.byte_tracker.update(
                output_results=self.detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = self.match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # format custom labels and draw bounding boxes
            for xyxy, confidence, class_id, tracker_id in detections:
                x1, y1, x2, y2 = xyxy.astype(int)
                class_name = CLASS_NAMES_DICT[class_id]
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                chick_counts[tracker_id] = class_name
                track_ids.append(tracker_id)

                color = self.get_color_for_class(class_id)
                label = f"#{tracker_id} {class_name} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw class counts
            for i, (class_name, count) in enumerate(class_counts.items()):
                cv2.putText(frame, f"{class_name}: {count}", (10, 30 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (255, 255, 255), self.config.FONT_THICKNESS)

        
        return frame, inference_time, class_counts

def initialize_camera(config: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video capture.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap



global_class_counts = {}  # Global variable to store class counts

def get_detection_results():
    # Convert the global_class_counts dictionary to a format better suited for a table
    data = [
        {'Item': class_name, 'Count': count} 
        for class_name, count in global_class_counts.items()
    ]
    return data

def main():
    detector = ObjectDetector(config)
    cap = initialize_camera(config)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame.")
                break

            # Process the frame
            processed_frame, inference_time, class_counts = detector.process_frame(frame)

            global_class_counts.clear()  # Reset counts for the new frame
            for class_name, count in class_counts.items():
                global_class_counts[class_name] = count

            # Encode the frame to bytes for streaming
            _, jpeg_frame = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n\r\n')

            # Optionally log or access global class counts
            logger.info(f"Global class counts: {global_class_counts}")

            if cv2.waitKey(1) == 27:  # ESC key
                break

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        cap.release()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
