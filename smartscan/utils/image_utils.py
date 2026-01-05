import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from smartscan.types import ImageSource
from smartscan.constants import SupportedFileTypes
from smartscan.errors import SmartScanError, ErrorCode

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_threshold]

    return keep

def draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    outline: str = "green",
    width: int = 2,
    font: ImageFont.ImageFont = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except IOError:
            font = ImageFont.load_default()

    img_w, img_h = image.size

    # Filter by confidence
    keep_indices = np.where(scores >= conf_threshold)[0]
    if keep_indices.size == 0:
        return image

    boxes_px = (boxes[keep_indices] * [img_w, img_h, img_w, img_h]).astype(int)
    scores_keep = scores[keep_indices]

    # NMS
    keep_nms = nms(boxes_px, scores_keep, nms_threshold)
    filtered_boxes = boxes_px[keep_nms]
    filtered_scores = scores_keep[keep_nms]

    # Draw
    for (x1, y1, x2, y2), score in zip(filtered_boxes, filtered_scores):
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
        text = f"{score:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])        
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=outline)
        draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)

    return image


def crop_faces(image: Image.Image, boxes: np.ndarray, scores: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.3):
    img_w, img_h = image.size
    keep_indices = np.where(scores >= conf_threshold)[0]
    if keep_indices.size == 0:
        return []

    boxes_px = (boxes[keep_indices] * [img_w, img_h, img_w, img_h]).astype(int)
    scores_keep = scores[keep_indices]

    keep_nms = nms(boxes_px, scores_keep, nms_threshold)
    filtered_boxes = boxes_px[keep_nms]

    cropped_faces = [image.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in filtered_boxes]
    return cropped_faces

def image_source_to_pil_image(source: ImageSource) -> Image.Image:
    if isinstance(source, str):
        if source.startswith(("http", "https")):
            return Image.open(requests.get(url=source, stream = True).raw)
        elif source.endswith(SupportedFileTypes.IMAGE):
            return Image.open(source)
        else:
            raise SmartScanError("Unsupported file type", code=ErrorCode.UNSUPPORTED_FILE_TYPE, details=f"Supported file types: {SupportedFileTypes.IMAGE + SupportedFileTypes.TEXT + SupportedFileTypes.VIDEO}")
    else:
        return Image.fromarray(source)
