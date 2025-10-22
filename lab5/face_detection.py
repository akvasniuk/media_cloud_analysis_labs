import sys
import os
import io
from PIL import Image, ExifTags
import cv2
import json
import numpy as np

def is_jpeg_signature(path):
    try:
        with open(path, 'rb') as f:
            start = f.read(2)
            f.seek(-2, os.SEEK_END)
            end = f.read(2)
        return start == b'\xff\xd8' and end == b'\xff\xd9'
    except Exception:
        return False

def validate_jpeg_with_pillow(path):
    try:
        with Image.open(path) as im:
            im.verify()
            return im.format == 'JPEG'
    except Exception:
        return False

def get_exif(path):
    exif_out = {}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return {}
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_out[tag] = value
    except Exception:
        return {}
    return exif_out

def apply_exif_orientation(pil_img, exif):
    orientation = exif.get('Orientation') if exif else None
    if not orientation:
        return pil_img
    method = {
        1: lambda x: x,
        2: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        3: lambda x: x.rotate(180, expand=True),
        4: lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        5: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
        6: lambda x: x.rotate(270, expand=True),
        7: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True),
        8: lambda x: x.rotate(90, expand=True),
    }
    return method.get(orientation, lambda x: x)(pil_img)

def pil_to_cv2(pil_img):
    rgb = pil_img.convert('RGB')
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def pretty_print_exif(exif):
    if not exif:
        print("No EXIF metadata found.")
        return
    print("EXIF metadata:")
    for k, v in exif.items():
        if isinstance(v, bytes):
            try:
                v = v.decode(errors='replace')
            except Exception:
                v = repr(v)
        print(f"  {k}: {v}")

def detect_faces_yunet(cv_img, model_path="./models/face_detection_yunet_2023mar.onnx",
                       score_threshold=0.8, nms_threshold=0.3, top_k=5000):
    """Detect faces using YuNet model."""
    h, w = cv_img.shape[:2]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
        )

    detector = cv2.FaceDetectorYN.create(
        model_path, "", (w, h), score_threshold, nms_threshold, top_k
    )
   
    faces = detector.detect(cv_img)

    if faces[1] is None:
        return cv_img.copy(), []

    out_img = cv_img.copy()
    boxes = []
    for f in faces[1]:
        x, y, bw, bh = map(int, f[:4])
        cv2.rectangle(out_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        boxes.append({'x': x, 'y': y, 'w': bw, 'h': bh})
    return out_img, boxes

def save_metadata_as_json(exif_data, boxes, json_path):
    serializable_exif = {}
    for k, v in exif_data.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            serializable_exif[k] = v
        else:
            serializable_exif[k] = repr(v) 

    output_data = {
        "exif": serializable_exif,
        "face_detection": {
            "count": len(boxes),
            "boxes": boxes  
        }
    }
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Saved metadata to: {json_path}")
    except Exception as e:
        print(f"Failed to write JSON metadata to {json_path}: {e}", file=sys.stderr)

def main(path):
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    print(f"Checking file: {path}")

    sig_ok = is_jpeg_signature(path)
    print(f"JPEG signature valid: {sig_ok}")

    pillow_ok = validate_jpeg_with_pillow(path)
    print(f"Pillow validation (not corrupted & format==JPEG): {pillow_ok}")

    if not (sig_ok and pillow_ok):
        print("File does not appear to be a valid JPEG or is corrupted.", file=sys.stderr)
        sys.exit(3)

    exif = get_exif(path)
    pretty_print_exif(exif)

    pil_img = Image.open(path)
    pil_img_corrected = apply_exif_orientation(pil_img, exif)
    cv_img = pil_to_cv2(pil_img_corrected)

    try:
        out_img, boxes = detect_faces_yunet(cv_img)
    except Exception as e:
        print("Error during face detection:", e, file=sys.stderr)
        sys.exit(4)

    base, ext = os.path.splitext(path)
    out_path = f"{base}_faces.jpg"
    if not cv2.imwrite(out_path, out_img):
        print(f"Failed to write output image to {out_path}", file=sys.stderr)
        sys.exit(5)
    
    json_path = f"{base}_meta.json"
    save_metadata_as_json(exif, boxes, json_path)

    print(f"Saved output with detections to: {out_path}")
    if boxes:
        print(f"Detected {len(boxes)} face(s):")
        for b in boxes:
            print(f"  {b}")
    else:
        print("No faces detected.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_detection.py <image.jpeg>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
