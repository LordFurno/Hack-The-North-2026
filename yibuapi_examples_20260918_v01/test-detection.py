#!/usr/bin/env python3
"""
Object detection test script using Qwen 3.5 Omni via YibuAPI.
Detects up to 5 objects, outputs JSON schema, and draws bounding boxes on the image.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from yibu_http import build_omni_messages, chat_completion, require_api_key

load_dotenv(Path(__file__).resolve().with_name(".env"))

# ---------------------------------------------------------------------------
# 1. JSON Schema Definition (Enforces max 5 objects)
# ---------------------------------------------------------------------------
DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "maxItems": 5,  # Restricts detection to a maximum of 5 objects
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Short name or label of detected object"
                    },
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Normalized bounding box [ymin, xmin, ymax, xmax] from 0 to 1000"
                    }
                },
                "required": ["label", "box_2d"],
                "additionalProperties": False
            }
        }
    },
    "required": ["objects"],
    "additionalProperties": False
}


# ---------------------------------------------------------------------------
# 2. Response Parsing
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict[str, Any]:
    """Parse model text that should contain one JSON object."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first:last + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise ValueError(f"Failed to parse JSON response text: {text}")


# ---------------------------------------------------------------------------
# 3. Drawing Helper Function
# ---------------------------------------------------------------------------
def draw_bounding_boxes(
    image_path: Path,
    objects: list[dict[str, Any]],
    output_path: Path
) -> None:
    """
    Converts 0-1000 scale normalized coordinates to image pixels,
    draws bounding boxes and labels onto the image, and saves it.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    # Use default PIL font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    colors = ["#FF3838", "#2C3E50", "#10B981", "#F59E0B", "#8B5CF6"]

    for idx, obj in enumerate(objects):
        ymin, xmin, ymax, xmax = obj["box_2d"]
        label = obj.get("label", "object")

        # Convert 0-1000 range to pixel coordinates
        px_xmin = int((xmin / 1000.0) * width)
        px_ymin = int((ymin / 1000.0) * height)
        px_xmax = int((xmax / 1000.0) * width)
        px_ymax = int((ymax / 1000.0) * height)

        color = colors[idx % len(colors)]

        # Draw bounding box rectangle
        draw.rectangle(
            [px_xmin, px_ymin, px_xmax, px_ymax],
            outline=color,
            width=3
        )

        # Draw label background box
        text_str = f" {label} "
        if font and hasattr(font, "getbbox"):
            bbox = font.getbbox(text_str)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = len(text_str) * 6, 12

        text_bg_ymin = max(0, px_ymin - text_h - 4)
        draw.rectangle(
            [px_xmin, text_bg_ymin, px_xmin + text_w + 4, text_bg_ymin + text_h + 4],
            fill=color
        )

        # Draw text label
        draw.text(
            (px_xmin + 2, text_bg_ymin + 2),
            text_str,
            fill="#FFFFFF",
            font=font
        )

    image.save(output_path)
    print(f"Annotated image saved to: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# 4. CLI Execution
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Detect up to 5 objects in an image using Qwen 3.5 Omni.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image file")
    parser.add_argument("--output", type=Path, default=Path("detected_output.jpg"), help="Path to save annotated image")
    parser.add_argument("--model", default="qwen3.5-omni-plus", help="Model name on YibuAPI")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image path {args.image} does not exist.")
        return 1

    prompt = (
        "Identify and locate the most prominent objects in this image. "
        "Detect a MAXIMUM of 5 objects. For each object, return its label and "
        "bounding box normalized to a 0-1000 scale as [ymin, xmin, ymax, xmax]. "
        "Return only valid JSON matching this schema: "
        f"{json.dumps(DETECTION_SCHEMA, separators=(',', ':'))}"
    )

    messages = build_omni_messages(prompt, image=args.image)

    print("Sending detection request to YibuAPI...")
    text, _response_json, _record = chat_completion(
        api_key=require_api_key(),
        model=args.model,
        messages=messages,
        purpose="object_detection_max5",
        max_tokens=512,
        temperature=0.1,
    )
    parsed_json = extract_json(text)

    objects = parsed_json.get("objects", [])
    print(f"\nDetected {len(objects)} object(s):")
    print(json.dumps(parsed_json, indent=2, ensure_ascii=False))

    # Render bounding boxes onto image
    draw_bounding_boxes(args.image, objects, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
