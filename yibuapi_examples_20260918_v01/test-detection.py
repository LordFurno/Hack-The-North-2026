#!/usr/bin/env python3
"""
Object detection test script using Qwen 3.5 Omni via YibuAPI.
Detects up to 5 objects, outputs JSON schema, and draws bounding boxes on the image.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
import httpx
from PIL import Image, ImageDraw, ImageFont

load_dotenv()
DEFAULT_BASE_URL = "https://yibuapi.com/v1"

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
# 2. HTTP Helper & Message Building
# ---------------------------------------------------------------------------
def require_api_key() -> str:
    key = os.getenv("YIBU_API_KEY")
    if not key:
        raise ValueError("Environment variable YIBU_API_KEY is not set.")
    return key


def _data_url(path: Path, fallback_mime: str) -> str:
    mime = mimetypes.guess_type(path.name)[0] or fallback_mime
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_omni_messages(prompt: str, image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(image_path, "image/jpeg")}
                }
            ]
        }
    ]


def extract_json(response_json: Mapping[str, Any]) -> dict[str, Any] | None:
    choices = response_json.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    text = message.get("content")
    if text and isinstance(text, str):
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    return None


def chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None = None,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    with httpx.Client(timeout=120.0, trust_env=False) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        res_data = response.json()
        parsed = extract_json(res_data)
        if not parsed:
            raise ValueError(f"Failed to parse JSON response: {res_data}")
        return parsed, res_data


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
        "bounding box normalized to a 0-1000 scale as [ymin, xmin, ymax, xmax]."
    )

    messages = build_omni_messages(prompt, args.image)

    print("Sending detection request to YibuAPI...")
    parsed_json, _ = chat_completion(
        api_key=require_api_key(),
        model=args.model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "object_detection_max5",
                "strict": True,
                "schema": DETECTION_SCHEMA
            }
        }
    )

    objects = parsed_json.get("objects", [])
    print(f"\nDetected {len(objects)} object(s):")
    print(json.dumps(parsed_json, indent=2, ensure_ascii=False))

    # Render bounding boxes onto image
    draw_bounding_boxes(args.image, objects, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
