"""Pydantic request models for the face swap API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FaceSwapRequest(BaseModel):
    """Request payload for face swap endpoints.

    All optional parameters have the same defaults as the runpod INPUT_SCHEMA
    for backward compatibility.
    """

    source_image: str = Field(
        ...,
        description="Base64-encoded source image(s), semicolon-separated for multiple",
    )
    target_image: str = Field(
        ...,
        description="Base64-encoded target image",
    )
    source_indexes: str = Field(
        "-1",
        description="Comma-separated zero-based source face indexes (-1 = auto)",
    )
    target_indexes: str = Field(
        "-1",
        description="Comma-separated zero-based target face indexes (-1 = auto)",
    )
    background_enhance: bool = Field(
        True,
        description="Enable RealESRGAN background upscaling",
    )
    face_restore: bool = Field(
        True,
        description="Enable CodeFormer face restoration",
    )
    face_upsample: bool = Field(
        True,
        description="Enable face-level upsampling",
    )
    upscale: int = Field(
        1,
        ge=1,
        le=4,
        description="Upscale factor (clamped 1-4 automatically)",
    )
    codeformer_fidelity: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="CodeFormer fidelity (0.0=higher quality, 1.0=higher fidelity)",
    )
    output_format: Literal["JPEG", "PNG"] = Field(
        "JPEG",
        description="Output image format",
    )
    min_face_size: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Minimum source face size as percentage of image dimension",
    )
    face_swapper_model: str = Field(
        "inswapper_128",
        description="Face swapper model name. See docs for full list of 13 models.",
    )
    face_swapper_resolution: Optional[str] = Field(
        None,
        description="Resolution string (e.g. '512x512'). Auto-detected if omitted.",
    )
    face_swapper_weight: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Identity blend weight (0.0-1.0)",
    )
    face_mask_blur: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Face mask edge softness",
    )
    face_mask_padding: str = Field(
        "0,0,0,0",
        description="Mask inset as 'top,right,bottom,left' percentages",
    )
    face_selector_mode: Literal["many", "one"] = Field(
        "many",
        description="'many' swaps all matching faces, 'one' swaps the best match",
    )
    face_selector_order: Literal[
        "left-right",
        "right-left",
        "top-bottom",
        "small-large",
        "large-small",
        "best-worst",
        "worst-best",
    ] = Field(
        "left-right",
        description="Sort order for target faces",
    )
    face_selector_gender: Optional[Literal["male", "female"]] = Field(
        None,
        description="Filter target faces by perceived gender",
    )
    face_selector_age_start: Optional[int] = Field(
        None,
        ge=0,
        le=120,
        description="Minimum target face age (inclusive)",
    )
    face_selector_age_end: Optional[int] = Field(
        None,
        ge=0,
        le=120,
        description="Maximum target face age (inclusive)",
    )
