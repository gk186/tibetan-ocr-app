import os
import cv2
from typing import List

from BDRC.Inference import OCRPipeline
from BDRC.Data import OpStatus, Encoding
from BDRC.Utils import (
    get_platform,
    read_ocr_model_config,
    read_line_model_config,
)


def load_pipeline(ocr_model_dir: str, line_model_dir: str = "Models/Lines") -> OCRPipeline:
    """Load OCR and line detection models and create an :class:`OCRPipeline`.

    Parameters
    ----------
    ocr_model_dir : str
        Directory containing the OCR model's ``model_config.json`` file.
    line_model_dir : str, optional
        Directory with ``config.json`` for line detection. Defaults to ``Models/Lines``.

    Returns
    -------
    OCRPipeline
        Ready-to-use pipeline instance.
    """
    ocr_config_path = os.path.join(ocr_model_dir, "model_config.json")
    line_config = read_line_model_config(line_model_dir)
    ocr_config = read_ocr_model_config(ocr_config_path)
    platform = get_platform()
    return OCRPipeline(platform, ocr_config, line_config)


def ocr_image(
    image_path: str,
    pipeline: OCRPipeline,
    *,
    merge_lines: bool = True,
    dewarp: bool = False,
    k_factor: float = 2.5,
    bbox_tolerance: float = 3.0,
    encoding: Encoding = Encoding.Unicode,
) -> List[str]:
    """Run OCR on a single image and return the recognised text lines.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    pipeline : OCRPipeline
        Pipeline instance created with :func:`load_pipeline`.
    merge_lines : bool, optional
        Whether to merge broken line segments before OCR.
    dewarp : bool, optional
        Apply TPS based dewarping.
    k_factor : float, optional
        Line extraction scaling factor.
    bbox_tolerance : float, optional
        Bounding box tolerance when cropping lines.
    encoding : Encoding, optional
        Desired output encoding.

    Returns
    -------
    list[str]
        Recognised text lines.
    """
    img = cv2.imread(image_path)
    status, result = pipeline.run_ocr(
        img,
        merge_lines=merge_lines,
        use_tps=dewarp,
        k_factor=k_factor,
        bbox_tolerance=bbox_tolerance,
        target_encoding=encoding,
    )
    if status != OpStatus.SUCCESS:
        raise RuntimeError(f"OCR failed: {result}")
    _, _, ocr_lines, _ = result
    return [line.text for line in ocr_lines]
