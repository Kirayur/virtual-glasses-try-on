"""
diffusion_refine.py — фотореалистичная доработка результата geometric tryon
(glasses_tryon.py) диффузионной моделью FLUX.

Пайплайн:
  1. glasses_tryon.py → result.jpg (точная геометрия)
  2. diffusion_refine.py → result_refined.jpg (тени, блики, преломления,
     сглаживание границы оправы)

Режимы маски (--mode):
  • full_face   — перерисовать всё лицо (макс. согласованность света, но риск
                  поплыть чертами)
  • glasses     — область очков + внутри линз (рекомендуется)
  • frame_only  — только контур оправы, минимальное вмешательство
  • temples_only — заменить дужки каталожного фото на маленький аккуратный штрих

Зависимости:
    pip install -U torch diffusers transformers accelerate sentencepiece protobuf pillow
    huggingface-cli login   # доступ к FLUX.2-dev на HF
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch


def _try_import_face_detector():
    # MediaPipe нужен только для масок full_face/glasses; на headless-серверах
    # без libGLES он не грузится, поэтому импорт ленивый и опциональный
    try:
        from glasses_tryon import FaceLandmarkDetector, LMK  # noqa: F401
        return FaceLandmarkDetector
    except Exception as e:
        print(f"[Diffusion] MediaPipe недоступен ({type(e).__name__}: {e}), маска full_face будет без landmark'ов")
        return None


def get_device_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def diff_mask(original_bgr: np.ndarray, composed_bgr: np.ndarray, thr: int = 10) -> np.ndarray:
    """Где оригинал ≠ composed — там оправа."""
    diff = cv2.absdiff(original_bgr, composed_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


# точки внешнего овала лица MediaPipe FaceMesh
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


def mask_full_face(all_lms: np.ndarray, shape: Tuple[int, int], feather_px: int = 25) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([all_lms[i] for i in FACE_OVAL_IDX if i < len(all_lms)], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)


def mask_glasses_area(original_bgr: np.ndarray, composed_bgr: np.ndarray, named: Dict[str, np.ndarray],
                      dilate_px: int = 18, feather_px: int = 15) -> np.ndarray:
    """Контур оправы (diff) + эллипсы по линзам — закрашиваем стёкла, чтобы FLUX
    перерисовал кожу под ними с бликом и лёгким преломлением."""
    h, w = original_bgr.shape[:2]
    base = diff_mask(original_bgr, composed_bgr)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    base = cv2.dilate(base, k, iterations=1)

    fill = np.zeros((h, w), dtype=np.uint8)
    for eye_outer, eye_inner, iris in [
        ('left_eye_outer', 'left_eye_inner', 'left_iris'),
        ('right_eye_outer', 'right_eye_inner', 'right_iris'),
    ]:
        if iris not in named:
            continue
        eye_w = float(np.linalg.norm(named[eye_outer] - named[eye_inner]))
        rx, ry = int(eye_w * 1.6), int(eye_w * 1.3)  # линза ≈ 1.6× ширины глаза
        cx, cy = int(named[iris][0]), int(named[iris][1])
        cv2.ellipse(fill, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    mask = cv2.bitwise_or(base, fill)
    return cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)


def mask_full_face_fallback(original_bgr: np.ndarray, composed_bgr: np.ndarray, feather_px: int = 25) -> np.ndarray:
    """Без лендмарков: эллипс вокруг bbox оправы, растянутый под примерный овал лица."""
    h, w = original_bgr.shape[:2]
    base = diff_mask(original_bgr, composed_bgr)
    ys, xs = np.where(base > 0)
    if len(xs) == 0:
        cx, cy, rx, ry = w // 2, h // 2, w // 3, h // 3
    else:
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        gw, gh = x2 - x1, max(y2 - y1, 1)
        rx = int(gw * 1.6)
        ry = int(max(gw * 1.3, gh * 4))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)


def mask_glasses_area_fallback(original_bgr: np.ndarray, composed_bgr: np.ndarray, dilate_px: int = 18,
                               feather_px: int = 15, lens_pad_ratio: float = 0.55) -> np.ndarray:
    """То же самое без лендмарков: ищем две связные компоненты (линзы) в diff'е
    и заливаем их выпуклую оболочку + общий bbox."""
    h, w = original_bgr.shape[:2]
    base = diff_mask(original_bgr, composed_bgr)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    dilated = cv2.dilate(base, k, iterations=1)

    long_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(31, w // 20), max(11, h // 80)))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, long_k)

    ys, xs = np.where(base > 0)
    if len(xs) > 0:
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        pad_y = int(max(y2 - y1, 1) * lens_pad_ratio)
        y1p, y2p = max(0, y1 - pad_y), min(h, y2 + pad_y)
        bbox_fill = np.zeros_like(closed)
        cv2.rectangle(bbox_fill, (x1, y1p), (x2, y2p), 255, -1)

        contours, _ = cv2.findContours(cv2.bitwise_or(closed, bbox_fill), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(closed)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            cv2.fillConvexPoly(result, hull, 255)
        closed = cv2.bitwise_or(closed, result)

    return cv2.GaussianBlur(closed, (feather_px * 2 + 1, feather_px * 2 + 1), 0)


def mask_frame_only(original_bgr: np.ndarray, composed_bgr: np.ndarray,
                    thickness_px: int = 6, feather_px: int = 5) -> np.ndarray:
    """Узкая полоса вдоль контура оправы — только сгладить край, без перерисовки."""
    base = diff_mask(original_bgr, composed_bgr, thr=8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness_px * 2 + 1, thickness_px * 2 + 1))
    mask = cv2.dilate(base, k, iterations=1)
    return cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)


def mask_temples_only(original_bgr: np.ndarray, composed_bgr: np.ndarray, lens_width_ratio: float = 0.60,
                      new_temple_extent_ratio: float = 0.10, feather_px: int = 8) -> np.ndarray:
    """
    Убирает торчащие дужки каталожного фото, оставляя саму рамку нетронутой.

    diff(original, composed) даёт всё, что налеплено на лицо: рамку + дужки.
    Центральная зона bbox'а (lens_width_ratio от ширины) — это сама рамка,
    её защищаем; всё, что снаружи — плохие дужки, отдаём в маску на закраску.
    Плюс небольшая полоска за пределами bbox — там FLUX дорисует маленький
    правильный штрих дужки.
    """
    h, w = original_bgr.shape[:2]
    base = diff_mask(original_bgr, composed_bgr, thr=8)
    ys, xs = np.where(base > 0)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    fw = max(x2 - x1, 1)
    cx = (x1 + x2) // 2

    frame_half_w = int(fw * lens_width_ratio / 2.0)
    frame_x1, frame_x2 = cx - frame_half_w, cx + frame_half_w

    new_ext = int(fw * new_temple_extent_ratio)
    roi_x1, roi_x2 = max(0, x1 - new_ext), min(w, x2 + new_ext)
    roi_y1, roi_y2 = max(0, y1), min(h, y2)

    mask = np.zeros((h, w), dtype=np.uint8)
    if frame_x1 > roi_x1:
        cv2.rectangle(mask, (roi_x1, roi_y1), (frame_x1, roi_y2), 255, -1)
    if roi_x2 > frame_x2:
        cv2.rectangle(mask, (frame_x2, roi_y1), (roi_x2, roi_y2), 255, -1)

    # добавляем фактические пиксели diff'а за пределами защищённой зоны —
    # дужки могут торчать чуть выше/ниже bbox'а
    diff_outside = base.copy()
    cv2.rectangle(diff_outside, (frame_x1, 0), (frame_x2, h), 0, -1)
    diff_outside = cv2.dilate(diff_outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    mask = cv2.bitwise_or(mask, diff_outside)

    cv2.imwrite("debug_temples_mask_binary.png", mask)
    print(f"[temples_mask] bbox diff: x=[{x1},{x2}] y=[{y1},{y2}]  fw={fw}")
    print(f"[temples_mask] protected center: x=[{frame_x1},{frame_x2}]")
    print(f"[temples_mask] mask coverage: {int(np.sum(mask > 0))} px white")

    return cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)


def estimate_frame_color(original_bgr: np.ndarray, composed_bgr: np.ndarray,
                         lens_width_ratio: float = 0.78) -> Tuple[Tuple[int, int, int], str]:
    """Медианный цвет рамки (по diff-пикселям в защищённой центральной зоне) +
    словесное описание по HSV — нужно для промпта temples_only."""
    base = diff_mask(original_bgr, composed_bgr, thr=12)
    ys, xs = np.where(base > 0)
    if len(xs) == 0:
        return (50, 50, 50), "dark"

    x1, x2 = int(xs.min()), int(xs.max())
    fw = max(x2 - x1, 1)
    cx = (x1 + x2) // 2
    frame_half_w = int(fw * lens_width_ratio / 2.0)
    frame_x1, frame_x2 = cx - frame_half_w, cx + frame_half_w

    central = np.zeros_like(base)
    central[:, frame_x1:frame_x2] = base[:, frame_x1:frame_x2]
    yy, xx = np.where(central > 0)
    if len(xx) == 0:
        return (50, 50, 50), "dark"

    pixels = composed_bgr[yy, xx]
    b, g, r = np.median(pixels, axis=0).astype(int).tolist()  # медиана устойчивее к бликам

    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if V < 50:
        descr = "black"
    elif S < 35:
        descr = "silver" if V > 200 else "light grey" if V > 150 else "dark grey"
    elif H < 10 or H >= 170:
        descr = "dark red" if V < 130 else "red"
    elif H < 22:
        descr = "dark brown tortoiseshell" if V < 130 else "brown"
    elif H < 35:
        descr = "gold" if V > 130 else "dark gold"
    elif H < 85:
        descr = "olive green" if V < 130 else "green"
    elif H < 130:
        descr = "navy blue" if V < 130 else "blue"
    else:
        descr = "purple"
    return (r, g, b), descr


# у каждого режима свой промпт — максимально конкретный под задачу маски
PROMPTS: Dict[str, Dict[str, str]] = {
    "full_face": {
        "prompt": (
            "Ultra-photorealistic portrait of the SAME person wearing the SAME "
            "eyeglasses already placed on their face. Preserve exact face identity, "
            "skin tone, hair, eye color, pose and expression. Render natural studio "
            "lighting consistent across the whole face. Glasses must have: thin soft "
            "contact shadow on the bridge of the nose and on the upper cheeks under "
            "the frame; subtle specular highlights on the frame edges; transparent "
            "clear lenses with very slight reflection at the top edge; skin visible "
            "behind the lenses with imperceptible refraction. 85mm portrait lens, "
            "sharp focus on eyes, professional retouching."
        ),
        "negative": (
            "different person, identity change, distorted face, asymmetric face, "
            "deformed eyes, crossed eyes, extra glasses, double frame, sunglasses, "
            "tinted lenses, opaque lenses, cartoon, illustration, painting, "
            "blurry, low quality, oversharpened, plastic skin, beauty filter, "
            "altered frame shape, missing frame, broken frame"
        ),
        "strength": 0.20,
    },
    "glasses": {
        "prompt": (
            "Photorealistic eyeglasses sitting on the face: keep the EXACT shape, "
            "color and position of the existing frame. Add only: (1) a soft thin "
            "contact shadow directly beneath the frame on the nose bridge and "
            "cheekbones, (2) subtle specular highlight along the top edge of the "
            "frame matching ambient light, (3) perfectly transparent clear glass "
            "lenses with very faint reflection in the upper corner, (4) the skin, "
            "eyebrows and eyes behind the lenses must remain clearly visible and "
            "anatomically correct with only a minimal refraction shift at the lens "
            "border. Studio portrait, natural skin texture, 85mm lens."
        ),
        "negative": (
            "changed frame shape, thicker frame, thinner frame, different color "
            "frame, tinted glass, dark lenses, sunglasses, opaque lenses, mirrored "
            "lenses, extra glasses, double rim, missing eyes, closed eyes, "
            "distorted eyes behind glass, deformed eyebrows, painted look, cartoon, "
            "illustration, blurry, low quality"
        ),
        "strength": 0.22,
    },
    "frame_only": {
        "prompt": (
            "Smooth and refine ONLY the outline of the existing eyeglass frame: "
            "anti-aliased clean edge, subtle micro-shadow where the frame touches "
            "the skin, tiny specular highlight on the metal/plastic rim. Do NOT "
            "change frame shape, color, thickness or position. Skin tone and "
            "texture identical to surrounding area."
        ),
        "negative": (
            "changed frame shape, repositioned frame, different color, thicker "
            "frame, thinner frame, blurry edge, halo, ringing artifacts, painted, "
            "cartoon, illustration, distorted, deformed"
        ),
        "strength": 0.15,
    },
    "temples_only": {
        "prompt": (
            "Photorealistic portrait. Clean side temple area of natural facial "
            "skin and hair texture, matching the surrounding face perfectly. At "
            "the very outer corner of the existing eyeglass frame, add a TINY "
            "short stub of a temple arm (1 cm long, ≈2 mm thick) in the SAME "
            "color and material as the frame — just a small hint visible right "
            "next to the lens rim. The rest of the side area must be pure skin "
            "and hair, with NO temple, NO arm, NO bar across the face. Keep the "
            "lens frame UNCHANGED. Studio portrait, natural skin, 85mm lens."
        ),
        "negative": (
            "wrong color temple, mismatched temple color, different color from "
            "frame, long temple, full temple arm, temple bar across the face, "
            "temple on the cheek, temple over the ear, thick temple, fat temple, "
            "wide temple, oversized temple, curved temple, wavy temple, "
            "double temple, two temples on one side, sunglasses arm, "
            "second pair of glasses, double frame, extra frame, repositioned "
            "frame, altered frame shape, painted, cartoon, illustration, "
            "blurry, low quality, distorted face, changed face, changed hair, "
            "missing ear, deformed ear, halo, ghosting, color bleed"
        ),
        "strength": 1,
    },
}


def _is_flux2(model_id: str) -> bool:
    m = model_id.lower()
    return "flux.2" in m or "flux2" in m


def load_flux_pipeline(device: str, dtype: torch.dtype, model_id: str = "black-forest-labs/FLUX.2-dev",
                       low_vram: bool = False):
    """
    Грузит inpainting-пайплайн под конкретные веса:
      1. AutoPipelineForInpainting сам подбирает класс по model_index.json
      2. если не вышло — перебор известных классов (FLUX.2 / FLUX.1 Fill)
      3. .to(device) либо enable_model_cpu_offload(), но не оба сразу —
         на H200 (143 GB VRAM) offload не нужен и только замедляет
    """
    import diffusers
    from diffusers import AutoPipelineForInpainting

    print(f"[Diffusion] Загрузка {model_id} ...")
    pipe = None
    used_cls_name = None

    try:
        pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype)
        used_cls_name = type(pipe).__name__
        print(f"[Diffusion] AutoPipelineForInpainting → {used_cls_name}")
    except Exception as e:
        print(f"[Diffusion] AutoPipelineForInpainting не сработал ({type(e).__name__}: {e}), пробую явные классы")

    if pipe is None:
        if _is_flux2(model_id):
            candidates = ["Flux2InpaintPipeline", "Flux2DevInpaintPipeline", "Flux2KleinInpaintPipeline"]
        elif "fill" in model_id.lower():
            candidates = ["FluxFillPipeline", "FluxInpaintPipeline"]
        else:
            candidates = ["FluxInpaintPipeline", "FluxFillPipeline"]

        last_err = None
        for name in candidates:
            cls = getattr(diffusers, name, None)
            if cls is None:
                continue
            try:
                pipe = cls.from_pretrained(model_id, torch_dtype=dtype)
                used_cls_name = name
                print(f"[Diffusion] Явно загружен {name}")
                break
            except Exception as e:
                last_err = e
                print(f"[Diffusion]   {name} не подошёл ({type(e).__name__})")

        if pipe is None:
            raise RuntimeError(
                f"Не удалось загрузить inpainting-пайплайн для {model_id}. "
                f"Проверены: {candidates}. Последняя ошибка: {last_err}. "
                f"Обнови diffusers: pip install -U git+https://github.com/huggingface/diffusers"
            )

    if low_vram and device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("[Diffusion] enable_model_cpu_offload() — экономия VRAM")
        except Exception as e:
            print(f"[Diffusion] cpu_offload недоступен ({e}), .to({device})")
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    print(f"[Diffusion] Пайплайн готов: {used_cls_name} на {device}")
    return pipe


def _resize_for_flux(img: Image.Image, target: int = 1024) -> Image.Image:
    """FLUX требует размеры кратные 16; сохраняем aspect ratio, длинная сторона = target."""
    w, h = img.size
    if w >= h:
        new_w, new_h = target, int(round(target * h / w / 16) * 16)
    else:
        new_h, new_w = target, int(round(target * w / h / 16) * 16)
    return img.resize((max(new_w, 16), max(new_h, 16)), Image.LANCZOS)


def refine_with_flux(
    original_path: str,
    composed_path: str,
    out_path: str = "result_refined.jpg",
    mode: str = "glasses",
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
    face_model_path: str = "face_landmarker_v2_with_blendshapes.task",
    num_steps: int = 28,
    guidance_scale: float = 30.0,  # FLUX.1-Fill: ~30, FLUX.2: ~8
    strength: Optional[float] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    target_size: int = 1024,
    seed: Optional[int] = 42,
    low_vram: bool = False,
) -> str:
    if mode not in PROMPTS:
        raise ValueError(f"mode must be one of {list(PROMPTS)}; got {mode!r}")

    device, dtype = get_device_dtype()
    print(f"[Diffusion] device={device} dtype={dtype}  mode={mode}")

    original_bgr = cv2.imread(original_path)
    composed_bgr = cv2.imread(composed_path)
    if original_bgr is None:
        raise FileNotFoundError(original_path)
    if composed_bgr is None:
        raise FileNotFoundError(composed_path)
    h, w = original_bgr.shape[:2]
    composed_bgr = cv2.resize(composed_bgr, (w, h))

    # лендмарки нужны только для точных масок full_face/glasses; на серверах
    # без libGLES MediaPipe не грузится — тогда идём в fallback-маски
    FaceLandmarkDetector = _try_import_face_detector()
    all_lms, named = None, None
    if FaceLandmarkDetector is not None and mode in ("full_face", "glasses"):
        try:
            detector = FaceLandmarkDetector(face_model_path)
            all_lms, named = detector.detect(original_bgr)
        except Exception as e:
            print(f"[Diffusion] Не удалось получить landmark'ы ({e}), fallback")
            all_lms, named = None, None

    if mode == "full_face":
        mask_np = mask_full_face(all_lms, (h, w)) if all_lms is not None else mask_full_face_fallback(original_bgr, composed_bgr)
    elif mode == "glasses":
        mask_np = mask_glasses_area(original_bgr, composed_bgr, named) if named is not None else mask_glasses_area_fallback(original_bgr, composed_bgr)
    elif mode == "frame_only":
        mask_np = mask_frame_only(original_bgr, composed_bgr)
    elif mode == "temples_only":
        mask_np = mask_temples_only(original_bgr, composed_bgr)
    else:
        raise ValueError(f"unknown mode: {mode}")

    mask_dbg = out_path.replace(".jpg", f"_mask_{mode}.jpg")
    cv2.imwrite(mask_dbg, mask_np)
    print(f"[Diffusion] mask: {mask_dbg}  ({int(np.sum(mask_np > 0))} px)")

    composed_pil = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_np).convert("L")
    composed_in = _resize_for_flux(composed_pil, target_size)
    mask_in = mask_pil.resize(composed_in.size, Image.LANCZOS)

    cfg = PROMPTS[mode]
    p_prompt = prompt if prompt is not None else cfg["prompt"]
    p_neg = negative_prompt if negative_prompt is not None else cfg["negative"]

    if mode == "temples_only" and prompt is None:
        # подставляем в промпт реальный цвет рамки с этого кадра
        (r, g, b), descr = estimate_frame_color(original_bgr, composed_bgr)
        p_prompt += (
            f" The temple stub MUST be the exact same {descr} color as the "
            f"existing frame on this photo (approximately RGB {r},{g},{b}). "
            f"Do NOT use any other color."
        )
        print(f"[Diffusion] Frame color detected: RGB({r},{g},{b}) → '{descr}'")
    p_strength = strength if strength is not None else cfg["strength"]

    pipe = load_flux_pipeline(device, dtype, model_id=model_id, low_vram=low_vram)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    print(f"[Diffusion] prompt: {p_prompt[:120]}...")
    print(f"[Diffusion] steps={num_steps} guidance={guidance_scale} strength={p_strength}")

    base_kwargs = dict(
        prompt=p_prompt,
        image=composed_in,
        mask_image=mask_in,
        height=composed_in.size[1],
        width=composed_in.size[0],
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # разные классы FLUX принимают разный набор kwargs — сверяемся с сигнатурой,
    # а не бьёмся об TypeError
    import inspect
    try:
        accepted = set(inspect.signature(pipe.__call__).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    extra = {}
    if "strength" in accepted:
        extra["strength"] = p_strength
    if "negative_prompt" in accepted and p_neg:
        extra["negative_prompt"] = p_neg
    if "true_cfg_scale" in accepted:
        extra.setdefault("true_cfg_scale", 1.0)  # у части FLUX-пайплайнов только так активируется negative_prompt

    print(f"[Diffusion] pipe.__call__ принимает: strength={'strength' in accepted}, negative_prompt={'negative_prompt' in accepted}")

    try:
        result_pil = pipe(**base_kwargs, **extra).images[0]
    except TypeError as e:
        print(f"[Diffusion] TypeError ({e}), пробую без extra")
        result_pil = pipe(**base_kwargs).images[0]

    result_pil = result_pil.resize((w, h), Image.LANCZOS)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    mask_soft = cv2.GaussianBlur(mask_np, (31, 31), 0).astype(np.float32)[..., None] / 255.0
    final = composed_bgr.astype(np.float32) * (1.0 - mask_soft) + result_bgr.astype(np.float32) * mask_soft
    final = np.clip(final, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, final)
    print(f"Refined ({mode}): {out_path}")

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FLUX диффузионная доработка примерки очков")
    ap.add_argument("--original", required=True, help="Оригинальное фото лица (без очков)")
    ap.add_argument("--composed", required=True, help="Результат glasses_tryon.py (с очками)")
    ap.add_argument("--out", default="result_refined.jpg")
    ap.add_argument("--mode", choices=list(PROMPTS), default="glasses", help="Тип маски inpainting")
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-Fill-dev",
                    help="По умолчанию FLUX.1-Fill-dev (стабильный inpaint); "
                         "FLUX.2-dev пока несовместим с inpaint-пайплайнами diffusers (Mistral3 vs Qwen3 mismatch)")
    ap.add_argument("--face-model", default="face_landmarker_v2_with_blendshapes.task")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=30.0, help="FLUX.1-Fill: ~30.0; FLUX.2: ~8.0")
    ap.add_argument("--strength", type=float, default=None, help="если не указан — берётся оптимальный для режима")
    ap.add_argument("--size", type=int, default=1024, help="длинная сторона при подаче в FLUX (кратно 16)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="прогнать все режимы подряд")
    ap.add_argument("--low-vram", action="store_true", help="enable_model_cpu_offload для GPU <32 GB (на H200 не нужно)")

    a = ap.parse_args()

    if a.all:
        base, ext = os.path.splitext(a.out)
        for m in PROMPTS:
            refine_with_flux(
                a.original, a.composed, out_path=f"{base}_{m}{ext}", mode=m,
                model_id=a.model_id, face_model_path=a.face_model, num_steps=a.steps,
                guidance_scale=a.guidance, strength=a.strength, target_size=a.size,
                seed=a.seed, low_vram=a.low_vram,
            )
    else:
        refine_with_flux(
            a.original, a.composed, a.out, mode=a.mode, model_id=a.model_id,
            face_model_path=a.face_model, num_steps=a.steps, guidance_scale=a.guidance,
            strength=a.strength, target_size=a.size, seed=a.seed, low_vram=a.low_vram,
        )
