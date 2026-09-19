"""
glasses_tryon.py — примерка очков с реальными физическими размерами.

Задачи:
  0) Геодезика вместо прямой линии — теперь не просто валидация, а часть фитинга.
  1) Скейл очков к лицу через interpupillary distance (63 мм — эталон).
  2) Мост очков совмещается с переносицей (landmark MediaPipe).
  3) Поворот и якорные точки подбираются минимизацией ошибки (MSE + геодезика + угол).
  4) Перебор кандидатов на крепление заушника между глазом и ухом.

Запуск:
    python glasses_tryon.py --face face.jpg --glasses tm_99086.jpg --out result.jpg

Зависимости:
    pip install mediapipe opencv-python numpy scikit-fmm scipy pillow
"""
from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import skfmm
    HAS_SKFMM = True
except ImportError:
    HAS_SKFMM = False


@dataclass
class GlassesSpec:
    """Физические размеры очков в мм. По умолчанию — Vogue VO4094."""
    bridge_mm: float = 18.0
    temple_mm: float = 135.0
    lens_width_mm: float = 52.0
    lens_height_mm: float = 44.0
    lens_diameter_mm: float = 55.0
    frame_width_mm: float = 129.0

    @property
    def total_front_width_mm(self) -> float:
        # физическая ширина фронта = 2 линзы + мост
        return 2 * self.lens_width_mm + self.bridge_mm


# индексы MediaPipe FaceMesh (478 точек), которые нам реально нужны
LMK = {
    'left_temple': 234,
    'right_temple': 454,
    'nose_bridge_top': 168,
    'nose_bridge_mid': 6,
    'nose_tip': 1,
    'left_eye_outer': 33,
    'left_eye_inner': 133,
    'right_eye_inner': 362,
    'right_eye_outer': 263,
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
    'left_iris': 468,
    'right_iris': 473,
    'left_ear': 127,
    'right_ear': 356,
    'left_cheek_upper': 227,
    'right_cheek_upper': 447,
    'left_cheek_mid': 137,
    'right_cheek_mid': 366,
}

# симметричные кандидаты между внешним углом глаза и ухом — здесь ищем крепление заушника
EAR_CANDIDATES_LEFT = [127, 162, 21, 54, 103, 67, 234, 227, 137, 177, 215]
EAR_CANDIDATES_RIGHT = [356, 389, 251, 284, 332, 297, 454, 447, 366, 401, 435]


class FaceLandmarkDetector:
    def __init__(self, model_path: str = "face_landmarker_v2_with_blendshapes.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                "Скачать: wget -O face_landmarker_v2_with_blendshapes.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)

    def detect(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)
        if not res.face_landmarks:
            raise RuntimeError("Лицо не найдено")

        h, w = image_bgr.shape[:2]
        lms = res.face_landmarks[0]
        all_pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)

        named = {name: all_pts[idx] for name, idx in LMK.items() if idx < len(all_pts)}
        # виртуальная точка моста — середина между верхом и серединой переносицы
        if 'nose_bridge_top' in named and 'nose_bridge_mid' in named:
            named['nose_bridge_mid_top'] = (named['nose_bridge_top'] + named['nose_bridge_mid']) / 2.0
        return all_pts, named


class GeodesicFaceMetric:
    """
    Лицо — не плоскость, а рельеф: расстояние по прямой между висками короче,
    чем по факту вдоль щеки. Строим карту "скорости" по градиенту яркости
    (перепады тона ≈ перепады рельефа) и решаем эйконал (Fast Marching) —
    это даёт кратчайший путь ПО ПОВЕРХНОСТИ, а не напролом через воздух.
    """

    def __init__(self, image_bgr: np.ndarray, sigma: float = 2.0, intensity_weight: float = 0.4):
        if not HAS_SKFMM:
            raise ImportError("Требуется scikit-fmm: pip install scikit-fmm")
        self.image = image_bgr
        self.h, self.w = image_bgr.shape[:2]
        self.w_int = intensity_weight
        self.speed = self._build_speed_map(sigma)

    def _build_speed_map(self, sigma: float) -> np.ndarray:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray = cv2.GaussianBlur(gray, (0, 0), sigma) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # там, где яркость резко меняется — скорость ниже, путь "дороже"
        speed = 1.0 / np.sqrt(1.0 + self.w_int * (gx ** 2 + gy ** 2))
        return np.clip(speed, 1e-3, 1.0)

    def travel_time_from(self, point) -> np.ndarray:
        """Одно решение FMM из точки — даёт расстояние сразу до ВСЕХ пикселей,
        поэтому источник стоит переиспользовать, а не гонять FMM на каждую пару точек."""
        phi = np.ones((self.h, self.w))
        x, y = self._clip(point)
        phi[y, x] = -1
        return skfmm.travel_time(phi, self.speed, dx=1.0)

    def read(self, dist_field: np.ndarray, point) -> float:
        x, y = self._clip(point)
        return float(dist_field[y, x])

    def geodesic(self, p1, p2) -> float:
        """Разовый запрос расстояния между двумя точками (для одиночной проверки)."""
        try:
            field = self.travel_time_from(p1)
        except Exception:
            return self.euclidean(p1, p2)
        return self.read(field, p2)

    def euclidean(self, p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _clip(self, point) -> Tuple[int, int]:
        x, y = int(round(point[0])), int(round(point[1]))
        return np.clip(x, 0, self.w - 1), np.clip(y, 0, self.h - 1)


IPD_MM_REFERENCE = 63.0  # среднее межзрачковое расстояние взрослого


def validate_geodesic(named: Dict[str, np.ndarray],
                      metric: Optional[GeodesicFaceMetric],
                      px_per_mm: float,
                      verbose: bool = True) -> Dict[str, float]:
    """Сравнивает прямую и геодезику висок-висок — просто для лога, на фитинг не влияет."""
    lt, rt = named['left_temple'], named['right_temple']
    euc_px = float(np.linalg.norm(lt - rt))
    out = {'euc_temple_mm': euc_px / px_per_mm}

    if metric is not None:
        geo_px = metric.geodesic(lt, rt)
        out['geo_temple_mm'] = geo_px / px_per_mm
        out['curvature_ratio'] = geo_px / euc_px

    if verbose:
        print(f"[geo] висок-висок: прямая {out['euc_temple_mm']:.1f} мм", end="")
        if 'geo_temple_mm' in out:
            print(f", геодезика {out['geo_temple_mm']:.1f} мм (кривизна ×{out['curvature_ratio']:.2f})")
        else:
            print(" (scikit-fmm не установлен, геодезика недоступна)")
    return out


def compute_face_scale(named: Dict[str, np.ndarray], spec: GlassesSpec) -> Dict[str, float]:
    """px/mm через IPD (эта зона лица плоская, прямая и геодезика тут почти совпадают)."""
    ipd_px = float(np.linalg.norm(named['left_iris'] - named['right_iris']))
    px_per_mm = ipd_px / IPD_MM_REFERENCE
    return {
        'px_per_mm': px_per_mm,
        'glasses_width_px': spec.frame_width_mm * px_per_mm,
        'lens_width_px': spec.lens_width_mm * px_per_mm,
        'lens_height_px': spec.lens_height_mm * px_per_mm,
        'bridge_px': spec.bridge_mm * px_per_mm,
        'temple_px': spec.temple_mm * px_per_mm,
    }


def _is_studio_white_bg(bgr: np.ndarray, frac_thr: float = 0.35) -> bool:
    """Эвристика для каталожных фото: светлая рамка по периметру = белый фон."""
    h, w = bgr.shape[:2]
    border = np.concatenate([
        bgr[0, :, :].reshape(-1, 3), bgr[-1, :, :].reshape(-1, 3),
        bgr[:, 0, :].reshape(-1, 3), bgr[:, -1, :].reshape(-1, 3),
    ], axis=0)
    gray = cv2.cvtColor(border.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).ravel()
    return float(np.mean(gray > 235)) > (1.0 - frac_thr)


def _alpha_from_white_bg(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def _alpha_from_rembg(bgr: np.ndarray, model_name: str = "isnet-general-use") -> Optional[np.ndarray]:
    """Удаление фона нейросетью. isnet — хороший баланс точности и веса для тонких оправ."""
    try:
        from rembg import remove, new_session
    except ImportError:
        return None
    try:
        session = new_session(model_name)
    except Exception as e:
        print(f"[glasses] модель {model_name!r} недоступна ({e}), беру u2net")
        session = new_session("u2net")

    rgba = remove(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5,
    )
    arr = np.array(rgba) if not isinstance(rgba, np.ndarray) else rgba
    return arr[..., 3] if arr.ndim == 3 and arr.shape[2] == 4 else None


def _keep_central_glasses_blob(alpha: np.ndarray, min_blob_frac: float = 0.005,
                               verbose: bool = True) -> np.ndarray:
    """Оставляет только связные компоненты рядом с самой крупной (сама оправа) —
    отрезает случайный мусор по краям кадра (руки, листья, фон)."""
    h, w = alpha.shape[:2]
    _, bin_mask = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, 8)
    if num <= 1:
        return alpha

    min_area = int(h * w * min_blob_frac)
    sizes = sorted(
        ((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area),
        reverse=True,
    )
    if not sizes:
        return alpha

    _, biggest_idx = sizes[0]
    by = centroids[biggest_idx, 1]
    keep = np.zeros_like(alpha)
    kept = 0
    for _, i in sizes:
        if abs(centroids[i, 1] - by) < h * 0.35:
            keep[labels == i] = alpha[labels == i]
            kept += 1

    if verbose:
        print(f"[glasses] компонент: {num - 1} → оставлено {kept}")
    return keep


def _autocrop_and_fill_lenses(bgr: np.ndarray, alpha: np.ndarray, fill_lenses: bool = True,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Кроп по bbox + для безободковых очков дорисовывает линзы (rembg их обычно съедает,
    т.к. они прозрачные) полупрозрачной заливкой, чтобы силуэт остался цельным."""
    ys, xs = np.where(alpha > 30)
    if len(xs) == 0:
        return bgr, alpha

    pad = max(5, int(0.02 * max(bgr.shape[:2])))
    h, w = bgr.shape[:2]
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    bgr_c, alpha_c = bgr[y1:y2, x1:x2].copy(), alpha[y1:y2, x1:x2].copy()

    if fill_lenses:
        hc, wc = alpha_c.shape
        band = np.zeros_like(alpha_c)
        top, bot = int(hc * 0.30), int(hc * 0.85)
        band[top:bot] = (alpha_c[top:bot] > 30).astype(np.uint8) * 255
        # смыкаем компоненты в полосе линз длинным горизонтальным ядром
        k_long = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, wc // 6), 3))
        bridged = cv2.morphologyEx(band, cv2.MORPH_CLOSE, k_long)

        contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lens_silhouette = np.zeros_like(alpha_c)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            cv2.fillConvexPoly(lens_silhouette, hull, 255)
            lens_silhouette = cv2.erode(lens_silhouette, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        soft = (lens_silhouette > 0) & (alpha_c < 30)
        alpha_c[soft] = 60  # полупрозрачно — не силуэт, а лёгкая засветка под линзой
        if verbose:
            print(f"[glasses] дорисовано линз: {int(soft.sum())} px")

    return bgr_c, alpha_c


def _knockout_lens_interior(bgr: np.ndarray, alpha: np.ndarray, white_thr: int = 220,
                            sat_thr: int = 30, verbose: bool = True) -> np.ndarray:
    """Делает прозрачными почти-белые пиксели внутри оправы — так на каталожном фото
    выглядит стекло, и после rembg оно обычно остаётся сплошной белой заглушкой."""
    if alpha.max() == 0:
        return alpha

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    near_white = (hsv[..., 1] < sat_thr) & (hsv[..., 2] > white_thr)
    candidates = near_white & (alpha > 30)
    if not candidates.any():
        if verbose:
            print("[glasses] knockout: нечего убирать")
        return alpha

    cand_mask = candidates.astype(np.uint8) * 255
    cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cand_mask = cv2.erode(cand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # не задеть обод

    new_alpha = alpha.copy()
    new_alpha[cand_mask > 0] = 0
    if verbose:
        print(f"[glasses] knockout стекла: {int((cand_mask > 0).sum())} px")
    return new_alpha


def _alpha_from_grabcut(bgr: np.ndarray, iters: int = 5) -> np.ndarray:
    """Fallback без rembg: считаем, что объект в центре кадра, 5% по краям — фон."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mx, my = int(w * 0.05), int(h * 0.05)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return _alpha_from_white_bg(bgr)

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.GaussianBlur(alpha, (3, 3), 0)


def load_glasses_rgba(path: str, force_remove_bg: bool = False, rembg_model: str = "isnet-general-use",
                      fill_lenses: bool = True, cleanup: bool = True, knockout_lens: bool = False,
                      verbose: bool = True) -> np.ndarray:
    """Достаёт очки с альфа-каналом: готовый PNG → threshold по белому фону → rembg → GrabCut."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить очки: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4 and not force_remove_bg:
        if verbose:
            print("[glasses] альфа уже есть в PNG")
        return img

    bgr = img[..., :3] if img.shape[2] == 4 else img
    if _is_studio_white_bg(bgr) and not force_remove_bg:
        if verbose:
            print("[glasses] белый фон каталога → threshold")
        alpha = _alpha_from_white_bg(bgr)
    else:
        if verbose:
            print(f"[glasses] обычное фото → rembg({rembg_model})")
        alpha = _alpha_from_rembg(bgr, model_name=rembg_model)
        if alpha is None:
            if verbose:
                print("[glasses] rembg не установлен, fallback на GrabCut")
            alpha = _alpha_from_grabcut(bgr)

    if cleanup:
        alpha = _keep_central_glasses_blob(alpha, verbose=verbose)
    if knockout_lens:
        alpha = _knockout_lens_interior(bgr, alpha, verbose=verbose)
    bgr, alpha = _autocrop_and_fill_lenses(bgr, alpha, fill_lenses=fill_lenses, verbose=verbose)
    return np.dstack([bgr, alpha])


def compute_glasses_anchors(scale: Dict[str, float], glasses_rgba: np.ndarray) -> Dict[str, np.ndarray]:
    """Якорные точки очков (мост, крепления и концы заушников) в их собственных пикселях."""
    gh, gw = glasses_rgba.shape[:2]
    k = scale['glasses_width_px'] / gw
    new_w, new_h = int(round(gw * k)), int(round(gh * k))

    cx, cy = new_w / 2.0, new_h * 0.50  # ось линз ≈ середина картинки по высоте
    half_front = scale['lens_width_px'] + scale['bridge_px'] / 2.0

    return {
        'resize_to': (new_w, new_h),
        'bridge_center': np.array([cx, cy], dtype=np.float32),
        'left_temple_attach': np.array([cx - half_front, cy], dtype=np.float32),
        'right_temple_attach': np.array([cx + half_front, cy], dtype=np.float32),
        'left_temple_tip': np.array([cx - half_front - scale['temple_px'], cy], dtype=np.float32),
        'right_temple_tip': np.array([cx + half_front + scale['temple_px'], cy], dtype=np.float32),
    }


def alpha_blend(face_bgr: np.ndarray, overlay_rgba: np.ndarray, top_left: Tuple[int, int]) -> np.ndarray:
    out = face_bgr.copy()
    x, y = top_left
    h, w = overlay_rgba.shape[:2]
    fh, fw = out.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, fw), min(y + h, fh)
    if x1 >= x2 or y1 >= y2:
        return out

    ov = overlay_rgba[y1 - y:y2 - y, x1 - x:x2 - x]
    a = ov[..., 3:4].astype(np.float32) / 255.0
    roi = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (roi * (1 - a) + ov[..., :3].astype(np.float32) * a).astype(np.uint8)
    return out


def _affine_transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return (np.hstack([pts, ones]) @ M.T).astype(np.float32)


BRIDGE_CANDIDATES = ['nose_bridge_mid_top']

# симметричные пары висков — от классического (234, 454) до точек ближе к уху,
# перебираем все и выбираем ту, что даёт наименьшую ошибку посадки
TEMPLE_PAIRS = [
    (234, 454), (227, 447), (137, 366), (177, 401), (215, 435),
    (116, 345), (143, 372), (162, 389), (130, 359), (226, 446),
]


def _build_rigid_M(angle_rad: float, src_center: np.ndarray, dst_center: np.ndarray) -> np.ndarray:
    """Поворот на angle_rad вокруг src_center + сдвиг в dst_center, масштаб = 1
    (масштаб уже зашит физическими мм на этапе compute_face_scale)."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = float(src_center[0]), float(src_center[1])
    tx = float(dst_center[0]) - (cos_a * cx - sin_a * cy)
    ty = float(dst_center[1]) - (sin_a * cx + cos_a * cy)
    return np.array([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=np.float64)


def fit_glasses_by_landmarks(
    face_named: Dict[str, np.ndarray],
    all_lms: np.ndarray,
    spec: GlassesSpec,
    scale: Dict[str, float],
    glasses_anchors: Dict[str, np.ndarray],
    metric: Optional[GeodesicFaceMetric] = None,
    geodesic_weight: float = 3.0,
    verbose: bool = True,
) -> Dict:
    """
    Перебирает симметричные пары висков и для каждой считает rigid-transform
    (поворот + сдвиг, масштаб фиксирован по физическим мм), затем штрафует по:
      - MSE проекции якорей очков относительно лендмарков лица,
      - несовпадению линз со зрачками,
      - развороту оправы относительно оси глаз,
      - (если есть geodesic-метрика) насколько физическая ширина фронта очков
        совпадает с расстоянием "по коже" между кандидатами — раньше это только
        печаталось для справки, теперь реально решает, какую пару взять.
    """
    best = {'score': float('inf')}
    g_bridge = glasses_anchors['bridge_center']
    g_l_att, g_r_att = glasses_anchors['left_temple_attach'], glasses_anchors['right_temple_attach']
    g_left_lens = (g_bridge + g_l_att) / 2.0
    g_right_lens = (g_bridge + g_r_att) / 2.0

    leo, reo = face_named['left_eye_outer'], face_named['right_eye_outer']
    li, ri = face_named['left_iris'], face_named['right_iris']

    for bridge_name in BRIDGE_CANDIDATES:
        if bridge_name not in face_named:
            continue
        nb = face_named[bridge_name]

        # одно решение FMM из моста — переиспользуем для геодезики ко всем кандидатам разом
        dist_field = metric.travel_time_from(nb) if metric is not None else None

        sym_candidates = []
        for li_idx, ri_idx in TEMPLE_PAIRS:
            if li_idx < len(all_lms) and ri_idx < len(all_lms):
                sym_candidates.append((all_lms[li_idx], all_lms[ri_idx], f"#{li_idx}", f"#{ri_idx}"))
        for i in range(len(TEMPLE_PAIRS) - 1):
            li1, ri1 = TEMPLE_PAIRS[i]
            li2, ri2 = TEMPLE_PAIRS[i + 1]
            if all(idx < len(all_lms) for idx in [li1, ri1, li2, ri2]):
                lp_mid = (all_lms[li1] + all_lms[li2]) / 2.0
                rp_mid = (all_lms[ri1] + all_lms[ri2]) / 2.0
                sym_candidates.append((lp_mid, rp_mid, f"mid({li1},{li2})", f"mid({ri1},{ri2})"))

        for lp, rp, l_label, r_label in sym_candidates:
            face_angle = math.atan2(*(rp - lp)[::-1])
            glasses_angle = math.atan2(*(g_r_att - g_l_att)[::-1])
            M = _build_rigid_M(face_angle - glasses_angle, g_bridge, nb)

            projected = _affine_transform(np.array([g_bridge, g_l_att, g_r_att], dtype=np.float32), M)
            point_mse = float(np.mean(np.sum((projected - np.array([nb, lp, rp])) ** 2, axis=1)))

            proj_lens = _affine_transform(np.array([g_left_lens, g_right_lens], dtype=np.float32), M)
            eye_err = float(np.linalg.norm(proj_lens[0] - li) + np.linalg.norm(proj_lens[1] - ri))

            eye_ang = math.degrees(math.atan2(*(reo - leo)[::-1]))
            gl_ang = math.degrees(math.atan2(*(projected[2] - projected[1])[::-1]))
            ang_err = abs(((eye_ang - gl_ang + 180) % 360) - 180)

            geo_err = 0.0
            geo_width_mm = None
            if dist_field is not None:
                # физическая ширина фронта очков должна совпадать с расстоянием
                # "по коже" от моста до каждого виска, а не с прямой (лицо не плоское)
                geo_width_mm = (metric.read(dist_field, lp) + metric.read(dist_field, rp)) / scale['px_per_mm']
                geo_err = (geo_width_mm - spec.total_front_width_mm) ** 2

            score = point_mse + eye_err * 2.0 + (ang_err ** 2) * 5.0 + geo_err * geodesic_weight
            if score < best['score']:
                best = {
                    'score': score,
                    'point_mse': point_mse,
                    'eye_alignment_err': eye_err,
                    'angle_err_deg': ang_err,
                    'geo_width_mm': geo_width_mm,
                    'bridge_landmark': bridge_name,
                    'left_ear_idx': l_label,
                    'right_ear_idx': r_label,
                    'M': M,
                }

    if verbose and 'M' in best:
        print(f"[fit] мост={best['bridge_landmark']}  виски={best['left_ear_idx']}/{best['right_ear_idx']}")
        print(f"[fit] MSE={best['point_mse']:.2f}  глаза={best['eye_alignment_err']:.2f}  "
              f"угол={best['angle_err_deg']:.2f}°"
              + (f"  геодезика-фронт={best['geo_width_mm']:.1f}мм (спека {spec.total_front_width_mm:.1f}мм)"
                 if best['geo_width_mm'] is not None else ""))
    return best


def render_glasses(face_bgr: np.ndarray, glasses_rgba: np.ndarray,
                   resize_to: Tuple[int, int], M: np.ndarray) -> np.ndarray:
    g = cv2.resize(glasses_rgba, resize_to, interpolation=cv2.INTER_AREA)
    fh, fw = face_bgr.shape[:2]
    warped = cv2.warpAffine(g, M, (fw, fh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255.0
    out = face_bgr.astype(np.float32) * (1 - a) + warped[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_debug(face_bgr: np.ndarray, named: Dict[str, np.ndarray], all_lms: np.ndarray,
               fit: Dict, glasses_anchors: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    vis = face_bgr.copy()
    for p in all_lms:
        cv2.circle(vis, (int(p[0]), int(p[1])), 1, (100, 100, 100), -1)

    bridge_pt = named[fit['bridge_landmark']]
    cv2.circle(vis, tuple(bridge_pt.astype(int)), 5, (0, 0, 255), -1)

    for eye_name in ['left_iris', 'right_iris']:
        cv2.circle(vis, tuple(named[eye_name].astype(int)), 5, (255, 0, 255), -1)

    if 'M' in fit and glasses_anchors is not None:
        for key in ['bridge_center', 'left_temple_attach', 'right_temple_attach']:
            pt = _affine_transform(np.array([glasses_anchors[key]]), fit['M'])[0]
            cv2.circle(vis, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

    cv2.putText(vis, f"bridge: {fit['bridge_landmark']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, f"L: {fit['left_ear_idx']}  R: {fit['right_ear_idx']}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def try_on_glasses(face_path: str, glasses_path: str, out_path: str = "result.jpg",
                   spec: GlassesSpec = GlassesSpec(),
                   model_path: str = "face_landmarker_v2_with_blendshapes.task",
                   debug_path: Optional[str] = "debug.jpg",
                   remove_bg: bool = False, fill_lenses: bool = True,
                   knockout_lens: bool = False, use_geodesic: bool = True) -> None:
    face_bgr = cv2.imread(face_path)
    if face_bgr is None:
        raise ValueError(f"Не загружено: {face_path}")

    detector = FaceLandmarkDetector(model_path)
    all_lms, named = detector.detect(face_bgr)

    # одна метрика на всё лицо: speed map считается один раз и переиспользуется
    # и для лога (validate_geodesic), и для реального фитинга
    metric = GeodesicFaceMetric(face_bgr) if (use_geodesic and HAS_SKFMM) else None

    scale = compute_face_scale(named, spec)
    validate_geodesic(named, metric, scale['px_per_mm'])
    print(f"[scale] px/mm={scale['px_per_mm']:.3f}  очки {spec.frame_width_mm}мм → {scale['glasses_width_px']:.0f}px")

    glasses_rgba = load_glasses_rgba(glasses_path, force_remove_bg=remove_bg,
                                     fill_lenses=fill_lenses, knockout_lens=knockout_lens)
    if debug_path:
        cv2.imwrite(debug_path.replace('.jpg', '_alpha.png'), glasses_rgba)

    anchors = compute_glasses_anchors(scale, glasses_rgba)
    fit = fit_glasses_by_landmarks(named, all_lms, spec, scale, anchors, metric=metric)

    result = render_glasses(face_bgr, glasses_rgba, anchors['resize_to'], fit['M'])
    cv2.imwrite(out_path, result)
    print(f"Сохранено: {out_path}")

    if debug_path:
        cv2.imwrite(debug_path, draw_debug(face_bgr, named, all_lms, fit, glasses_anchors=anchors))
        print(f"Debug: {debug_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Примерка очков с реальными физическими размерами")
    ap.add_argument("--face", required=True, help="фото лица")
    ap.add_argument("--glasses", required=True, help="фото очков")
    ap.add_argument("--out", default="result.jpg")
    ap.add_argument("--model", default="face_landmarker_v2_with_blendshapes.task")
    ap.add_argument("--debug", default="debug.jpg")
    ap.add_argument("--remove-bg", action="store_true", help="удалить фон у очков через rembg/GrabCut")
    ap.add_argument("--no-fill-lenses", action="store_true", help="не дорисовывать линзы (для оправ с полным ободом)")
    ap.add_argument("--knockout-lens", action="store_true", help="убрать белую заглушку стекла из каталожного фото")
    ap.add_argument("--no-geodesic", action="store_true", help="выключить геодезику (быстрее, но без учёта рельефа лица)")
    ap.add_argument("--bridge", type=float, default=18.0)
    ap.add_argument("--temple", type=float, default=135.0)
    ap.add_argument("--lens-width", type=float, default=52.0)
    ap.add_argument("--lens-height", type=float, default=44.0)
    ap.add_argument("--lens-diam", type=float, default=55.0)
    ap.add_argument("--frame-width", type=float, default=129.0)
    a = ap.parse_args()

    spec = GlassesSpec(bridge_mm=a.bridge, temple_mm=a.temple, lens_width_mm=a.lens_width,
                       lens_height_mm=a.lens_height, lens_diameter_mm=a.lens_diam, frame_width_mm=a.frame_width)
    try_on_glasses(a.face, a.glasses, a.out, spec=spec, model_path=a.model, debug_path=a.debug,
                   remove_bg=a.remove_bg, fill_lenses=not a.no_fill_lenses,
                   knockout_lens=a.knockout_lens, use_geodesic=not a.no_geodesic)
