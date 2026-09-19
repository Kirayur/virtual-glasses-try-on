"""
glasses_tryon.py — примерка очков с реальными физическими размерами.

Реализует 5 задач:
  0) Валидация геодезического расстояния (сопоставление с реальными мм).
  1) Скейл очков к лицу (через interpupillary distance / реальные мм).
  2) Наложение центра моста очков на переносицу (landmark MediaPipe).
  3) Подбор поворота / опорных точек, минимизирующих MSE между ключевыми
     точками очков (мост, виски, заушники) и landmark'ами лица.
  4) Поиск оптимальной точки в области ухо↔глаз для крепления заушника.

Использование:
    python glasses_tryon.py --face <face.jpg> --glasses <tm_99086.jpg> --out result.jpg

Зависимости:
    pip install mediapipe opencv-python numpy scikit-fmm scipy pillow
"""

from __future__ import annotations
import os
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# MediaPipe Tasks
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Опциональные импорты (для геодезики)
try:
    import skfmm
    HAS_SKFMM = True
except ImportError:
    HAS_SKFMM = False


#  Реальные параметры очков (тут используем модель Ray-Ban Erika RX7046, tm_99086.jpg, при желании можно брать любую другую модель)
@dataclass
class GlassesSpec:
    """Физические размеры очков в миллиметрах."""
    # Vogue VO4094 (по умолчанию)
    bridge_mm:       float = 18.0    # переносица (между линзами)
    temple_mm:       float = 135.0   # длина заушника
    lens_width_mm:   float = 52.0
    lens_height_mm:  float = 44.0
    lens_diameter_mm:float = 55.0
    frame_width_mm:  float = 129.0   # полная ширина оправы (висок↔висок)

    @property
    def total_front_width_mm(self) -> float:
        """Ширина фронта = 2 линзы + мост (≈ frame_width)."""
        return 2 * self.lens_width_mm + self.bridge_mm


#  Индексы MediaPipe FaceMesh (478 точек)
LMK = {
    # Виски (внешний контур щеки/виска)
    'left_temple':       234,
    'right_temple':      454,
    # Переносица
    'nose_bridge_top':   168,   # верхняя часть переносицы (между бровей)
    'nose_bridge_mid':   6,     # середина переносицы
    'nose_tip':          1,
    # Углы глаз
    'left_eye_outer':    33,
    'left_eye_inner':    133,
    'right_eye_inner':   362,
    'right_eye_outer':   263,
    # Верх/низ глаз
    'left_eye_top':      159,
    'left_eye_bottom':   145,
    'right_eye_top':     386,
    'right_eye_bottom':  374,
    # Зрачки
    'left_iris':         468,
    'right_iris':        473,
    # Уши — крайние точки овала лица на уровне глаз
    'left_ear':          127,   # выше виска, ближе к уху
    'right_ear':         356,
    # Кандидаты на крепление заушника (между глазом и ухом)
    'left_cheek_upper':  227,
    'right_cheek_upper': 447,
    'left_cheek_mid':    137,
    'right_cheek_mid':   366,
}

# Кандидаты MediaPipe-точек для поиска оптимального крепления заушника
# (между внешним углом глаза и ухом).
EAR_CANDIDATES_LEFT  = [127, 162, 21, 54, 103, 67, 234, 227, 137, 177, 215]
EAR_CANDIDATES_RIGHT = [356, 389, 251, 284, 332, 297, 454, 447, 366, 401, 435]


#  Детектор MediaPipe
class FaceLandmarkDetector:
    def __init__(self, model_path: str = "face_landmarker_v2_with_blendshapes.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                "Скачайте: wget -O face_landmarker_v2_with_blendshapes.task "
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
        """Возвращает (все 478 точек в пикселях, словарь именованных точек)."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)
        if not res.face_landmarks:
            raise RuntimeError("Лицо не найдено")

        h, w = image_bgr.shape[:2]
        lms = res.face_landmarks[0]
        all_pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)

        named = {}
        for name, idx in LMK.items():
            if idx < len(all_pts):
                named[name] = all_pts[idx]

        # Виртуальная точка: середина между nose_bridge_top (168) и nose_bridge_mid (6)
        if 'nose_bridge_top' in named and 'nose_bridge_mid' in named:
            named['nose_bridge_mid_top'] = (named['nose_bridge_top'] + named['nose_bridge_mid']) / 2.0

        return all_pts, named


#  Геодезическая метрика (упрощённая, на основе FMM по интенсивности)
class GeodesicFaceMetric:
    """
    Трактует изображение как 2.5D поверхность F(x,y) = (x, y, I(x,y)).
    speed = 1 / sqrt(1 + w·|∇I|²)
    """
    def __init__(self, image_bgr: np.ndarray, sigma: float = 2.0,
                 intensity_weight: float = 0.4):
        if not HAS_SKFMM:
            raise ImportError("Требуется scikit-fmm: pip install scikit-fmm")
        self.image = image_bgr
        self.h, self.w = image_bgr.shape[:2]
        self.sigma = sigma
        self.w_int = intensity_weight
        self.speed = self._build_speed_map()

    def _build_speed_map(self) -> np.ndarray:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray = cv2.GaussianBlur(gray, (0, 0), self.sigma) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        det_g = 1.0 + self.w_int * (gx**2 + gy**2)
        speed = 1.0 / np.sqrt(det_g)
        return np.clip(speed, 1e-3, 1.0)

    def euclidean(self, p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def geodesic(self, p1, p2) -> float:
        phi = np.ones((self.h, self.w))
        x, y = int(round(p1[0])), int(round(p1[1]))
        x = np.clip(x, 0, self.w - 1); y = np.clip(y, 0, self.h - 1)
        phi[y, x] = -1
        try:
            dist = skfmm.travel_time(phi, self.speed, dx=1.0)
        except Exception:
            return self.euclidean(p1, p2)
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        x2 = np.clip(x2, 0, self.w - 1); y2 = np.clip(y2, 0, self.h - 1)
        return float(dist[y2, x2])


# Валидация геодезического расстояния
# константа: среднее межзрачковое расстояние взрослого ≈ 63 мм.
IPD_MM_REFERENCE = 63.0

def validate_geodesic(face_bgr: np.ndarray,
                      named: Dict[str, np.ndarray],
                      verbose: bool = True) -> Dict[str, float]:
    """
    Сравнивает евклидово и геодезическое расстояния с реальной анатомией.
    Используем IPD (interpupillary distance) как эталон в мм.
    Возвращает: {px_per_mm, euc_temple_mm, geo_temple_mm, error_pct}
    """
    li = named['left_iris']; ri = named['right_iris']
    lt = named['left_temple']; rt = named['right_temple']

    ipd_px = float(np.linalg.norm(li - ri))
    px_per_mm = ipd_px / IPD_MM_REFERENCE

    euc_temple_px = float(np.linalg.norm(lt - rt))
    euc_temple_mm = euc_temple_px / px_per_mm

    out = {
        'px_per_mm': px_per_mm,
        'ipd_px': ipd_px,
        'euc_temple_mm': euc_temple_mm,
    }

    if HAS_SKFMM:
        metric = GeodesicFaceMetric(face_bgr, sigma=2.0, intensity_weight=0.4)
        geo_temple_px = metric.geodesic(lt, rt)
        geo_temple_mm = geo_temple_px / px_per_mm
        out['geo_temple_px'] = geo_temple_px
        out['geo_temple_mm'] = geo_temple_mm
        # Реальная ширина головы взрослого ≈ 145–155 мм.
        # frame_width оправы 125 мм должна быть < ширины головы.
        out['curvature_ratio'] = geo_temple_px / euc_temple_px

    if verbose:
        print("─── Валидация геодезики (Task 0) ───")
        print(f"  IPD (пикселей):        {ipd_px:.1f}")
        print(f"  px/mm (по IPD=63мм):   {px_per_mm:.3f}")
        print(f"  Евклид висок-висок:    {euc_temple_mm:.1f} мм  (норма ~140–155)")
        if 'geo_temple_mm' in out:
            print(f"  Геодезика висок↔висок: {out['geo_temple_mm']:.1f} мм")
            print(f"  Curvature ratio:       {out['curvature_ratio']:.3f}")
        print("────────────────────────────────────")
    return out


#  Скейл очков к лицу
def compute_face_scale(named: Dict[str, np.ndarray],
                       spec: GlassesSpec) -> Dict[str, float]:
    """
    Подсчитывает коэффициент масштабирования очков.
    Очки рендерятся в реальном размере (мм) → переводятся в пиксели через px/mm.
    """
    ipd_px = float(np.linalg.norm(named['left_iris'] - named['right_iris']))
    px_per_mm = ipd_px / IPD_MM_REFERENCE

    glasses_width_px = spec.frame_width_mm * px_per_mm
    lens_w_px        = spec.lens_width_mm  * px_per_mm
    lens_h_px        = spec.lens_height_mm * px_per_mm
    bridge_px        = spec.bridge_mm      * px_per_mm
    temple_px        = spec.temple_mm      * px_per_mm

    return {
        'px_per_mm': px_per_mm,
        'glasses_width_px': glasses_width_px,
        'lens_width_px':    lens_w_px,
        'lens_height_px':   lens_h_px,
        'bridge_px':        bridge_px,
        'temple_px':        temple_px,
    }


#  Загрузка PNG очков с альфой
def _is_studio_white_bg(bgr: np.ndarray, frac_thr: float = 0.35) -> bool:
    """Эвристика: каталожное фото на белом фоне = много светлых пикселей по периметру."""
    h, w = bgr.shape[:2]
    border = np.concatenate([
        bgr[0, :, :].reshape(-1, 3),
        bgr[-1, :, :].reshape(-1, 3),
        bgr[:, 0, :].reshape(-1, 3),
        bgr[:, -1, :].reshape(-1, 3),
    ], axis=0)
    gray = cv2.cvtColor(border.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).ravel()
    return float(np.mean(gray > 235)) > (1.0 - frac_thr)


def _alpha_from_white_bg(bgr: np.ndarray) -> np.ndarray:
    """Старый путь: thresh по белому фону + морфология."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    alpha = cv2.morphologyEx(
        alpha, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return alpha


def _alpha_from_rembg(bgr: np.ndarray,
                      model_name: str = "isnet-general-use") -> Optional[np.ndarray]:
    """
    Удаление фона нейросетью rembg.
    Лучшие модели для тонких объектов (очки):
        • 'birefnet-general'    — SOTA, самая точная (требует загрузки ~400 MB)
        • 'isnet-general-use'   — отличный компромисс, тонкие края (175 MB)
        • 'u2net'               — быстрая, но грубая для тонких объектов
    """
    try:
        from rembg import remove, new_session  # type: ignore
    except ImportError:
        return None
    try:
        session = new_session(model_name)
    except Exception as e:
        print(f"[glasses]   модель {model_name!r} недоступна ({e}); "
              f"откатываюсь к u2net")
        session = new_session("u2net")
    rgba = remove(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5,
    )
    if isinstance(rgba, np.ndarray) and rgba.shape[2] == 4:
        return rgba[..., 3]
    arr = np.array(rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3]
    return None


def _keep_central_glasses_blob(alpha: np.ndarray,
                               min_blob_frac: float = 0.005,
                               verbose: bool = True) -> np.ndarray:
    """
    Постобработка альфы:
      • Удаляет связные компоненты меньше min_blob_frac площади.
      • Группирует все оставшиеся компоненты, у которых центроид
        лежит в центральной полосе изображения (±35% по Y от центра).
        Это отрезает листья/руки/прочий мусор по краям.
    """
    h, w = alpha.shape[:2]
    _, bin_mask = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, 8)
    if num <= 1:
        return alpha

    keep = np.zeros_like(alpha)
    min_area = int(h * w * min_blob_frac)
    cx_img, cy_img = w / 2.0, h / 2.0

    sizes = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        sizes.append((area, i))

    if not sizes:
        return alpha

    # Самый большой объект — почти всегда фронт очков (мост + крепления)
    sizes.sort(reverse=True)
    biggest_area, biggest_idx = sizes[0]
    bx, by = centroids[biggest_idx]

    kept_ids = []
    for area, i in sizes:
        cy = centroids[i, 1]
        # Берём компоненты на ±35% высоты от Y самого большого объекта
        if abs(cy - by) < h * 0.35:
            kept_ids.append(i)
            keep[labels == i] = alpha[labels == i]

    if verbose:
        total = num - 1
        print(f"[glasses]   связных компонент: {total} → оставлено {len(kept_ids)} "
              f"(удалены: мелкие <{min_blob_frac*100:.1f}% и далёкие по Y)")
    return keep


def _autocrop_and_fill_lenses(bgr: np.ndarray, alpha: np.ndarray,
                              fill_lenses: bool = True,
                              verbose: bool = True
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Для безободковых очков rembg часто теряет линзы (они прозрачные).
    Эта функция:
      1) Кропает изображение по bbox альфы (убирает пустые поля)
      2) Если fill_lenses=True — заполняет промежутки между фронт-блоками
         (там где должны быть линзы) полупрозрачной альфой 60/255, чтобы
         силуэт был связным и физические размеры (мост, линзы) корректно
         накладывались. Сами цвета БГР остаются исходные (там просто
         белый фон, что на лице будет смотреться как лёгкая засветка
         — но без этого диффузия дорисует линзы).
    """
    ys, xs = np.where(alpha > 30)
    if len(xs) == 0:
        return bgr, alpha
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # Небольшой padding
    pad = max(5, int(0.02 * max(bgr.shape[:2])))
    h, w = bgr.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    bgr_c = bgr[y1:y2, x1:x2].copy()
    alpha_c = alpha[y1:y2, x1:x2].copy()

    if fill_lenses:
        # Закрашиваем «дырки» внутри фронта очков:
        # выпуклая оболочка центральной горизонтальной полосы -> лёгкая альфа.
        hc, wc = alpha_c.shape
        band_top = int(hc * 0.30)
        band_bot = int(hc * 0.85)
        band = np.zeros_like(alpha_c)
        band[band_top:band_bot] = (alpha_c[band_top:band_bot] > 30).astype(np.uint8) * 255

        # Соединяем компоненты в полосе морфологическим закрытием с длинным горизонтальным ядром
        k_long = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, wc // 6), 3))
        bridged = cv2.morphologyEx(band, cv2.MORPH_CLOSE, k_long)

        # Выпуклая оболочка даёт силуэт линз
        contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lens_silhouette = np.zeros_like(alpha_c)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            cv2.fillConvexPoly(lens_silhouette, hull, 255)
            # Чуть-чуть «продавим» внутрь чтобы не зайти за реальную оправу
            lens_silhouette = cv2.erode(
                lens_silhouette,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )

        # Лёгкая альфа (60/255) ТОЛЬКО там, где её ещё не было
        soft = (lens_silhouette > 0) & (alpha_c < 30)
        alpha_c[soft] = 60
        if verbose:
            n_soft = int(soft.sum())
            print(f"[glasses]   дорисовано линз: {n_soft} px (полупрозрачно, α=60)")

    return bgr_c, alpha_c


def _knockout_lens_interior(bgr: np.ndarray, alpha: np.ndarray,
                            white_thr: int = 220,
                            sat_thr: int = 30,
                            verbose: bool = True) -> np.ndarray:
    """
    Делает прозрачными пиксели внутри оправы, которые "почти белые/прозрачные"
    (стекло на каталожном фото обычно светло-серое или почти белое).

    Алгоритм:
      1) Берём bbox объекта (внутри alpha>30)
      2) Внутри bbox находим пиксели: low saturation & high value (≈ белые)
      3) Эрозия → не задеть тонкий обод оправы
      4) Сбрасываем их альфу в 0
    """
    h, w = bgr.shape[:2]
    if alpha.max() == 0:
        return alpha

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_chan = hsv[..., 1]  # saturation
    v_chan = hsv[..., 2]  # brightness

    # Почти-белые/прозрачные пиксели: низкая насыщенность + высокая яркость
    near_white = (s_chan < sat_thr) & (v_chan > white_thr)

    # Действуем только внутри объекта (где альфа > 0)
    inside_object = alpha > 30
    candidates = near_white & inside_object

    if not candidates.any():
        if verbose:
            print(f"[glasses]   knockout: подходящих пикселей не найдено")
        return alpha

    # Морфологическое сужение чтобы не задеть сам обод
    cand_mask = candidates.astype(np.uint8) * 255
    cand_mask = cv2.morphologyEx(
        cand_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # Применяем только к пикселям, окружённым непрозрачным объектом (заглубление)
    # — оставляем чуть отступа от края оправы
    cand_mask = cv2.erode(
        cand_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    new_alpha = alpha.copy()
    knockout = cand_mask > 0
    n_pix = int(knockout.sum())
    new_alpha[knockout] = 0

    if verbose:
        print(f"[glasses]   knockout стекла: {n_pix} px → прозрачно "
              f"(внутри оправы убраны почти-белые)")

    return new_alpha


def _alpha_from_grabcut(bgr: np.ndarray, iters: int = 5) -> np.ndarray:
    """
    Fallback без rembg: OpenCV GrabCut.
    Считаем, что объект (очки) — в центре, отступ 5% от краёв = фон.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return _alpha_from_white_bg(bgr)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Сглаживаем края
    alpha = cv2.morphologyEx(
        alpha, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha


def load_glasses_rgba(path: str, force_remove_bg: bool = False,
                      rembg_model: str = "isnet-general-use",
                      fill_lenses: bool = True,
                      cleanup: bool = True,
                      knockout_lens: bool = False,
                      verbose: bool = True) -> np.ndarray:
    """
    Загружает изображение очков с альфа-каналом.
      • Если PNG уже с alpha → берём как есть.
      • Если фон явно белый (каталог) → простой threshold.
      • Иначе → rembg (нейросеть) → GrabCut (fallback).

    Постобработка (нужна для безободковых очков и кривых снимков):
      • cleanup        — оставить только связные компоненты в центре кадра
                         (убирает листья, руки, кусочки фона).
      • fill_lenses    — дорисовать прозрачные линзы между креплениями
                         (полупрозрачно α=60, чтобы силуэт был связным).
      • автокроп до bbox объекта.

    rembg_model: 'isnet-general-use' (по умолч., лучше для тонких объектов),
                 'birefnet-general' (SOTA, ~400 MB), 'u2net' (быстро, грубо).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить очки: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Уже есть альфа?
    if img.shape[2] == 4 and not force_remove_bg:
        if verbose:
            print("[glasses] Используем встроенный альфа-канал PNG")
        return img

    bgr = img[..., :3] if img.shape[2] == 4 else img

    studio = _is_studio_white_bg(bgr)
    if studio and not force_remove_bg:
        if verbose:
            print("[glasses] Каталожное фото на белом фоне → threshold")
        alpha = _alpha_from_white_bg(bgr)
    else:
        if verbose:
            print(f"[glasses] Обычное фото → rembg({rembg_model}) ...")
        alpha = _alpha_from_rembg(bgr, model_name=rembg_model)
        if alpha is None:
            if verbose:
                print("[glasses]   rembg не установлен → fallback на GrabCut")
                print("[glasses]   pip install --user rembg onnxruntime")
            alpha = _alpha_from_grabcut(bgr)
        else:
            if verbose:
                print("[glasses]   rembg OK")

    # Постобработка
    if cleanup:
        alpha = _keep_central_glasses_blob(alpha, verbose=verbose)

    # Knockout: убираем почти-белые пиксели внутри оправы (стекло из каталога)
    if knockout_lens:
        alpha = _knockout_lens_interior(bgr, alpha, verbose=verbose)

    bgr, alpha = _autocrop_and_fill_lenses(bgr, alpha,
                                           fill_lenses=fill_lenses,
                                           verbose=verbose)

    return np.dstack([bgr, alpha])


#  ЗАДАЧА 2: Наложение центра моста на переносицу
def compute_glasses_anchors(spec: GlassesSpec, scale: Dict[str, float],
                            glasses_rgba: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Возвращает якорные точки очков в координатах ИХ собственного изображения (px):
      - bridge_center: центр моста (между линзами)
      - left_temple_attach / right_temple_attach: точки крепления заушников
      - left_temple_tip   / right_temple_tip:    концы заушников (для зацепа за ухо)
    Координаты — в пикселях изображения очков (после ресайза до scale).
    """
    gh, gw = glasses_rgba.shape[:2]
    target_w_px = scale['glasses_width_px']
    k = target_w_px / gw
    new_w = int(round(gw * k))
    new_h = int(round(gh * k))

    # Центр изображения очков — мост посередине, по вертикали ≈ верх линзы (~40% сверху)
    cx = new_w / 2.0
    cy = new_h * 0.50  # ось линз ≈ середина высоты изображения

    half_front = (scale['lens_width_px'] + scale['bridge_px'] / 2.0)
    return {
        'resize_to': (new_w, new_h),
        'bridge_center':         np.array([cx, cy], dtype=np.float32),
        'left_temple_attach':    np.array([cx - half_front, cy], dtype=np.float32),
        'right_temple_attach':   np.array([cx + half_front, cy], dtype=np.float32),
        # Концы заушников отстоят на temple_mm от точек крепления (вдоль X)
        'left_temple_tip':       np.array([cx - half_front - scale['temple_px'], cy],
                                          dtype=np.float32),
        'right_temple_tip':      np.array([cx + half_front + scale['temple_px'], cy],
                                          dtype=np.float32),
    }


def alpha_blend(face_bgr: np.ndarray, overlay_rgba: np.ndarray,
                top_left: Tuple[int, int]) -> np.ndarray:
    """Альфа-композитинг overlay на face в позиции top_left."""
    out = face_bgr.copy()
    x, y = top_left
    h, w = overlay_rgba.shape[:2]
    fh, fw = out.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, fw), min(y + h, fh)
    if x1 >= x2 or y1 >= y2:
        return out

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    ov = overlay_rgba[oy1:oy2, ox1:ox2]
    bgr = ov[..., :3].astype(np.float32)
    a   = (ov[..., 3:4].astype(np.float32)) / 255.0
    roi = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (roi * (1 - a) + bgr * a).astype(np.uint8)
    return out


#  ЗАДАЧЕ 3 + 4: MSE-оптимизация поворота и выбор оптимальных landmark'ов
def _affine_transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Применяет 2x3 матрицу аффинного преобразования к Nx2 точкам."""
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return (np.hstack([pts, ones]) @ M.T).astype(np.float32)


# Точка моста очков (Task 3)
# nose_bridge_mid_top — виртуальная точка, середина между 168 и 6
BRIDGE_CANDIDATES = ['nose_bridge_mid_top']
# Симметричные пары кандидатов для висков (Task 4: оптимальная точка ухо↔глаз).
# Каждая пара (left, right) — зеркально-симметричные точки MediaPipe FaceMesh.
TEMPLE_PAIRS = [
    (234, 454),   # виски (классические)
    (227, 447),   # верхняя скула
    (137, 366),   # средняя скула
    (177, 401),   # нижний висок
    (215, 435),   # боковой контур
    (116, 345),   # над скулой
    (143, 372),   # под глазом сбоку
    (162, 389),   # верхний висок
    (130, 359),   # внешний угол глаза (ниже)
    (226, 446),   # перед ухом
]


def fit_glasses_by_landmarks(
    face_named: Dict[str, np.ndarray],
    all_lms: np.ndarray,
    spec: GlassesSpec,
    scale: Dict[str, float],
    glasses_anchors: Dict[str, np.ndarray],
    verbose: bool = True,
) -> Dict:
    """
    ЗАДАЧИ 3 + 4.

    Стратегия:
      • Масштаб очков СТРОГО фиксирован из физических размеров (px/mm × 125 мм).
        Никакой подгонки под ширину лица — очки могут быть шире или уже.
      • Поворот определяется осью глаз (или осью висков).
      • Сдвиг — мост очков → переносица.
      • Перебираются симметричные пары висков для минимизации MSE.
      • Rigid transform: rotation + translation, scale = 1.
    """
    best = {'score': float('inf')}

    g_bridge = glasses_anchors['bridge_center']
    g_l_att  = glasses_anchors['left_temple_attach']
    g_r_att  = glasses_anchors['right_temple_attach']

    # Центры линз в координатах очков
    g_left_lens  = (g_bridge + g_l_att) / 2.0
    g_right_lens = (g_bridge + g_r_att) / 2.0

    leo = face_named['left_eye_outer']
    reo = face_named['right_eye_outer']
    li  = face_named['left_iris']
    ri  = face_named['right_iris']

    def _build_rigid_M(angle_rad: float, bridge_src: np.ndarray,
                       bridge_dst: np.ndarray) -> np.ndarray:
        """Строит матрицу 2×3: поворот на angle_rad вокруг bridge_src + сдвиг в bridge_dst.
        Масштаб = 1 (строго из физических размеров)."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Поворот вокруг bridge_src
        cx, cy = float(bridge_src[0]), float(bridge_src[1])
        tx = float(bridge_dst[0]) - (cos_a * cx - sin_a * cy)
        ty = float(bridge_dst[1]) - (sin_a * cx + cos_a * cy)
        M = np.array([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty],
        ], dtype=np.float64)
        return M

    for bridge_name in BRIDGE_CANDIDATES:
        if bridge_name not in face_named:
            continue
        nb = face_named[bridge_name]

        # Собираем все симметричные кандидаты: сами пары + середины между соседними
        sym_candidates = []
        for li_idx, ri_idx in TEMPLE_PAIRS:
            if li_idx < len(all_lms) and ri_idx < len(all_lms):
                sym_candidates.append((all_lms[li_idx], all_lms[ri_idx],
                                       f"#{li_idx}", f"#{ri_idx}"))
        # Добавляем середины между соседними парами (тоже симметричные)
        for i in range(len(TEMPLE_PAIRS) - 1):
            li1, ri1 = TEMPLE_PAIRS[i]
            li2, ri2 = TEMPLE_PAIRS[i + 1]
            if all(idx < len(all_lms) for idx in [li1, ri1, li2, ri2]):
                lp_mid = (all_lms[li1] + all_lms[li2]) / 2.0
                rp_mid = (all_lms[ri1] + all_lms[ri2]) / 2.0
                sym_candidates.append((lp_mid, rp_mid,
                                       f"mid({li1},{li2})", f"mid({ri1},{ri2})"))

        for lp, rp, l_label, r_label in sym_candidates:
            # Угол поворота = ось висков на лице
            temple_axis = rp - lp
            face_angle = math.atan2(temple_axis[1], temple_axis[0])

            # Ось очков в их координатах (горизонтальная → angle=0)
            glasses_axis = g_r_att - g_l_att
            glasses_angle = math.atan2(glasses_axis[1], glasses_axis[0])

            # Нужный поворот
            rotation = face_angle - glasses_angle

            # Rigid transform: поворот + сдвиг (мост → переносица)
            M = _build_rigid_M(rotation, g_bridge, nb)

            # Проецируем якорные точки очков
            src_pts = np.array([g_bridge, g_l_att, g_r_att], dtype=np.float32)
            projected = _affine_transform(src_pts, M)
            dst_pts = np.array([nb, lp, rp], dtype=np.float32)
            point_mse = float(np.mean(np.sum((projected - dst_pts) ** 2, axis=1)))

            # Где окажутся линзы → должны попадать в зрачки
            src_lens = np.array([g_left_lens, g_right_lens], dtype=np.float32)
            proj_lens = _affine_transform(src_lens, M)
            eye_err = float(np.linalg.norm(proj_lens[0] - li) +
                            np.linalg.norm(proj_lens[1] - ri))

            # Ось очков должна совпадать с осью глаз
            eye_axis = reo - leo
            eye_ang  = math.degrees(math.atan2(eye_axis[1], eye_axis[0]))
            gl_axis  = projected[2] - projected[1]
            gl_ang   = math.degrees(math.atan2(gl_axis[1], gl_axis[0]))
            ang_err  = abs(((eye_ang - gl_ang + 180) % 360) - 180)

            score = point_mse + eye_err * 2.0 + (ang_err ** 2) * 5.0

            if score < best['score']:
                best = {
                    'score':             score,
                    'point_mse':         point_mse,
                    'eye_alignment_err': eye_err,
                    'angle_err_deg':     ang_err,
                    'bridge_landmark':   bridge_name,
                    'left_ear_idx':      l_label,
                    'right_ear_idx':     r_label,
                    'M':                 M,
                }

    if verbose and 'M' in best:
        print("─── Оптимизация (Task 3 + 4) ───")
        print(f"  Лучший мост:        {best['bridge_landmark']}")
        print(f"  Левый висок-якорь:  landmark #{best['left_ear_idx']}")
        print(f"  Правый висок-якорь: landmark #{best['right_ear_idx']}")
        print(f"  Точечный MSE:       {best['point_mse']:.2f}")
        print(f"  Ошибка глаз↔линзы:  {best['eye_alignment_err']:.2f}")
        print(f"  Угловая ошибка:     {best['angle_err_deg']:.2f}°")
        print("────────────────────────────────")
    return best


#  Композиция: применяем аффинное преобразование к изображению очков
def render_glasses(face_bgr: np.ndarray,
                   glasses_rgba: np.ndarray,
                   resize_to: Tuple[int, int],
                   M: np.ndarray) -> np.ndarray:
    """Ресайзит очки до resize_to, потом warpAffine в систему координат лица."""
    g = cv2.resize(glasses_rgba, resize_to, interpolation=cv2.INTER_AREA)
    fh, fw = face_bgr.shape[:2]
    warped = cv2.warpAffine(g, M, (fw, fh),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    # Альфа-композитинг
    bgr = warped[..., :3].astype(np.float32)
    a   = (warped[..., 3:4].astype(np.float32)) / 255.0
    out = face_bgr.astype(np.float32) * (1 - a) + bgr * a
    return np.clip(out, 0, 255).astype(np.uint8)


#  ОТЛАДОЧНАЯ ВИЗУАЛИЗАЦИЯ
def draw_debug(face_bgr: np.ndarray,
               named: Dict[str, np.ndarray],
               all_lms: np.ndarray,
               fit: Dict,
               glasses_anchors: Dict[str, np.ndarray] = None) -> np.ndarray:
    vis = face_bgr.copy()
    # Все landmark'и серым
    for p in all_lms:
        cv2.circle(vis, (int(p[0]), int(p[1])), 1, (100, 100, 100), -1)

    # Переносица — красный
    bridge_pt = named[fit['bridge_landmark']]
    cv2.circle(vis, (int(bridge_pt[0]), int(bridge_pt[1])), 5, (0, 0, 255), -1)
    cv2.circle(vis, (int(bridge_pt[0]), int(bridge_pt[1])), 5, (255, 255, 255), 1)

    # Зрачки — пурпурный
    for eye_name in ['left_iris', 'right_iris']:
        pt = named[eye_name]
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), -1)
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), 1)

    # Спроецированные якоря очков (мост, виски) — зелёный
    if 'M' in fit and glasses_anchors is not None:
        M = fit['M']
        for key in ['bridge_center', 'left_temple_attach', 'right_temple_attach']:
            if key in glasses_anchors:
                pt = _affine_transform(np.array([glasses_anchors[key]]), M)[0]
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), 1)

    # Подписи
    cv2.putText(vis, f"bridge: {fit['bridge_landmark']}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, f"L: {fit['left_ear_idx']}  R: {fit['right_ear_idx']}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


#  ПАЙПЛАЙН
def try_on_glasses(face_path: str,
                   glasses_path: str,
                   out_path: str = "result.jpg",
                   spec: GlassesSpec = GlassesSpec(),
                   model_path: str = "face_landmarker_v2_with_blendshapes.task",
                   debug_path: Optional[str] = "debug.jpg",
                   remove_bg: bool = False,
                   fill_lenses: bool = True,
                   knockout_lens: bool = False) -> None:

    face_bgr = cv2.imread(face_path)
    if face_bgr is None:
        raise ValueError(f"Не загружено: {face_path}")

    detector = FaceLandmarkDetector(model_path)
    all_lms, named = detector.detect(face_bgr)

    # Task 0: валидация геодезики
    validate_geodesic(face_bgr, named, verbose=True)

    # Task 1: реальный скейл очков
    scale = compute_face_scale(named, spec)
    print(f"[Task 1] px/mm = {scale['px_per_mm']:.3f}, "
          f"очки {spec.frame_width_mm}мм → {scale['glasses_width_px']:.0f}px")

    # Task 2: якоря очков (мост, концы заушников)
    glasses_rgba = load_glasses_rgba(glasses_path, force_remove_bg=remove_bg,
                                     fill_lenses=fill_lenses,
                                     knockout_lens=knockout_lens)

    # Отладка: сохраняем извлечённую альфу рядом с результатом
    if debug_path:
        alpha_dbg = debug_path.replace('.jpg', '_alpha.png')
        cv2.imwrite(alpha_dbg, glasses_rgba)
        print(f"Alpha: {alpha_dbg}")

    anchors = compute_glasses_anchors(spec, scale, glasses_rgba)

    # Task 3 + 4: оптимальные точки + аффинное преобразование
    fit = fit_glasses_by_landmarks(named, all_lms, spec, scale, anchors)

    # Рендер
    result = render_glasses(face_bgr, glasses_rgba, anchors['resize_to'], fit['M'])
    cv2.imwrite(out_path, result)
    print(f"Сохранено: {out_path}")

    if debug_path:
        dbg = draw_debug(face_bgr, named, all_lms, fit, glasses_anchors=anchors)
        cv2.imwrite(debug_path, dbg)
        print(f"Debug: {debug_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Примерка очков с реальными физическими размерами")
    ap.add_argument("--face",    required=True, help="фото лица")
    ap.add_argument("--glasses", required=True, help="фото очков")
    ap.add_argument("--out",     default="result.jpg")
    ap.add_argument("--model",   default="face_landmarker_v2_with_blendshapes.task")
    ap.add_argument("--debug",   default="debug.jpg")
    ap.add_argument("--remove-bg", action="store_true",
                    help="Принудительно удалить фон у очков через rembg/GrabCut "
                         "(нужно для обычных фото, не из интернет-магазина)")
    ap.add_argument("--no-fill-lenses", action="store_true",
                    help="НЕ заполнять область линз полупрозрачной заливкой. "
                         "Используй для оправ с полным ободом (Vogue, Ray-Ban, "
                         "и т.п.) — иначе внутри линз будет белый туман. "
                         "Для безободковых очков заливка нужна.")
    ap.add_argument("--knockout-lens", action="store_true",
                    help="Убрать почти-белые пиксели внутри оправы "
                         "(чтобы стекло из каталожного фото стало прозрачным). "
                         "Используй когда внутри линз остаётся белая заливка "
                         "после rembg.")

    # Параметры очков (мм). По умолчанию — Vogue VO4094.
    ap.add_argument("--bridge",      type=float, default=18.0,  help="Переносица (мост), мм")
    ap.add_argument("--temple",      type=float, default=135.0, help="Длина заушника, мм")
    ap.add_argument("--lens-width",  type=float, default=52.0,  help="Ширина линзы, мм")
    ap.add_argument("--lens-height", type=float, default=44.0,  help="Высота линзы, мм")
    ap.add_argument("--lens-diam",   type=float, default=55.0,  help="Диаметр линзы, мм")
    ap.add_argument("--frame-width", type=float, default=129.0, help="Ширина оправы, мм")

    a = ap.parse_args()
    spec = GlassesSpec(
        bridge_mm=a.bridge,
        temple_mm=a.temple,
        lens_width_mm=a.lens_width,
        lens_height_mm=a.lens_height,
        lens_diameter_mm=a.lens_diam,
        frame_width_mm=a.frame_width,
    )
    try_on_glasses(a.face, a.glasses, a.out, spec=spec,
                   model_path=a.model, debug_path=a.debug,
                   remove_bg=a.remove_bg,
                   fill_lenses=not a.no_fill_lenses,
                   knockout_lens=a.knockout_lens)
