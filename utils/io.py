import cv2
import os
import re

_FRAME_ID_RE = re.compile(r"(?:^|[^a-zA-Z0-9])frame_(?P<id>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_LAST_INT_RE = re.compile(r"(\d+)")


def parse_frame_id(path: str) -> int | None:
    """
    Extrae un frame_id numérico desde el nombre de archivo.

    Prioriza patrones tipo: frame_000610.png -> 610.
    Fallback: último grupo de dígitos en el basename.
    """
    if path is None:
        return None
    name = os.path.basename(str(path))

    m = _FRAME_ID_RE.search(name)
    if m is not None:
        return int(m.group("id"))

    ints = _LAST_INT_RE.findall(name)
    if not ints:
        return None
    return int(ints[-1])

def iter_frames_from_folder(folder_path):
    """
    Itera imágenes de una carpeta en orden lexicográfico.
    Devuelve (frame, frame_name).
    """
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(exts)
    )

    for fname in files:
        path = os.path.join(folder_path, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue
        yield frame, fname


def decode_action(key: int) -> str:
    if key is None:
        return "none"

    k = int(key)

    if k in (ord("q"), 27):
        return "quit"

    if k in (ord("c"),):
        return "catchup"

    if k in (32,):  # SPACE
        return "toggle_auto"

    # Letras
    if k in (ord("a"), ord("A")):
        return "left"
    if k in (ord("d"), ord("D")):
        return "right"

    # Flechas (waitKeyEx)
    if k in (2424832, 65361):  # left (según backend)
        return "left"
    if k in (2555904, 65363):  # right (según backend)
        return "right"

    return "none"


def list_image_files(folder: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files: list[str] = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        _, ext = os.path.splitext(name.lower())
        if ext in exts:
            files.append(p)
    files.sort()
    return files


def read_bgr(path: str):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def basename(path: str) -> str:
    return os.path.basename(path)
