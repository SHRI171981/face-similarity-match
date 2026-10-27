import numpy as np
import face_recognition
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps
from typing import BinaryIO, List, Tuple, Union

# Longest image side fed to the detector; full 12 MP phone photos make the CNN detector far slower
MAX_DIMENSION = 1600

def load_image(source: Union[str, Path, BinaryIO]) -> np.ndarray:
    """
    Loads an image as an RGB matrix, applying the EXIF orientation tag so
    phone photos taken in portrait are upright before face detection,
    and downscaling it to at most MAX_DIMENSION pixels on its longest side.
    """
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        return np.array(image.convert("RGB"))

def encode_faces(img_array: np.ndarray) -> List[np.ndarray]:
    """
    Detects faces with dlib's CNN model, which catches profile, angled and
    sunglasses-wearing faces the default HOG detector misses, then encodes each one.
    """
    locations = face_recognition.face_locations(img_array, model="cnn")
    return face_recognition.face_encodings(img_array, known_face_locations=locations)

def extract_faces(path: Path) -> List[Tuple[np.ndarray, str]]:
    """
    Loads an image matrix into memory and extracts the encoding of every face in it.
    Returns a list of (encoding, filename) tuples, empty if detection fails.
    """
    try:
        img_array = load_image(path)
        return [(encoding, path.name) for encoding in encode_faces(img_array)]
    except Exception:
        return []

def build_index_parallel(image_dir: str, output_file: str) -> None:
    """
    Distributes I/O and computation workload across independent processes.
    Serializes extracted vectors to a compressed binary format, one row per face.
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in valid_extensions
    ]

    encodings = []
    filenames = []
    # The CNN detector needs several GB per process, so one worker per core exhausts RAM
    with ProcessPoolExecutor(max_workers=1) as executor:
        results = executor.map(extract_faces, image_paths)

        for faces in results:
            for encoding, filename in faces:
                encodings.append(encoding)
                filenames.append(filename)

    np.savez_compressed(
        output_file,
        encodings=np.array(encodings),
        filenames=np.array(filenames)
    )

if __name__ == "__main__":
    build_index_parallel("./mock_drive_images", "face_index.npz")
