import io
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from extract import load_image, encode_faces

app = FastAPI(title="Face Vector Search MVP")
app.mount("/images", StaticFiles(directory="mock_drive_images"), name="images")

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse("static/index.html")

# dlib's standard same-person cutoff: encodings closer than this are treated as a match
MATCH_THRESHOLD = 0.6

try:
    index_data = np.load("face_index.npz")
    INDEX_ENCODINGS = index_data["encodings"]
    INDEX_FILENAMES = index_data["filenames"]
except FileNotFoundError:
    INDEX_ENCODINGS = np.array([])
    INDEX_FILENAMES = np.array([])

class SearchResult(BaseModel):
    filename: str
    distance: float

@app.post("/search", response_model=List[SearchResult])
async def search_face(file: UploadFile = File(...), top_k: int = 5, threshold: float = MATCH_THRESHOLD):
    """
    Accepts an image upload, extracts the primary face encoding, and returns the
    photos containing a face within `threshold` Euclidean distance, closest first.
    """
    if INDEX_ENCODINGS.size == 0:
        raise HTTPException(status_code=500, detail="Vector index not initialized.")

    image_bytes = await file.read()
    img_array = load_image(io.BytesIO(image_bytes))

    query_encodings = encode_faces(img_array)

    if not query_encodings:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    query_vector = query_encodings[0]

    distances = face_recognition.face_distance(INDEX_ENCODINGS, query_vector)

    # The index holds one row per face, so keep only the best-matching face per photo
    best_per_file = {}
    for idx in np.argsort(distances):
        if distances[idx] > threshold:
            break
        filename = str(INDEX_FILENAMES[idx])
        if filename not in best_per_file:
            best_per_file[filename] = float(distances[idx])

    results = [
        SearchResult(filename=filename, distance=distance)
        for filename, distance in list(best_per_file.items())[:top_k]
    ]

    return results
