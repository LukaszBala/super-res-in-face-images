from gfpgan_facade import gfpgan_upscale
from real_esrgan_facade import upscale_image as real_esrgan_upscale
from RLFN.rlfn_forward_pass import super_resolve_image
from face_detection.face_detection_utils import detect_faces
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
from PIL import Image
import io
from typing import Annotated
from opencv_resizes import bicubic_resize, fsrcnn, edsr

app = FastAPI()

upsampling_methods = {
    'bicubic': bicubic_resize,
    'fsrcnn': fsrcnn,
    'edsr': edsr,
    'real-esrgan': real_esrgan_upscale,
    'gfpgan': gfpgan_upscale,
    'trained_model': super_resolve_image
}


def validate_jpeg_buffer(buffer):
    try:
        img = Image.open(io.BytesIO(buffer))
        img.verify()
        return True
    except Exception as e:
        print(f"Invalid image buffer: {e}")
        return False


@app.post("/detect_faces/")
async def detect_faces_from_file(file: UploadFile = File(...)):
    # Save the uploaded file
    print('bedziemy detect robic');
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    if not validate_jpeg_buffer(nparr):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid JPEG image.")

    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
    except Exception as e:
        print(f"Error decoding image: {e}")
        raise HTTPException(status_code=500, detail="Image decoding failed.")
    #
    faces = detect_faces(img)
    #
    # # Convert coordinates to JSON serializable format
    coordinates = [{"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)} for
                   (top, right, bottom, left) in faces]

    # coordinates = []

    return JSONResponse(content=coordinates)


@app.post("/process_face/")
async def process_face(file: UploadFile = File(...), scale: Annotated[float, Form()] = 4.0,
                       method: Annotated[str, Form()] = 'bicubic'):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    height, width = img.shape[:2]
    max_dimension = 600
    if max(width, height) > max_dimension:
        print('reshaping');
        scaling_factor = max_dimension / max(width, height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    print(f'Image received with scale: {scale} and method: {method}')
    print(f'shape after reshape', img.shape)

    # Select the upsampling method
    if method not in upsampling_methods:
        return {"error": "Invalid upsampling method"}

    upsampling_function = upsampling_methods[method]

    # Upscale the image
    upscaled_img = upsampling_function(img)

    # Convert the image to JPEG format
    _, img_encoded = cv2.imencode('.jpg', upscaled_img)
    img_bytes = img_encoded.tobytes()

    # Return the image as a StreamingResponse
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run('server:app', host="0.0.0.0", port=8000, reload=True)
