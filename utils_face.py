
# utils_face.py      얼굴 검출 유틸(OpenCV)

import cv2
from typing import List, Tuple

# OpenCV가 제공하는 기본 haarcascade 경로(환경마다 경로 다를 수 있습니다)
# 아래 파일을 프로젝트 루트에 복사해두고 상대경로로 쓰는 걸 권장:
# haarcascade_frontalface_default.xml
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def detect_faces_bgr(bgr_image, scaleFactor=1.1, minNeighbors=5) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    boxes = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                          minSize=(30,30))
    # boxes: (x, y, w, h)
    return boxes
