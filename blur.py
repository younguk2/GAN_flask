#블러 효과 함수
#import dlib
import cv2
import numpy as np
from PIL import Image, ImageEnhance
# def set_blur_image(image_path,blurStyle):
#     # 이미지 불러오기
#     image_path = image_path
#     image = cv2.imread(image_path)

#     # 얼굴, 눈 검출을 위한 Haar cascade 분류기 불러오기
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#     # dlib 얼굴 특징 감지기 및 모델 불러오기 (입술, 눈썹)
#     predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#     detector = dlib.get_frontal_face_detector()

#     # 그레이스케일로 변환
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 얼굴 검출
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # 마스크 초기화
#     mask = np.zeros_like(image)

#     # 각 얼굴에 대해 마스크 생성 및 눈, 눈썹, 입술 제외
#     for (x, y, w, h) in faces:
#         face_gray = gray[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(face_gray)
        
#         # dlib 얼굴 영역 감지 및 특징
#         dlib_rectangle = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
#         detected_landmarks = predictor(gray, dlib_rectangle).parts()
#         landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
        
#         # 얼굴 영역의 중심 좌표
#         center_x = x + w // 2
#         center_y = y + h // 2
        
#         # 얼굴 윤곽을 따르는 타원 생성
#         mask = cv2.ellipse(mask, (center_x, center_y), (w // 2, h // 2), 0, 0, 360, (255, 255, 255), -1)

#         # 눈, 눈썹, 입술 영역 마스크에서 제거
#         # 여기서는 예시로 눈 영역 제외만 구현. 눈썹과 입술 영역 제외는 landmarks를 사용하여 구현해야 함.
#         # 눈 영역 제외
#     for (ex, ey, ew, eh) in eyes:
#         eye_center_x = x + ex + ew // 2
#         eye_center_y = y + ey + eh // 2
#         eye_radius = int(min(ew, eh) * 0.35)  # 눈의 반지름을 어느정도로 할지 결정, 예시로 현재 눈의 가로 또는 세로 길이의 35%로 설정
        
#         # 눈 영역 주변에 원 모양으로 마스크 생성하여 눈 영역 제외
#         cv2.circle(mask, (eye_center_x, eye_center_y), eye_radius, (0, 0, 0), -1)

#     x = blurStyle
#     # 가우시안 블러 적용
#     filtered_image = cv2.bilateralFilter(image, 20, x, x)

#     # 마스크 적용
#     filter_result = np.where(mask == np.array([255, 255, 255]), filtered_image, image)
#     # 결과 이미지 출력
#     cv2.imwrite(image_path, filter_result)

def adjust_brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

def adjust_saturation(image, saturation_factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation_factor)