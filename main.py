from dotenv import load_dotenv
import openai
import os
import cv2
import numpy as np
from PIL import Image
import requests
from rembg import remove
def delete_background(img_path,imgNum):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        padding = 13000  # 원하는 패딩 크기
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)

        cropped = img[y:y + h, x:x + w]

        # Remove background using rembg
        output = remove(cropped)
        # Save the result
        cv2.imwrite("nobackground{}.png".format(imgNum), output)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        return img
    
def extend_image(image_path, new_height):
    # 이미지 열기
    img = Image.open(image_path)
    
    # 이미지 크기 가져오기
    width, height = img.size
    
    # 새로운 이미지 크기 계산
    new_size = (width, new_height)
    
    # 배경이 늘어난 이미지 생성
    new_img = Image.new("RGB", new_size, (255, 255, 255))  # 흰색 배경
    
    # 기존 이미지를 새로운 이미지에 붙이기
    new_img.paste(img, (0, 0))
    
    return new_img

def suit_style(file_path,imgNum):
    load_dotenv()
    openai.api_key = os.environ.get('REACT_APP_API_KEY')
    client = openai

    # Haar Cascade 파일 경로 (OpenCV 설치 시 제공)
    haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    # Haar Cascade 분류기 로드
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    # 이미지 읽기
    image_path = file_path

    input_image_path = file_path
    output_image_path = file_path
    new_height = 340  # 새로운 이미지의 높이 300

    extended_img = extend_image(input_image_path, new_height)
    extended_img.save(output_image_path)

    image = cv2.imread(image_path)
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 마스크 생성
    mask = np.zeros_like(image)
    # 검출된 얼굴에 대해 마스크 적용
    for (x, y, w, h) in faces:
        # 원형 마스크 생성 (중심 좌표와 반지름)
        center = (x + w // 2, y + h // 2)
        radius = int(max(w, h) // 1.5)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # 원본 이미지와 마스크를 AND 연산하여 얼굴만 남김
    masked_image = cv2.bitwise_and(image, mask)
    # 결과 저장
    output_path = './masked.png'
    cv2.imwrite(output_path, masked_image)






    # 이미지를 읽고 RGBA로 변환
    image = Image.open(image_path).convert("RGBA")

    # 마스크 이미지를 읽고 RGBA로 변환
    mask_image = Image.open(output_path).convert("L")  # Grayscale로 변환
    mask_rgba = Image.new("RGBA", mask_image.size)
    mask_rgba.paste(mask_image, (0, 0), mask_image)

    # 변환된 이미지를 저장
    image.save('./rgba.png')
    mask_rgba.save('./masked_rgba.png')


    response = client.images.edit(
        image=open('./rgba.png', "rb"),
        mask=open('./masked_rgba.png', "rb"),
        prompt="A Person not wearing a suit, transformed to wear a formal suit photorealistic to attend job ,no background, Make your neck color similar to your face skin color",
        n=1,
        size="1024x1024"
    )

    image_url = response.data[0].url
    print("Generated image URL:", image_url)

    # 이미지 URL로부터 이미지 파일을 다운로드하고 저장
    image_response = requests.get(image_url)

    # 파일로 저장
    with open("conversionref{}.jpg".format(imgNum), "wb") as f:
        f.write(image_response.content)

