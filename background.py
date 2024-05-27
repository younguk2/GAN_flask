import cv2
from PIL import Image, ImageEnhance
from rembg import remove

# 이미지를 크롭하는 함수
def crop_image(image_path, cropped_image_path):
    # 이미지 로드
    img = cv2.imread(image_path)

    # 얼굴 인식
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 얼굴 크롭 및 저장
    for i, (x, y, w, h) in enumerate(faces):
        #cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
        cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
        cv2.imwrite(cropped_image_path, cropped)

#배경 합성 함수 
def merge_images(background_path, person_path, output_path):
    # 배경 이미지 열기 및 크기 조정
    background = Image.open(background_path).convert('RGBA')
    background_resized = background.resize((512,512))

    # 인물 이미지 열기
    person = Image.open(person_path).convert('RGBA')

    # 새로운 프레임 생성
    frame = Image.new('RGBA', background_resized.size, (255, 255, 255, 0))

    # 인물 이미지를 프레임의 가운데에 배치
    x_offset = (background_resized.width - person.width) // 2
    y_offset = (background_resized.height - person.height) // 2
    frame.paste(person, (x_offset, y_offset), person)

    # 배경 이미지와 프레임 합성
    merged_image = Image.alpha_composite(background_resized, frame)

    # 이미지 저장
    merged_image.save(output_path)

#배경 제거 함수
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