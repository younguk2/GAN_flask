from flask import Flask, render_template, request
import logging
import boto3
from dotenv import load_dotenv
import os 
import uuid
from PIL import Image, ImageEnhance
import numpy as np
import os
import argparse
from flask_cors import CORS
from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver

import cv2
from flask import Flask, render_template, request
import logging
import base64
import json
from io import BytesIO
from werkzeug.utils import secure_filename
from rembg import remove
#from set_blur_image import skin
import requests
from flask import Flask, jsonify, request
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from main import suit_style
def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


    
def extract_right_half(image_path, output_dir):
    # 이미지 파일 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # 이미지의 너비 및 높이 확인
    height, width = image.shape[:2]

    # 이미지를 반으로 나누기
    right_half = image[:, width // 2:]

    # 이미지 파일 이름 추출
    filename = os.path.basename(image_path)
    filename_no_extension, extension = os.path.splitext(filename)

    # 오른쪽 부분만 추출된 이미지 저장
    output_path = os.path.join(output_dir, f"{filename_no_extension}{extension}")
    cv2.imwrite(output_path, right_half)
    print(f"{output_path}에 오른쪽 부분 이미지 저장 완료")

def conversion_image(ref_path):
    print("*************************")
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')
    parser.add_argument('--ref_dir', type=str, default=ref_path,
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')
    parser.add_argument('--con_dir', type=str, default='expr/results/celeba_hq',
                        help='output directory when aligning faces')
    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)

    # 인자 추가
    parser.add_argument("--mode", type=str, default="sample", help="모드 설정")
    parser.add_argument("--checkpoint_dir", type=str, default="expr/checkpoints/celeba_hq", help="체크포인트 디렉토리")
    parser.add_argument("--result_dir", type=str, default="conversion", help="결과 디렉토리")
    parser.add_argument("--src_dir", type=str, default="downloadImage", help="소스 디렉토리")

    args = parser.parse_args()
    print(args)
    main(args)
    print("*************************")
    #오른쪽 사진만 추출하는 코드
    print(args.con_dir)
    files = ["conversionref1.jpg","conversionref2.jpg","conversionref3.jpg","conversionref4.jpg","conversionref5.jpg","conversionref6.jpg","conversionref7.jpg","conversionref8.jpg","conversionref9.jpg","conversionref10.jpg"]
    for files in os.listdir(args.con_dir):
        if files.endswith(".jpg") or files.endswith(".jpeg"):
            image_path = os.path.join(args.con_dir, files)
            extract_right_half(image_path, args.con_dir)

def adjust_brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

def adjust_saturation(image, saturation_factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation_factor)

# 디렉토리 안의 파일 모두 제거하는 함수
def clear_directory(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        os.remove(file_path)

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
    background_resized = background.resize((256,256))

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
    
#블러 효과 함수
# import dlib
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

# load .env
load_dotenv()
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION')
client = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_DEFAULT_REGION
                      )

app = Flask(__name__)
CORS(app, origins=['https://jobhakdasik.site'], supports_credentials=True)
bucket = r'jobhakdasik2000-bucket'
local_dir = r'.\downloadImage\download.png'


# 취업사진 합성 요청
@app.route('/profile/edit',methods=['POST'])
def editPicture():
    #Spring에서 받은 JSON 형식의 데이터들
    #이미지 파일
    imagePath = request.files['file'].read()
    filename = secure_filename(request.files['file'].filename)
    extension = filename.rsplit('.', 1)[1].lower()

    imagePath = BytesIO(imagePath)
    image = Image.open(imagePath)

    #성별 male : 남자 , female 여자
    sex = str(request.form['sex'])
    #명도 1을 기준으로 높을 수록 밝음
    brightness = float(request.form['brightness'])
    #채도 1을 기준으로 높을 수록 진함
    saturation = float(request.form['saturation'])
    #생성 여부 ture : GAN 합성 시작 , false : GAN 합성 안함
    conversion = str(request.form['conversion'])
    #정장 합성 suit1,suit2... 만약 정장은 넣지 않는다면?..
    suitStyle = str(request.form['suitstyle'])
    #머리카락 스타일 femalehair1,femalehair2
    hairStyle = str(request.form['hairstyle'])
    #블러효과의 정도 low,mid,high 
    blurStyle = int(request.form['blurstyle'])
    #입술 옵션 true false
    lipOption = str(request.form['lipoption'])
    #배경 선택 None , background1(파랑 배경),bacground2(노랑 배경),background3(빨강 배경)
    background = str(request.form['background'])
    #유저 ID
    loginId = str(request.form['loginId'])

    result={
        'imagePath':imagePath,
        'sex':sex,
        'bright':brightness,
        'saturation':saturation,
        'conversion':conversion,
        'suitStyle':suitStyle,
        'hairStyle':hairStyle,
        'blurStyle':blurStyle,
        'lipOption':lipOption,
        'background':background,
        'loginId':loginId
    }
    print(result)
    # 크롭된 이미지를 저장할 디렉토리 생성
    cropped_dir = "./cropped"
    os.makedirs(cropped_dir, exist_ok=True)

    save_path = f"{cropped_dir}/{filename}"
    image.save(save_path)
    # 크롭 작업을 수행할 이미지 경로
    cropped_image_path = f"{cropped_dir}/{filename}"
    # 이미지 크롭 및 저장
    print('*************cropped*********')
    print(save_path)
    print(cropped_image_path)
    crop_image(save_path, cropped_image_path)
    cropped_image_path = Image.open(cropped_image_path)

    save_path = os.path.join(r'./downloadImage/d1','download.'+extension)
    cropped_image_path.save(save_path)
    save_path = os.path.join(r'./downloadImage/d2','download.'+extension)
    cropped_image_path.save(save_path)

    return_data=[]
    if conversion == 'true':
        ref_dir = './ref/'+hairStyle
        conversion_image(ref_dir)
       
        file_path = [r'./conversionref1.jpg',
                     r'./conversionref2.jpg',
                     r'./conversionref3.jpg',
                     r'./conversionref4.jpg',
                     r'./conversionref5.jpg',
                     r'./conversionref6.jpg',
                     ]

        # 이미지 반 자르기
        for path in file_path:
            img = cv2.imread(path)
            if img is None:
                print(f"이미지를 읽을 수 없습니다: {path}")
                return
            height, width = img.shape[:2]
            img = img[:, width // 2:]
            cv2.imwrite(path, img)

        folder_name = loginId
        if not folder_name:
            return jsonify({'error': 'Folder name is required'}), 400

        # 폴더 생성 (S3에서는 폴더가 객체 키의 접미사로 생성됨)
        folder_key = f'{folder_name}/'  # 끝에 슬래시를 붙여 폴더로 인식되게 함
        client.put_object(Bucket=bucket, Key=folder_key)

        if sex == 'male':
            for imgNum in range(1,6):
                suit_style(file_path[imgNum],imgNum+1)
            if(background != 'none'):
                for imgNum in range(1,6):
                    # 블러 효과 함수 호출
                    #set_blur_image(file_path[imgNum],blurStyle)
                    delete_background(file_path[imgNum], imgNum)
                    # 배경 합성
                    background_path = './background/' + background + ".png"
                    person_path = 'nobackground{}.png'.format(imgNum)
                    output_path = './res{}.png'.format(imgNum)
                    
                    # 이미지 합치기
                    merge_images(background_path, person_path, output_path)
                    
                    # 객체 경로 생성
                    object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.jpg'
                    
                    # S3 업로드 코드 (주석 처리된 부분을 필요에 따라 활성화하세요)
                    response = client.upload_file(output_path, bucket, object_path, ExtraArgs={'ACL': 'public-read'})
                    
                    # 결과 데이터 추가
                    return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/' + object_path, 'message': 'success'})

            else:
                for imgNum in range(1,6):
                    # 블러 효과 함수 호출
                    #set_blur_image(file_path[imgNum],blurStyle)
                    # 객체 경로 생성
                    object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.jpg'
                    # S3 업로드 코드 (주석 처리된 부분을 필요에 따라 활성화하세요)
                    response = client.upload_file(file_path[imgNum], bucket, object_path, ExtraArgs={'ACL': 'public-read'})
                    # 결과 데이터 추가
                    return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/' + object_path, 'message': 'success'})
        else:
            for imgNum in range(0,5):
                suit_style(file_path[imgNum],imgNum+1)
            if(background != 'none'):
                # 블러 효과 함수 호출
                #set_blur_image(file_path[imgNum],blurStyle)
                for imgNum in range(0,5):
                    delete_background(file_path[imgNum], imgNum+1)
                    # 배경 합성
                    background_path = './background/' + background + ".png"
                    person_path = 'nobackground{}.png'.format(imgNum+1)
                    output_path = './res{}.png'.format(imgNum+1)
                    
                    # 이미지 합치기
                    merge_images(background_path, person_path, output_path)
                    
                    # 객체 경로 생성
                    object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.jpg'
                    
                    # S3 업로드 코드 (주석 처리된 부분을 필요에 따라 활성화하세요)
                    response = client.upload_file(output_path, bucket, object_path, ExtraArgs={'ACL': 'public-read'})
                    
                    # 결과 데이터 추가
                    return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/' + object_path, 'message': 'success'})
            else:
                for imgNum in range(0,5):
                    # 블러 효과 함수 호출
                    #set_blur_image(file_path[imgNum],blurStyle)
                    # 객체 경로 생성
                    object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.jpg'
                    # S3 업로드 코드 (주석 처리된 부분을 필요에 따라 활성화하세요)
                    response = client.upload_file(file_path[imgNum], bucket, object_path, ExtraArgs={'ACL': 'public-read'})
                    # 결과 데이터 추가
                    return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/' + object_path, 'message': 'success'})
           
        # 디렉토리 경로
        dir_path1 = r'./downloadImage/d1'
        dir_path2 = r'./downloadImage/d2'

        # 디렉토리 안의 파일 모두 제거
        for file_name in os.listdir(dir_path1):
            file_path = os.path.join(dir_path1, file_name)
            os.remove(file_path)

        for file_name in os.listdir(dir_path2):
            file_path = os.path.join(dir_path2, file_name)
            os.remove(file_path)
    else:
        # .h5 모델을 이용해서 사진을 생성하거나 밝기,색상 조절을 한다.
        image_path = r".\downloadImage\d1\download."+extension
        image = Image.open(image_path)

        # 밝기 조절 
        brightness_factor = brightness  # 1.5 => 밝기 50% 증가   0.5 => 밝기 50% 감소 
        image = adjust_brightness(image, brightness_factor)
        image.show()

        # 채도 조절 
        saturation_factor = saturation  # 0.5 => 채도 50% 감소  1.5 채도 50% 증가
        image = adjust_saturation(image, saturation_factor)
        image.show()

        output_path = r"./res."+extension
        image.save(output_path)

        if(background != 'none'):
            #배경 없애기
            delete_background(output_path)
            #배경 합성
            background_path = './background/'+background+".png"
            person_path = 'nobackground.png'
            output_path = './res.'+extension
            # 이미지 합치기
            merge_images(background_path, person_path, output_path)

        #이미지를 아마존 S3에 업로드한다 (idPhoto)
        object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.'+extension
        file_path = r'./res.'+extension
        response = client.upload_file(file_path, bucket, object_path,ExtraArgs={'ACL': 'public-read'})
        return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/'+object_path, 'message': 'success'})

        # 디렉토리 경로
        dir_path1 = r'./downloadImage/d1'
        dir_path2 = r'./downloadImage/d2'

        # 디렉토리 안의 파일 모두 제거
        for file_name in os.listdir(dir_path1):
            file_path = os.path.join(dir_path1, file_name)
            os.remove(file_path)

        for file_name in os.listdir(dir_path2):
            file_path = os.path.join(dir_path2, file_name)
            os.remove(file_path)

    #crop된 파일들 저장된 거 삭제
    #clear_directory(cropped_dir)
    return return_data

@app.route('/rembg')
def removeBackground():
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open('conversionref2.jpg', 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'Gnx4XP54gG1VofH1UukLXXh7'},
    )
    if response.status_code == requests.codes.ok:
        with open('no-bg.png', 'wb') as out:
            out.write(response.content)
    else:
        print("Error:", response.status_code, response.text)
    return 'a'

if __name__=='__main__':
    app.run(host="localhost",port=12300)