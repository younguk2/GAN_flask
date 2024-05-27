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
import requests
from flask import Flask, jsonify, request
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import random

from dallE import suit_style
from dallE import removeBackground
from background import *
from conversion import *
from blur import *

# 디렉토리 안의 파일 모두 제거하는 함수
def clear_directory(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        os.remove(file_path)

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

    save_path = os.path.join(r'./downloadImage/d1','download.'+extension)
    cropped_image_path.save(save_path)
    save_path = os.path.join(r'./downloadImage/d2','download.'+extension)
    cropped_image_path.save(save_path)

    return_data=[]
    if conversion == 'true':
        random_value = random.choice(['', '1', '2'])
        ref_dir = './ref/'+hairStyle+random_value
        print('현재 선택된 ref_dir = '+ref_dir)
        conversion_image(ref_dir)
       
        file_path = [r'./conversionref1.jpg',
                     r'./conversionref2.jpg',
                     r'./conversionref3.jpg'
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
        folder_key = f'idPhoto/{folder_name}/'  # 끝에 슬래시를 붙여 폴더로 인식되게 함
        client.put_object(Bucket=bucket, Key=folder_key)

        if sex == 'male':
            suit_style(file_path[1],2,sex)
            suit_style(file_path[2],3,sex)
            if(background != 'none'):
                for imgNum in range(1,3):
                    if(imgNum < 3):
                        # 블러 효과 함수 호출
                        #set_blur_image(file_path[imgNum],blurStyle)

                        # 파이썬 라이브러리를 이용한 배경제거 
                        delete_background(file_path[imgNum], imgNum)

                        # API를 이용한 배경제거 
                        #removeBackground(file_path[imgNum],imgNum)

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
                for imgNum in range(1,3):
                    # 블러 효과 함수 호출
                    #set_blur_image(file_path[imgNum],blurStyle)
                    # 객체 경로 생성
                    object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.jpg'
                    # S3 업로드 코드 (주석 처리된 부분을 필요에 따라 활성화하세요)
                    response = client.upload_file(file_path[imgNum], bucket, object_path, ExtraArgs={'ACL': 'public-read'})
                    # 결과 데이터 추가
                    return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/' + object_path, 'message': 'success'})
        else:
            suit_style(file_path[0],1,sex)
            suit_style(file_path[1],2,sex)
            if(background != 'none'):
                # 블러 효과 함수 호출
                #set_blur_image(file_path[imgNum],blurStyle)
                for imgNum in range(0,2):
                    if(imgNum < 2):
                        # 파이썬 라이브러리를 이용한 배경제거 
                        delete_background(file_path[imgNum], imgNum+1)

                        # API를 이용한 배경제거 
                        #removeBackground(file_path[imgNum],imgNum+1)

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
                for imgNum in range(0,2):
                    # 블러 효과 함수 호출
                    #set_blur_image(file_path[imgNum],blurStyle)
                    # 객체 경로 생성
                    object_path = r'idPhoto/{}/'.format(folder_name) + str(uuid.uuid4()) + 'download.jpg'
                    # S3 업로드 코드 (주석 처리된 부분을 필요에 따라 활성화하세요)
                    response = client.upload_file(file_path[imgNum], bucket, object_path, ExtraArgs={'ACL': 'public-read'})
                    # 결과 데이터 추가
                    return_data.append({'UploadedFilePath': 'https://jobhakdasik2000-bucket.s3.ap-northeast-2.amazonaws.com/' + object_path, 'message': 'success'})
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

if __name__=='__main__':
    app.run(host="localhost",port=12300)