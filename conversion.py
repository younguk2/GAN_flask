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
    files = ["conversionref1.jpg","conversionref2.jpg","conversionref3.jpg"]
    for files in os.listdir(args.con_dir):
        if files.endswith(".jpg") or files.endswith(".jpeg"):
            image_path = os.path.join(args.con_dir, files)
            extract_right_half(image_path, args.con_dir)