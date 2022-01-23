"""
@author: Phi-C
@date: 20220118
"""
import os
import glob
import time
import random
from multiprocessing import Process, Queue
import sys
sys.path.append('/mnt/e/workspace/FaceX-Zoo/')
sys.path.append('/mnt/e/workspace/FaceX-Zoo/face_sdk')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import yaml
import cv2
import torch
import numpy as np
from utils.custom_decorators import calc_execute_time
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)


def face_detection(data, q):
    '''
    Reference: face_sdk/api_usage/face_detect.py
    '''
    face_img_root, face_det_root, face_ali_root = data

    # 如果把face_detection模型和face_alignment模型作为global varialbe, 那么主进程和人脸检测(人脸对齐)进程都会有一份face_detection(face_alignment)模型
    # 会出现报错：Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    # common setting for all model, need not modify.
    model_path = 'models'
    # model setting, modified along with model
    scene = 'non-mask'
    # model configuration for FACE DETECTION
    det_model_category = 'face_detection'
    det_model_name =  model_conf[scene][det_model_category]
    logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, det_model_category, det_model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')
    try:
        det_model, det_cfg = faceDetModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face detection model!')
    faceDetModelHandler = FaceDetModelHandler(det_model, 'cuda:0', det_cfg)

    def detectionCore(img_root, det_root, ali_root):
        # read image
        image_list = os.listdir(img_root)
        for image_name in image_list:
            text_name = image_name.split('.')[0] + '.txt'
            image_path = os.path.join(img_root, image_name)
            det_file_path = os.path.join(det_root, text_name)
            ali_file_path = os.path.join(ali_root, text_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            try:
                dets = faceDetModelHandler.inference_on_image(image)
            except Exception as e:
                logger.error('Face detection failed!')
                logger.error(e)
                sys.exit(-1)
            else:
                logger.info('Successful face detection!')

            bboxs = dets
            print(det_file_path)
            with open(det_file_path, "w") as fd:
                for box in bboxs:
                    line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
                        str(int(box[2])) + " " + str(int(box[3])) + " " + \
                        str(box[4]) + " \n"
                    fd.write(line)

            if len(bboxs) > 0:
                q.put((image_path, det_file_path, ali_file_path))
            logger.info('Successfully generate face detection results!')        

    image_list = glob.glob(face_img_root + '/*/*.jpg')
    if len(image_list) == 0:
        # 所有图片都放在face_img_root下
        detectionCore(face_img_root, face_det_root, face_ali_root)
    else:
        # face_img_root下是各ID文件夹, 图片放在ID文件夹下
        person_ids = os.listdir(face_img_root)
        for person_id in person_ids[:10]:
            img_person_root = os.path.join(face_img_root, person_id)
            det_person_root = os.path.join(face_det_root, person_id)
            ali_person_root = os.path.join(face_ali_root, person_id)
            if not os.path.exists(det_person_root):
                os.makedirs(det_person_root)
            if not os.path.exists(ali_person_root):
                os.makedirs(ali_person_root)
            detectionCore(img_person_root, det_person_root, ali_person_root)
    q.put(None)

def face_alignment(q):
    '''
    Reference: face_sdk/api_usage/face_alignment.py
    '''
    # common setting for all model, need not modify.
    model_path = 'models'
    # model setting, modified along with model
    scene = 'non-mask'
    # model configuration for FACE ALIGNMENT
    ali_model_category = 'face_alignment'
    ali_model_name =  model_conf[scene][ali_model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, ali_model_category, ali_model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')
    try:
        ali_model, ali_cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face landmark model!')
    faceAlignModelHandler = FaceAlignModelHandler(ali_model, 'cuda:0', ali_cfg)

    while True:
        while q.empty():
            time.sleep(1)
        item = q.get()
        if item is None:
            break
        image_path, det_file_path, ali_file_path = item

        # read image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        with open(det_file_path, 'r') as f:
            lines = f.readlines()
        try:
            for i, line in enumerate(lines):
                line = line.strip().split()
                det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
                landmarks = faceAlignModelHandler.inference_on_image(image, det)

                with open(ali_file_path, "w") as fd:
                    for (x, y) in landmarks.astype(np.int32):
                        line = str(x) + ' ' + str(y) + ' '
                        fd.write(line)
        except Exception as e:
            logger.error('Face landmark failed!')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Successful face landmark!')

def get_info_file(alignment_root, face_info_file, mask_temp_file):
    alignment_files = glob.glob(alignment_root + '/*/*.txt')

    def get_info_file_core(src_root, id=None):
        files = os.listdir(src_root)
        if id is None:
            prefix = ''
        else:
            prefix = id + '/'
        # 保存图片的人脸关键点信息
        with open(face_info_file, 'a') as f1:
            for file in files:
                file_path = os.path.join(src_root, file)
                img_name = prefix + file.split('.')[0] + '.jpg'
                with open(file_path, 'r') as f2:
                    lms = f2.readline()
                lms = lms.strip().split(' ')
                assert (len(lms) % 212 == 0), '{} {}'.format(file_path, len(lms))
                num = len(lms) // 212
                for idx in range(num):
                    landmarks = ','.join(lms[idx*212: (idx+1)*212])
                    content = '{} {}\n'.format(img_name, landmarks)
                    f1.write(content)
        # 为图片选择口罩模板
        with open(mask_temp_file, 'a') as f1:
            for file in files:
                img_name = prefix + file.split('.')[0] + '.jpg'
                temp_name = '{}.png'.format(random.randint(0, 7))
                content = '{} {}\n'.format(img_name, temp_name)
                f1.write(content)


    if len(alignment_files) == 0:
        # 全在一个文件夹下
        get_info_file_core(alignment_root)
    else:
        # 存在ID目录
        ids = os.listdir(alignment_root)
        for id in ids:
            get_info_file_core(os.path.join(alignment_root, id), id)

@calc_execute_time(show_time=True)
def controller():
    face_img_root = '/mnt/e/mnt/datasets/lfw/lfw'
    face_det_root = '/mnt/e/mnt/datasets/lfw/lfw_det'
    face_ali_root = '/mnt/e/mnt/datasets/lfw/lfw_ali'
    face_info_file = '/mnt/e/mnt/datasets/lfw/face_info.txt'
    mask_temp_file = '/mnt/e/mnt/datasets/lfw/mask_temp.txt'
    q = Queue()
    detProcess = Process(target=face_detection, args=((face_img_root, face_det_root, face_ali_root), q))
    aliProcess = Process(target=face_alignment, args=(q, ))
    aliProcess.start()
    detProcess.start()
    detProcess.join()
    aliProcess.join()
    get_info_file(face_ali_root, face_info_file, mask_temp_file)



if __name__ == "__main__":
    # controller() 
    get_info_file('/mnt/e/mnt/datasets/lfw/lfw_ali', '/mnt/e/mnt/datasets/lfw/face_info.txt', '/mnt/e/mnt/datasets/lfw/mask_temp.txt')
