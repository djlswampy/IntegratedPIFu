import sys
import os
import json

# 시스템 경로에 상위 디렉토리를 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# OpenCV에서 EXR 파일을 읽을 수 있도록 환경 변수 설정
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle

from PIL import Image

from lib.options import BaseOptions
from lib.networks import define_G
from lib.data.NormalDataset import NormalDataset

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

# 옵션 파서 초기화
parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

# 버프 데이터셋을 사용할지 여부
generate_for_buff_dataset = False 

# 노멀 맵 저장 경로 설정
if generate_for_buff_dataset:
    trained_normal_maps_path = "/home/jo/IntegratedPIFu/dataset/normal_maps"
else:
    trained_normal_maps_path = "pretrained_normal_maps"

# 배치 크기 설정
batch_size = 2

def generate_maps(opt):
    global gen_test_counter
    global lr 

    # CUDA 사용 가능 여부에 따라 디바이스 설정
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device))
    
    # 데이터셋 로드 및 데이터 로더 설정
    if generate_for_buff_dataset:
        # 'lib.data.BuffDataset'에서 BuffDataset 클래스를 가져옴
        from lib.data.BuffDataset import BuffDataset
        # BuffDataset 클래스의 인스턴스를 생성하고, 옵션(opt)을 전달하여 초기화
        train_dataset = BuffDataset(opt) 
        train_dataset.is_train = False

        # DataLoader를 생성. DataLoader는 데이터셋을 배치 단위로 로드하고,
        # 여기서는 train_dataset을 기준으로 배치 크기(batch_size)를 설정.
        # shuffle=False는 데이터셋을 순서대로 로드하겠다는 의미.
        # num_workers는 데이터를 로드할 때 사용할 CPU 스레드 수를 설정하며, 
        # pin_memory는 텐서들이 고정된 메모리(Pinned memory)에 할당될지 여부를 설정.
        train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

        # 생성된 DataLoader의 길이를 출력 (DataLoader에 있는 전체 배치 수)
        print('train loader size: ', len(train_data_loader))

        # 데이터 로더 목록을 생성, 첫 번째 데이터 로더(train_data_loader)를 'first_data_loader'라는 이름으로 등록
        data_loader_list = [ ('first_data_loader', train_data_loader) ]

    else:
        train_dataset = NormalDataset(opt, evaluation_mode=False)
        train_dataset.is_train = False
        test_dataset = NormalDataset(opt, evaluation_mode=True)
        test_dataset.is_train = False

        train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

        test_data_loader = DataLoader(test_dataset, 
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
        
        print('train loader size: ', len(train_data_loader))
        print('test loader size: ', len(test_data_loader))

        data_loader_list = [ ('first_data_loader', train_data_loader), ('second_data_loader', test_data_loader) ]
    
    # 네트워크 정의
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
    netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

    # 모델 로드 경로 설정
    F_modelnormal_path = "/home/public/integratedpifu_checkpoint/netF_model_state_dict.pickle"
    B_modelnormal_path = "/home/public/integratedpifu_checkpoint/netB_model_state_dict.pickle"

    print('Resuming from ', F_modelnormal_path)
    print('Resuming from ', B_modelnormal_path)

    # 모델 상태 로드
    if device == 'cpu':
        import io

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = CPU_Unpickler(handle).load()

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = CPU_Unpickler(handle).load()

    else:
        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = pickle.load(handle)

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = pickle.load(handle)

    netF.load_state_dict(netF_state_dict, strict=True)
    netB.load_state_dict(netB_state_dict, strict=True)
        
    # 모델을 디바이스에 할당
    netF = netF.to(device=device)
    netB = netB.to(device=device)

    # 평가 모드로 설정
    netF.eval()
    netB.eval()

    # 데이터 로더를 통해 데이터 처리
    with torch.no_grad():
        # 튜플을 반복하는 for 루프 data_loader_list는 튜플들이 들어있는 리스트고 각 튜플은 (description, data_loader) 형태로 되어있음
        for description, data_loader in data_loader_list:
            print('description: {0}'.format(description))
            # enumerate() 함수는 반복 가능한 객체(리스트, 튜플 등)를 반복할 때, 인덱스와 값을 동시에 반환해주는 함수. idx, batch_data를 동시에 받기 위해 사용
            for idx, batch_data in enumerate(data_loader):
                print("batch {}".format(idx))

                # 데이터 가져오기
                subject_list = batch_data['name']
                render_filename_list = batch_data['render_path']  
                render_tensor = batch_data['original_high_res_render'].to(device=device)

                # 네트워크를 통해 결과 생성
                res_netF = netF.forward(render_tensor)
                res_netB = netB.forward(render_tensor)

                res_netF = res_netF.detach().cpu().numpy()
                res_netB = res_netB.detach().cpu().numpy()

                for i in range(batch_size):
                    try:
                        subject = subject_list[i]
                        render_filename = render_filename_list[i]
                    except:
                        print('Last batch of data_loader reached!')
                        break

                    # 결과 저장 경로 설정
                    if generate_for_buff_dataset:
                        save_normalmapF_path = os.path.join(trained_normal_maps_path, "rendered_nmlF_" +  subject + ".npy")
                        save_normalmapB_path = os.path.join(trained_normal_maps_path, "rendered_nmlB_" +  subject + ".npy")

                        save_netF_normalmap_path = os.path.join(trained_normal_maps_path, "rendered_nmlF_" +  subject + ".png")
                        save_netB_normalmap_path = os.path.join(trained_normal_maps_path, "rendered_nmlB_" +  subject + ".png")

                    else:
                        if not os.path.exists(os.path.join(trained_normal_maps_path, subject)):
                            os.makedirs(os.path.join(trained_normal_maps_path, subject))

                        yaw = render_filename.split('_')[-1].split('.')[0]
                        save_normalmapF_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlF_" + yaw + ".npy")
                        save_normalmapB_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlB_" + yaw + ".npy")

                        save_netF_normalmap_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlF_" + yaw + ".png")
                        save_netB_normalmap_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlB_" + yaw + ".png")

                    # 결과를 파일로 저장
                    generated_map = res_netF[i]
                    np.save(save_normalmapF_path, generated_map)

                    generated_map = res_netB[i]
                    np.save(save_normalmapB_path, generated_map)

                    # 이미지를 저장
                    save_netF_normalmap = (np.transpose(res_netF[i], (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netF_normalmap = save_netF_normalmap.astype(np.uint8)
                    save_netF_normalmap = Image.fromarray(save_netF_normalmap)
                    save_netF_normalmap.save(save_netF_normalmap_path)

                    save_netB_normalmap = (np.transpose(res_netB[i], (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netB_normalmap = save_netB_normalmap.astype(np.uint8)
                    save_netB_normalmap = Image.fromarray(save_netB_normalmap)
                    save_netB_normalmap.save(save_netB_normalmap_path)

# 메인 함수 실행
if __name__ == '__main__':
    generate_maps(opt)