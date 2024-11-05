import sys
import os
import json

# 상위 디렉토리의 모듈을 불러올 수 있도록 시스템 경로 수정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
from PIL import Image

# 옵션, 모델, 데이터셋 불러오기
from lib.options import BaseOptions
from lib.model import RelativeDepthFilter
from lib.data import DepthDataset

# 옵션 파싱 및 전역 변수 설정
parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

# 특정 설정에 따른 플래그
generate_for_buff_dataset = False 
generate_refined_trained_depth_maps = True # True일 경우, 정제된 depth maps 생성, False일 경우 coarse depth maps 생성
batch_size = 2

# 경로 설정
if generate_refined_trained_depth_maps:
    trained_depth_maps_path = "trained_depth_maps"
else:
    trained_depth_maps_path = "trained_coarse_depth_maps"

# 특정 데이터셋의 경우 경로를 변경
if generate_for_buff_dataset:
    print("Overwriting trained_depth_maps_path for Buff dataset")
    trained_depth_maps_path = "buff_dataset/buff_depth_maps"

def generate_maps(opt):
    global gen_test_counter

    # 디바이스 설정 (GPU 사용 가능 시 CUDA 사용)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device))
    
    # BuffDataset을 사용할 경우 설정
    if generate_for_buff_dataset:
        from lib.data.BuffDataset import BuffDataset
        train_dataset = BuffDataset(opt)
        train_dataset.is_train = False
        train_data_loader = DataLoader(train_dataset, 
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=opt.num_threads, pin_memory=opt.pin_memory)

        print('train loader size: ', len(train_data_loader))
        data_loader_list = [ ('first_data_loader', train_data_loader) ]
    else:
        # 일반 DepthDataset 사용
        train_dataset = DepthDataset(opt, evaluation_mode=False)
        train_dataset.is_train = False

        test_dataset = DepthDataset(opt, evaluation_mode=True)
        test_dataset.is_train = False

        # 훈련 및 테스트 데이터 로더 설정
        train_data_loader = DataLoader(train_dataset, 
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=opt.num_threads, pin_memory=opt.pin_memory)

        test_data_loader = DataLoader(test_dataset, 
                                      batch_size=batch_size, shuffle=False,
                                      num_workers=opt.num_threads, pin_memory=opt.pin_memory)

        print('train loader size: ', len(train_data_loader))
        print('test loader size: ', len(test_data_loader))

        data_loader_list = [ ('first_data_loader', train_data_loader), ('second_data_loader', test_data_loader) ]
    
    # RelativeDepthFilter 모델 초기화
    depthfilter = RelativeDepthFilter(opt)

    # 모델 로드
    if generate_refined_trained_depth_maps:
        modeldepthfilter_path = "/home/public/IntegratedPIFu/checkpoints/first_fine_depth_test/depthfilter_model_state_dict.pickle"
    else:
        modeldepthfilter_path = "/home/public/IntegratedPIFu/checkpoints/first_depth_test/depthfilter_model_state_dict.pickle"

    print('Resuming from ', modeldepthfilter_path)

    # CPU에서 로드할 때는 특별한 처리
    if device == 'cpu':
        import io

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(modeldepthfilter_path, 'rb') as handle:
            net_state_dict = CPU_Unpickler(handle).load()
    else:
        with open(modeldepthfilter_path, 'rb') as handle:
            net_state_dict = pickle.load(handle)

    # 모델 상태 로드 및 디바이스 설정
    depthfilter.load_state_dict(net_state_dict, strict=True)
    depthfilter = depthfilter.to(device=device)
    depthfilter.eval()

    # 데이터 로더에서 데이터 가져오기
    with torch.no_grad():
        for description, data_loader in data_loader_list:
            print('description: {0}'.format(description))
            for idx, batch_data in enumerate(data_loader):
                print("batch {}".format(idx))

                # 데이터 추출
                subject_list = batch_data['name']
                render_filename_list = batch_data['render_path']

                render_tensor = batch_data['original_high_res_render'].to(device=device)

                # 정제된 depth maps 생성 시 coarse depth map과 병합
                if generate_refined_trained_depth_maps:
                    coarse_depth_map_tensor = batch_data['coarse_depth_map'].to(device=device)
                    render_tensor = torch.cat([render_tensor, coarse_depth_map_tensor], dim=1)

                    if opt.use_normal_map_for_depth_training:
                        nmlF_high_res_tensor = batch_data['nmlF_high_res'].to(device=device)
                        render_tensor = torch.cat([render_tensor, nmlF_high_res_tensor], dim=1)
                else:
                    center_indicator_tensor = batch_data['center_indicator'].to(device=device)
                    render_tensor = torch.cat([render_tensor, center_indicator_tensor], dim=1)

                    if opt.use_normal_map_for_depth_training:
                        nmlF_high_res_tensor = batch_data['nmlF_high_res'].to(device=device)
                        render_tensor = torch.cat([render_tensor, nmlF_high_res_tensor], dim=1)

                # 모델을 사용해 depth maps 생성
                depthfilter.filter(render_tensor)
                generated_depth_maps = depthfilter.generate_depth_map().detach().cpu().numpy()

                # 생성된 depth map 저장
                for i in range(batch_size):
                    try:
                        subject = subject_list[i]
                        render_filename = render_filename_list[i]
                    except:
                        print('Last batch of data_loader reached!')
                        break

                    # 저장 경로 설정
                    if generate_for_buff_dataset:
                        if generate_refined_trained_depth_maps:
                            save_depthmap_path = os.path.join(trained_depth_maps_path, "rendered_depthmap_" + subject + ".npy")
                            save_depthmap_image_path = os.path.join(trained_depth_maps_path, "rendered_depthmap_" + subject + ".png")
                        else:
                            save_depthmap_path = os.path.join(trained_depth_maps_path, "rendered_coarse_depthmap_" + subject + ".npy")
                            save_depthmap_image_path = os.path.join(trained_depth_maps_path, "rendered_coarse_depthmap_" + subject + ".png")
                    else:
                        if not os.path.exists(os.path.join(trained_depth_maps_path, subject)):
                            os.makedirs(os.path.join(trained_depth_maps_path, subject))
                        yaw = render_filename.split('_')[-1].split('.')[0]
                        save_depthmap_path = os.path.join(trained_depth_maps_path, subject, "rendered_depthmap_" + yaw + ".npy")
                        save_depthmap_image_path = os.path.join(trained_depth_maps_path, subject, "rendered_depthmap_" + yaw + ".png")

                    # numpy 배열로 저장
                    generated_map = generated_depth_maps[i]
                    np.save(save_depthmap_path, generated_map)

                    # 이미지로 저장
                    save_depthmap_image = (np.transpose(generated_map, (1, 2, 0))) * 255.0 / 2
                    save_depthmap_image = save_depthmap_image.astype(np.uint8)[:, :, 0]
                    save_depthmap_image = Image.fromarray(save_depthmap_image, 'L')
                    save_depthmap_image.save(save_depthmap_image_path)

# 메인 함수 실행
if __name__ == '__main__':
    generate_maps(opt)