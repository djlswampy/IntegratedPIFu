# 필요한 모듈과 패키지 임포트
import sys
import os
import json

# 상위 디렉토리를 시스템 경로에 추가하여 다른 모듈을 임포트 가능하도록 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from PIL import Image

# 커스텀 모듈 임포트
from lib.options import BaseOptions
from lib.model import RelativeDepthFilter
from lib.data import DepthDataset

# 시드 설정: 모든 난수 생성에 동일한 시드를 사용하여 재현 가능성을 확보
seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 옵션 파서 생성
parser = BaseOptions()
opt = parser.parse()  # 옵션 파서에서 명령줄 인자 분석
gen_test_counter = 0  # 테스트 샘플 생성 카운터 초기화

# 학습 관련 하이퍼파라미터 설정
lr = 1e-3  # 초기 학습률
depth_schedule = [50]  # 특정 에포크에서 학습률 감소
num_of_epoch = 2  # 전체 학습 에포크 수
batch_size = 2  # 배치 크기
load_model = False  # 학습된 모델을 로드할지 여부 (True로 설정 시 modeldepthfilter_path 수정 필요)

# 두 번째 스테이지의 경우 학습률 변경
if opt.second_stage_depth:
    print("Changing lr!")
    lr = 5e-5  

# 학습률 조정 함수
def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """지정된 에포크에서 학습률을 감소시키는 함수"""
    if epoch in schedule:
        lr *= learning_rate_decay  # 학습률을 감소
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 모든 파라미터 그룹에 대해 학습률 업데이트
    return lr

# 학습 메인 함수 정의
def train(opt):
    global gen_test_counter
    global lr 

    # GPU 사용 가능 여부 확인 후 디바이스 설정
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device))

    # 학습 데이터셋 로드
    train_dataset = DepthDataset(opt, evaluation_mode=False)
    
    # 데이터 로더 설정 (배치 사이즈 및 다중 스레드 사용)
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train loader size: ', len(train_data_loader))

    # 모델 생성
    depthfilter = RelativeDepthFilter(opt)

    # 체크포인트 및 결과 저장 경로가 존재하지 않으면 생성
    if (not os.path.exists(opt.checkpoints_path)):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path)):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name))):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))

    # 이전에 학습된 모델 로드
    if load_model:
        modeldepthfilter_path = "apps/checkpoints/Date_08_Jan_22_Time_02_03_43/depthfilter_model_state_dict.pickle"
        print('Resuming from ', modeldepthfilter_path)

        with open(modeldepthfilter_path, 'rb') as handle:
           net_state_dict = pickle.load(handle)

        depthfilter.load_state_dict(net_state_dict, strict=True)

    # 옵션 로그 파일 생성
    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # 모델을 지정된 디바이스로 이동
    depthfilter = depthfilter.to(device=device)
    
    # 옵티마이저 설정 (RMSprop 사용)
    optimizer = torch.optim.RMSprop(depthfilter.parameters(), lr=lr, momentum=0, weight_decay=0)

    # 학습 시작 에포크
    start_epoch = 0
    for epoch in range(start_epoch, num_of_epoch):
        print("start of epoch {}".format(epoch))
        depthfilter.train()  # 모델을 학습 모드로 설정

        # 배치 단위로 데이터 학습
        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx))

            # 데이터 로드
            if opt.second_stage_depth:
                render_tensor = train_data['original_high_res_render'].to(device=device)
                coarse_depth_map_tensor = train_data['coarse_depth_map'].to(device=device)
                depth_map_tensor = train_data['depth_map'].to(device=device)
                render_tensor = torch.cat([render_tensor, coarse_depth_map_tensor], dim=1)

                if opt.use_normal_map_for_depth_training:
                    nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                    render_tensor = torch.cat([render_tensor, nmlF_high_res_tensor], dim=1)
            else:
                render_tensor = train_data['original_high_res_render'].to(device=device)
                depth_map_tensor = train_data['depth_map'].to(device=device)
                center_indicator_tensor = train_data['center_indicator'].to(device=device)
                render_tensor = torch.cat([render_tensor, center_indicator_tensor], dim=1)

                if opt.use_normal_map_for_depth_training:
                    nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                    render_tensor = torch.cat([render_tensor, nmlF_high_res_tensor], dim=1)

            # 모델을 통해 오류 계산
            error = depthfilter.forward(images=render_tensor, groundtruth_depthmap=depth_map_tensor)
            optimizer.zero_grad()  # 옵티마이저 초기화
            error['Err'].backward()  # 오류에 대한 역전파 수행
            curr_loss = error['Err'].item()  # 현재 배치의 손실 값 저장

            optimizer.step()  # 모델 파라미터 업데이트

            print('Name: {0} | Epoch: {1} | error: {2:.06f} | LR: {3:.06f} '.format(
                opt.name, epoch, curr_loss, lr)
            )

        # 학습률 조정
        lr = adjust_learning_rate(optimizer, epoch, lr, schedule=depth_schedule, learning_rate_decay=0.05)

        # 모델 평가 및 결과 저장
        with torch.no_grad():
            if True:
                with open('%s/%s/depthfilter_model_state_dict.pickle' % (opt.checkpoints_path, opt.name), 'wb') as handle:
                    pickle.dump(depthfilter.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                print('generate depth map (train) ...')
                train_dataset.is_train = False
                depthfilter.eval()  # 모델을 평가 모드로 설정

                # 테스트 샘플 생성
                for gen_idx in tqdm(range(1)):
                    index_to_use = gen_test_counter % len(train_dataset)
                    gen_test_counter += 10
                    train_data = train_dataset.get_item(index=index_to_use)
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])

                    if opt.second_stage_depth:
                        image_tensor = train_data['original_high_res_render'].to(device=device)
                        coarse_depth_map_tensor = train_data['coarse_depth_map'].to(device=device)
                        image_tensor = torch.unsqueeze(image_tensor, 0)
                        coarse_depth_map_tensor = torch.unsqueeze(coarse_depth_map_tensor, 0)
                        original_depth_map = train_data['depth_map'].cpu().numpy()
                        image_tensor = torch.cat([image_tensor, coarse_depth_map_tensor], dim=1)

                        if opt.use_normal_map_for_depth_training:
                            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                            nmlF_high_res_tensor = torch.unsqueeze(nmlF_high_res_tensor, 0)
                            image_tensor = torch.cat([image_tensor, nmlF_high_res_tensor], dim=1)

                    else:
                        image_tensor = train_data['original_high_res_render'].to(device=device)
                        image_tensor = torch.unsqueeze(image_tensor, 0)
                        center_indicator_tensor = train_data['center_indicator'].to(device=device)
                        center_indicator_tensor = torch.unsqueeze(center_indicator_tensor, 0)
                        image_tensor = torch.cat([image_tensor, center_indicator_tensor], dim=1)

                        if opt.use_normal_map_for_depth_training:
                            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                            nmlF_high_res_tensor = torch.unsqueeze(nmlF_high_res_tensor, 0)
                            image_tensor = torch.cat([image_tensor, nmlF_high_res_tensor], dim=1)

                        original_depth_map = train_data['depth_map'].cpu().numpy()

                    depthfilter.filter(image_tensor)
                    generated_depth_map = depthfilter.generate_depth_map()
                    generated_depth_map = generated_depth_map.detach().cpu().numpy()[0, 0:1, :, :]

                    if opt.second_stage_depth:
                        save_differencemap_path = save_path[:-4] + 'generated_differencemap.png'
                        save_differencemap = generated_depth_map - coarse_depth_map_tensor.detach().cpu().numpy()[0, 0:1, :, :]
                        save_differencemap = (np.transpose(save_differencemap, (1, 2, 0))) * 255.0 / 4 + 255.0 / 2
                        save_differencemap = save_differencemap.astype(np.uint8)[:, :, 0]
                        save_differencemap = Image.fromarray(save_differencemap, 'L')
                        save_differencemap.save(save_differencemap_path)

                    save_img_path = save_path[:-4] + '.png'
                    save_depthmap_path = save_path[:-4] + 'generated_depthmap.png'

                    save_img = (np.transpose(image_tensor[0, :3, ...].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_img = save_img.astype(np.uint8)
                    save_img = Image.fromarray(save_img)
                    save_img.save(save_img_path)

                    save_depthmap = (np.transpose(generated_depth_map, (1, 2, 0))) * 255.0 / 2
                    save_depthmap = save_depthmap.astype(np.uint8)[:, :, 0]
                    save_depthmap = Image.fromarray(save_depthmap, 'L')
                    save_depthmap.save(save_depthmap_path)

                    original_depth_map = (original_depth_map[0, :, :]) * 255.0 / 2
                    original_depth_map = original_depth_map.astype(np.uint8)
                    original_depth_map = Image.fromarray(original_depth_map, 'L')
                    original_depth_map_save_path = '%s/%s/train_eval_epoch%d_%s_groundtruth.png' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    original_depth_map.save(original_depth_map_save_path)

                train_dataset.is_train = True

# 메인 함수 호출
if __name__ == '__main__':
    train(opt)