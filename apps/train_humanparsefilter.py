import sys
import os
import json

# 상위 디렉토리에서 모듈을 가져올 수 있도록 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle

from PIL import Image

from lib.options import BaseOptions
from lib.model import HumanParseFilter
from lib.data import HumanParseDataset

# 시드 설정: 재현성을 위해 난수 시드 고정
seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 옵션 파서 초기화 및 설정
parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

# 학습률, 스케줄, 에포크 수, 배치 크기 등 설정
lr = 1e-3 
parse_schedule = [50]  # 특정 에포크에서 학습률 감소
num_of_epoch = 70
batch_size = 4
num_classes = 7  # 신체 부위 카테고리 수
load_model = False  # 모델 로드 여부 설정

def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """에포크에 따라 학습률을 조정"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def get_palette(num_cls):
    """세그멘테이션 마스크 시각화를 위한 색상 맵 반환
    Args:
        num_cls: 클래스 수
    Returns:
        색상 맵 리스트
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def train(opt):
    global gen_test_counter
    global lr 

    # GPU 사용 여부 확인 후 설정
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device))
    
    # 학습 데이터셋 초기화
    train_dataset = HumanParseDataset(opt, evaluation_mode=False)

    # 데이터 로더 설정 (배치 사이즈, 셔플, 스레드 수, 핀 메모리 사용)
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train loader size: ', len(train_data_loader))

    # HumanParseFilter 모델 초기화
    humanParsefilter = HumanParseFilter(opt)

    # 체크포인트 및 결과 디렉토리 생성
    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)
    if not os.path.exists(f'{opt.checkpoints_path}/{opt.name}'):
        os.makedirs(f'{opt.checkpoints_path}/{opt.name}')
    if not os.path.exists(f'{opt.results_path}/{opt.name}'):
        os.makedirs(f'{opt.results_path}/{opt.name}')

    # 세그멘테이션 마스크의 색상 팔레트 설정
    palette = get_palette(num_classes)
     
    # 기존 모델 로드 (필요한 경우)
    if load_model: 
        modelparsefilter_path = "apps/checkpoints/Date_06_Jan_22_Time_00_49_24/humanParsefilter_model_state_dict.pickle"
        print('Resuming from ', modelparsefilter_path)
        with open(modelparsefilter_path, 'rb') as handle:
           net_state_dict = pickle.load(handle)
        humanParsefilter.load_state_dict(net_state_dict, strict=True)

    # 옵션 설정을 로그로 저장
    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # 모델을 선택된 장치(GPU 또는 CPU)로 이동
    humanParsefilter = humanParsefilter.to(device=device)
 
    # 옵티마이저 설정
    optimizer = torch.optim.RMSprop(humanParsefilter.parameters(), lr=lr, momentum=0, weight_decay=0)

    # 학습 루프
    start_epoch = 0
    for epoch in range(start_epoch, num_of_epoch):
        print("start of epoch {}".format(epoch))
        humanParsefilter.train()

        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx))

            # 학습 데이터 로드
            render_tensor = train_data['original_high_res_render'].to(device=device)
            human_parse_map_high_res_tensor = train_data['human_parse_map_high_res'].to(device=device)
            
            # 법선 맵 사용 시 텐서 병합
            if opt.use_normal_map_for_parse_training:
                nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                render_tensor = torch.cat([render_tensor, nmlF_high_res_tensor], dim=1)

            # 손실 계산
            error = humanParsefilter.forward(images=render_tensor, groundtruth_parsemap=human_parse_map_high_res_tensor)
            optimizer.zero_grad()
            error['Err'].backward()
            curr_loss = error['Err'].item()
            optimizer.step()

            # 현재 손실 값 출력
            print(
            'Name: {0} | Epoch: {1} | error: {2:.06f} | LR: {3:.06f} '.format(
                opt.name, epoch, curr_loss, lr)
            )

        # 학습률 조정
        lr = adjust_learning_rate(optimizer, epoch, lr, schedule=parse_schedule, learning_rate_decay=0.05)

        # 평가 및 결과 저장
        with torch.no_grad():
            if True:
                # 모델 저장
                with open(f'{opt.checkpoints_path}/{opt.name}/humanParsefilter_model_state_dict.pickle', 'wb') as handle:
                    pickle.dump(humanParsefilter.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f'{opt.checkpoints_path}/{opt.name}/optimizer.pickle', 'wb') as handle:
                    pickle.dump(optimizer.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                # 테스트 평가
                print('generate parse map (train) ...')
                train_dataset.is_train = False
                humanParsefilter.eval()
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    index_to_use = gen_test_counter % len(train_dataset)
                    gen_test_counter += 10
                    train_data = train_dataset.get_item(index=index_to_use)
                    save_path = f'{opt.results_path}/{opt.name}/train_eval_epoch{epoch}_{train_data["name"]}.obj'

                    image_tensor = train_data['original_high_res_render'].to(device=device)
                    image_tensor = torch.unsqueeze(image_tensor, 0)

                    if opt.use_normal_map_for_parse_training:
                        nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                        nmlF_high_res_tensor = torch.unsqueeze(nmlF_high_res_tensor, 0)
                        image_tensor = torch.cat([image_tensor, nmlF_high_res_tensor], dim=1)

                    original_parse_map = train_data['human_parse_map_high_res']
                    original_parse_map = torch.argmax(original_parse_map, dim=0).cpu().numpy()

                    # 세그멘테이션 맵 생성
                    humanParsefilter.filter(image_tensor)
                    generated_parse_map = humanParsefilter.generate_parse_map()
                    generated_parse_map = generated_parse_map.detach().cpu().numpy()[0, :, :]

                    save_img_path = save_path[:-4] + '.png'
                    save_parsemap_path = save_path[:-4] + 'generated_parsemap.png'

                    save_img = (np.transpose(image_tensor[0, :3, ...].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_img = save_img.astype(np.uint8)
                    save_img = Image.fromarray(save_img)
                    save_img.save(save_img_path)

                    save_parsemap = Image.fromarray(np.asarray(generated_parse_map, dtype=np.uint8))
                    save_parsemap.putpalette(palette)
                    save_parsemap.save(save_parsemap_path)

                    original_parse_map = original_parse_map.astype(np.uint8)
                    original_parse_map = Image.fromarray(original_parse_map)
                    original_parse_map.putpalette(palette)
                    original_parse_map_save_path = f'{opt.results_path}/{opt.name}/train_eval_epoch{epoch}_{train_data["name"]}_groundtruth.png'
                    original_parse_map.save(original_parse_map_save_path)

                train_dataset.is_train = True

if __name__ == '__main__':
    train(opt)