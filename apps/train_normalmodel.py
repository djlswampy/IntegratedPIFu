import sys
import os
import json
from datetime import datetime  # datetime 모듈 추가
import time  # time 모듈도 추가 (epoch_time, total_time 계산에 필요)


# 상위 디렉토리를 경로에 추가하고, 환경 변수를 설정하여 OpenEXR 파일을 사용할 수 있도록 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random 
import numpy as np
import cv2
import pickle

from PIL import Image

# 사용자 정의 모듈 임포트
from lib.options import BaseOptions
from lib.networks import define_G
from lib.data.NormalDataset import NormalDataset

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

# 랜덤 시드 설정 (재현 가능성을 위해)
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 명령줄 옵션 파싱
parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

# 학습 설정
lr = 2e-4  # 학습률
normal_schedule = [20] # 특정 에포크에서 학습률 감소
# batch_size = 1
batch_size = 2
load_model = False  # 모델을 로드할지 여부 (True로 설정 시, 모델 경로도 설정해야 함)
# TODO: 에포크 변수 추가해서 에포크 받도록 수정

# 학습 기록을 위한 딕셔너리 초기화
training_history = {
    'epochs': [],
    'train_losses': [],
    'val_losses': [],
    'learning_rate': [],
    'best_val_loss': float('inf'),
    'time_per_epoch': [],
    'total_time': 0,
    'best_epoch': -1,
    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'last_update': "",
}

def evaluate(netF, data_loader, device, smoothL1Loss):
    """검증 데이터셋에 대한 모델 평가 함수"""
    netF.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for val_data in data_loader:
            render_tensor = val_data['original_high_res_render'].to(device=device)
            nmlF_high_res_tensor = val_data['nmlF_high_res'].to(device=device)
            
            res_netF = netF.forward(render_tensor)
            loss = smoothL1Loss(res_netF, nmlF_high_res_tensor)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def save_evaluation_results(netF, dataset, opt, epoch, mode, device, start_idx=0, num_samples=5):
    """학습/검증 데이터에 대한 노멀맵 생성 및 저장"""
    netF.eval()
    with torch.no_grad():
        for idx in range(start_idx, start_idx + num_samples):
            index_to_use = idx % len(dataset)
            data = dataset.get_item(index=index_to_use)
            
            # 결과 저장 경로 설정
            save_path = '%s/%s/%s_eval_epoch%d_%s.obj' % (
                opt.results_path, opt.name, mode, epoch, data['name'])
            
            # 데이터 준비 및 예측
            image_tensor = data['original_high_res_render'].to(device=device)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            original_nmlF_map = data['nmlF_high_res'].cpu().numpy()
            
            res_netF = netF.forward(image_tensor)
            res_netF = res_netF.detach().cpu().numpy()[0, :, :, :]
            
            # 결과 저장
            save_netF_normalmap_path = save_path[:-4] + 'netF_normalmap.png'
            numpy_save_netF_normalmap_path = save_path[:-4] + 'netF_normalmap.npy'
            GT_netF_normalmap_path = save_path[:-4] + 'netF_groundtruth.png'
            
            # 예측 결과 저장
            np.save(numpy_save_netF_normalmap_path, res_netF)
            save_netF_normalmap = (np.transpose(res_netF, (1, 2, 0)) * 0.5 + 0.5) * 255.0
            save_netF_normalmap = save_netF_normalmap.astype(np.uint8)
            save_netF_normalmap = Image.fromarray(save_netF_normalmap)
            save_netF_normalmap.save(save_netF_normalmap_path)
            
            # Ground truth 저장
            GT_netF_normalmap = (np.transpose(original_nmlF_map, (1, 2, 0)) * 0.5 + 0.5) * 255.0
            GT_netF_normalmap = GT_netF_normalmap.astype(np.uint8)
            GT_netF_normalmap = Image.fromarray(GT_netF_normalmap)
            GT_netF_normalmap.save(GT_netF_normalmap_path)



"""에포크가 일정 시점에 도달했을 때 학습률을 감소시키는 함수"""
def adjust_learning_rate(optimizer_list, epoch, lr, schedule, learning_rate_decay):
    # optimizer_list가 리스트가 아닌 경우 리스트로 변환
    if not isinstance(optimizer_list, list):
        optimizer_list = [optimizer_list]
        
    for optimizer in optimizer_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

# 학습 함수 정의
def train(opt):
    global gen_test_counter
    global lr

    # GPU 사용 가능 여부 확인 후 디바이스 설정
    if torch.cuda.is_available():
        device = 'cuda:0'  # GPU 사용
    else:
        device = 'cpu'  # CPU 사용

    print("using device {}".format(device))
    
    # 훈련 데이터셋 초기화
    train_dataset = NormalDataset(opt, evaluation_mode=False, validation_mode=False)

    # 검증 데이터셋 초기화
    val_dataset = NormalDataset(opt, evaluation_mode=False, validation_mode=True)


    
    # 훈련 데이터 데이터 로더 초기화
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    # 검증 데이터 데이터 로더 초기화
    val_data_loader = DataLoader(val_dataset, 
                                    batch_size=batch_size, shuffle=not opt.serial_batches,
                                    num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train loader size: ', len(train_data_loader))
    print('train loader size: ', len(val_data_loader))


    # 손실 함수로 Smooth L1 Loss 사용
    smoothL1Loss = nn.SmoothL1Loss()
    
    # 네트워크 정의 (Generator)
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

    # 모델 가중치 로드 (필요 시)
    if load_model: 
        F_modelnormal_path = "apps/checkpoints/Date_12_Nov_21_Time_01_38_54/netF_model_state_dict.pickle"

        print('Resuming from ', F_modelnormal_path)
        print('Resuming from ', B_modelnormal_path)

        with open(F_modelnormal_path, 'rb') as handle:
            netF_state_dict = pickle.load(handle)


        netF.load_state_dict(netF_state_dict, strict=True)
        
    # 경로가 존재하지 않으면 체크포인트와 결과 저장 경로 생성
    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)
    if not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name)):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if not os.path.exists('%s/%s' % (opt.results_path, opt.name)):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))

    # 옵션 로그 저장
    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # 네트워크를 디바이스에 로드
    netF = netF.to(device=device)

    # 옵티마이저 초기화
    optimizer_netF = torch.optim.RMSprop(netF.parameters(), lr=lr, momentum=0, weight_decay=0)

    # 학습 시작
    start_epoch = 0
    total_start_time = time.time()

    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()
        print(f"Start of epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 훈련 모드
        netF.train()

        epoch_train_loss = 0.0
        num_batches = 0

        train_len = len(train_data_loader)
        # train_idx는 현재 처리중인 배치의 인덱스를 의미
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx))

            # 데이터를 GPU 또는 CPU로 전송
            render_tensor = train_data['original_high_res_render'].to(device=device)
            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
            nmlB_high_res_tensor = train_data['nmlB_high_res'].to(device=device)

            # 네트워크의 순전파
            res_netF = netF.forward(render_tensor)

            # 손실 계산 (Smooth L1 Loss)
            err_netF = smoothL1Loss(res_netF, nmlF_high_res_tensor) 
   
            # 옵티마이저를 통한 네트워크 학습
            optimizer_netF.zero_grad()
            err_netF.backward()
            curr_loss_netF = err_netF.item()
            optimizer_netF.step()

            # 한 에포크 동안 출력된 모든 손실값 더하기
            epoch_train_loss += curr_loss_netF
            # 한 에포크 동안 처리된 총 배치의 수를 카운트
            num_batches += 1

            if train_idx % 5 == 0:  # 로그 출력 간격 조절
                print(
                    'Name: {0} | Epoch: {1} | Batch: {2}/{3} | Loss: {4:.6f} | LR: {5:.6f}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader),
                        curr_loss_netF, lr)
                )


        # 에포크당 평균 훈련 손실 계산
        avg_train_loss = epoch_train_loss / num_batches
        
        # 검증 데이터로 평가
        avg_val_loss = evaluate(netF, val_data_loader, device, smoothL1Loss)

        # 에포크 수행 시간 계산
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        
  

        # 학습 기록 업데이트
        training_history['epochs'].append(epoch)
        training_history['train_losses'].append(float(avg_train_loss))  # numpy float를 native float로 변환
        training_history['val_losses'].append(float(avg_val_loss))
        training_history['learning_rate'].append(float(lr))
        training_history['time_per_epoch'].append(float(epoch_time))
        training_history['total_time'] = float(total_time)
        training_history['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
                
        # 최고 성능 모델 체크 및 저장
        if avg_val_loss < training_history['best_val_loss']:
            training_history['best_val_loss'] = float(avg_val_loss)
            training_history['best_epoch'] = epoch
            
            # 최고 성능 모델 저장
            with open('%s/%s/netF_best_model.pickle' % (opt.checkpoints_path, opt.name), 'wb') as handle:
                pickle.dump(netF.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        # 현재 모델 상태 저장
        with open('%s/%s/netF_model_state_dict.pickle' % (opt.checkpoints_path, opt.name), 'wb') as handle:
            pickle.dump(netF.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 학습 기록 저장
        with open('%s/%s/training_history.json' % (opt.results_path, opt.name), 'w') as f:
            json.dump(training_history, f)

        print(f"""
Epoch {epoch} Summary:
- Training Loss: {avg_train_loss:.6f}
- Validation Loss: {avg_val_loss:.6f}
- Learning Rate: {lr:.6f}
- Epoch Time: {epoch_time:.2f}s
- Total Time: {total_time:.2f}s
- Best Val Loss: {training_history['best_val_loss']:.6f} (Epoch {training_history['best_epoch']})
""")


        # 학습률 조정
        lr = adjust_learning_rate(optimizer_netF, epoch, lr, schedule=normal_schedule, learning_rate_decay=0.1)

                # 샘플 이미지 생성 및 저장
        print('Generating sample normal maps...')
        save_evaluation_results(netF, train_dataset, opt, epoch, 'train', device)
        save_evaluation_results(netF, val_dataset, opt, epoch, 'val', device)
        

if __name__ == '__main__':
    train(opt)