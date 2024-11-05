# 1. train_integratedPIFu에서 low-pifu 옵션 import
# 2. load_model_weights, load_model_weights_for_high_res_too = False
# 3. test_script_activate = False

NAME='first_low_pifu_test'
NUM_EPOCH='2'
SERIAL_BATCHES='--serial_batches'  # 랜덤 배치 옵션
BATCH_SIZE='4'

# 훈련 명령어 실행
python apps/train_integratedPIFu.py ${SERIAL_BATCHES} \
  --name ${NAME} \
  --num_epoch ${NUM_EPOCH} \
  --batch_size ${BATCH_SIZE}