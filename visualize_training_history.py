import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

def visualize_training_history(history_path):
    # JSON 파일 로드
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # seaborn 스타일 설정
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 서브플롯 생성
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, height_ratios=[3, 1, 1])
    
    epochs = history['epochs']
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # 1. 손실 그래프 (train vs validation)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.scatter(history['best_epoch'], history['best_val_loss'], 
               color='green', s=100, zorder=5, label='Best Model')
    
    # 과적합 판단을 위한 차이 시각화
    ax1.fill_between(epochs, train_losses, val_losses, 
                    where=(np.array(val_losses) > np.array(train_losses)),
                    color='red', alpha=0.1, label='Potential Overfitting')
    
    ax1.set_title('Training and Validation Loss Over Time', pad=20)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 최적 지점 표시
    ax1.axvline(x=history['best_epoch'], color='g', linestyle='--', alpha=0.5)
    ax1.text(history['best_epoch'], min(train_losses), 
             f'Best Epoch: {history["best_epoch"]}', 
             horizontalalignment='right')
    
    # 2. 학습률 변화
    ax2.plot(epochs, history['learning_rate'], 'g-', label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    
    # 3. 에포크당 수행 시간
    ax3.plot(epochs, history['time_per_epoch'], 'm-', label='Time per Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    
    # 과적합 분석
    train_val_diff = np.array(val_losses) - np.array(train_losses)
    is_overfitting = train_val_diff > 0.01  # 임계값 설정
    
    # 분석 결과 텍스트 추가
    analysis_text = (
        f'Training Analysis:\n'
        f'Best Validation Loss: {history["best_val_loss"]:.6f} (Epoch {history["best_epoch"]})\n'
        f'Total Training Time: {history["total_time"]:.1f}s\n'
        f'Potential Overfitting: {"Yes" if np.any(is_overfitting) else "No"}'
    )
    plt.figtext(1.02, 0.5, analysis_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('/home/dong/projects/IntegratedPIFu/training_history_visualization.png', bbox_inches='tight', dpi=300)
    plt.close()


# 사용 예시
history_path = '/home/dong/projects/IntegratedPIFu/results/first_normal_test/training_history.json'
visualize_training_history(history_path)