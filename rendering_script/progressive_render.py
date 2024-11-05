#!/usr/bin/env python3

import os
import subprocess
import sys

"""
200개의 데이터를 추가로 랜더링하는 스크립트
"""

def main():
    # 설정
    data_dir = '/home/public/data/integratedpifu_data/thuman_progressive_data'  
    blank_blend = '/home/public/IntegratedPIFu/rendering_script/blank.blend'
    scripts = [
        '/home/public/IntegratedPIFu/rendering_script/render_full_mesh.py',
        '/home/public/IntegratedPIFu/rendering_script/render_normal_map_of_full_mesh.py'
    ]
    angles = [i * 36 for i in range(10)]  # 0도부터 324도까지 36도 간격으로 10개 각도 설정

    # 필수 파일 및 디렉토리 존재 여부 확인
    if not os.path.isdir(data_dir):
        print(f"데이터 디렉토리 '{data_dir}'이 존재하지 않습니다.")
        sys.exit(1)
    
    if not os.path.isfile(blank_blend):
        print(f"Blender 파일 '{blank_blend}'이 존재하지 않습니다.")
        sys.exit(1)
    
    for script in scripts:
        if not os.path.isfile(script):
            print(f"스크립트 '{script}'이 존재하지 않습니다.")
            sys.exit(1)
    
    # 서브젝트 목록 가져오기
    subjects = sorted(os.listdir(data_dir))
    if len(subjects) == 0:
        print(f"'{data_dir}'에 서브젝트가 존재하지 않습니다.")
        sys.exit(1)
    
    total = len(subjects) * len(angles) * len(scripts)
    count = 0

    for subject in subjects:
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path):
            print(f"'{subject_path}'는 디렉토리가 아니므로 건너뜁니다.")
            continue  # 디렉토리가 아닌 항목은 건너뜀
        for angle in angles:
            for script in scripts:
                cmd = [
                    'blender',
                    blank_blend,
                    '-b',  # 배치 모드
                    '-P',
                    script,
                    '--',
                    subject,       # 첫 번째 인자: 서브젝트
                    str(angle),    # 두 번째 인자: 각도
                    data_dir       # 세 번째 인자: 데이터 경로 추가
                ]
                print(f"서브젝트: {subject}, 각도: {angle}, 스크립트: {script} 실행 중...")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"명령어 실행 중 오류 발생: {' '.join(cmd)}")
                    print(e)
                count += 1
                print(f"진행 상황: {count}/{total} ({(count/total)*100:.2f}%)")
    
    print("모든 렌더링 작업이 완료되었습니다.")

if __name__ == '__main__':
    main()