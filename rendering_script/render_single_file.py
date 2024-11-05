#!/usr/bin/env python3

import os
import subprocess
import sys

"""
0002 서브젝트를 렌더링하는 스크립트
"""

def main():
    # 설정
    base_data_dir = '/home/public/data/integratedpifu_data/thuman_progressive_data'  
    subject = '0002'  # 렌더링할 서브젝트 이름
    data_dir = os.path.join(base_data_dir, subject)  # 서브젝트 전체 경로
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
    
    total = len(angles) * len(scripts)
    count = 0

    # 렌더링 작업 시작
    for angle in angles:
        for script in scripts:
            cmd = [
                'blender',
                blank_blend,
                '-b',  # 배치 모드
                '-P',
                script,
                '--',
                subject,      # 첫 번째 인자: 서브젝트 이름 (중복 없이 단일 폴더 이름만 전달)
                str(angle),   # 두 번째 인자: 각도
                base_data_dir # 세 번째 인자: 최상위 데이터 경로 전달
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
