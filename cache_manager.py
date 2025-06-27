"""
캐시 관리 유틸리티 스크립트
"""
import os
import json
import shutil
from datetime import datetime

def show_cache_status(cache_dir='cache'):
    """캐시 상태 표시"""
    print("=== 캐시 상태 확인 ===")
    
    if not os.path.exists(cache_dir):
        print("캐시 디렉토리가 존재하지 않습니다.")
        return
    
    cache_files = os.listdir(cache_dir)
    if not cache_files:
        print("캐시가 비어있습니다.")
        return
    
    print(f"캐시 디렉토리: {cache_dir}")
    
    total_size = 0
    for file in cache_files:
        file_path = os.path.join(cache_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            print(f"  {file}: {size / 1024:.1f} KB")
    
    print(f"총 캐시 크기: {total_size / (1024*1024):.2f} MB")
    
    # 메타데이터 확인
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n마지막 업데이트: {metadata.get('last_update', 'Unknown')}")
            print(f"데이터 체크섬: {metadata.get('data_checksum', 'Unknown')[:10]}...")
        except Exception as e:
            print(f"메타데이터 로드 오류: {e}")

def clear_cache(cache_dir='cache'):
    """캐시 초기화"""
    print("=== 캐시 초기화 ===")
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("✓ 캐시가 성공적으로 초기화되었습니다.")
        except Exception as e:
            print(f"✗ 캐시 초기화 실패: {e}")
    else:
        print("캐시 디렉토리가 존재하지 않습니다.")

def backup_cache(cache_dir='cache', backup_dir='cache_backup'):
    """캐시 백업"""
    print("=== 캐시 백업 ===")
    
    if not os.path.exists(cache_dir):
        print("백업할 캐시가 없습니다.")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}_{timestamp}"
        shutil.copytree(cache_dir, backup_path)
        print(f"✓ 캐시가 {backup_path}에 백업되었습니다.")
    except Exception as e:
        print(f"✗ 캐시 백업 실패: {e}")

def restore_cache(backup_path, cache_dir='cache'):
    """캐시 복원"""
    print(f"=== 캐시 복원: {backup_path} ===")
    
    if not os.path.exists(backup_path):
        print("백업 파일이 존재하지 않습니다.")
        return
    
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        shutil.copytree(backup_path, cache_dir)
        print(f"✓ 캐시가 {backup_path}에서 복원되었습니다.")
    except Exception as e:
        print(f"✗ 캐시 복원 실패: {e}")

def optimize_cache(cache_dir='cache'):
    """캐시 최적화 (오래된 임시 파일 제거)"""
    print("=== 캐시 최적화 ===")
    
    if not os.path.exists(cache_dir):
        print("캐시 디렉토리가 존재하지 않습니다.")
        return
    
    # 임시 파일 패턴들
    temp_patterns = ['.tmp', '.temp', '~', '.bak']
    
    removed_count = 0
    removed_size = 0
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if any(file.endswith(pattern) for pattern in temp_patterns):
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed_count += 1
                    removed_size += size
                    print(f"  제거: {file}")
                except Exception as e:
                    print(f"  제거 실패: {file} - {e}")
    
    if removed_count > 0:
        print(f"✓ {removed_count}개 파일 제거 ({removed_size / 1024:.1f} KB 절약)")
    else:
        print("제거할 임시 파일이 없습니다.")

if __name__ == "__main__":
    print("=== 캐시 관리 유틸리티 ===")
    print("1. 캐시 상태 확인")
    print("2. 캐시 초기화")
    print("3. 캐시 백업")
    print("4. 캐시 복원")
    print("5. 캐시 최적화")
    print("6. 종료")
    
    while True:
        choice = input("\n선택하세요 (1-6): ").strip()
        
        if choice == "1":
            show_cache_status()
        elif choice == "2":
            confirm = input("캐시를 초기화하시겠습니까? (y/N): ").strip().lower()
            if confirm == 'y':
                clear_cache()
        elif choice == "3":
            backup_cache()
        elif choice == "4":
            backup_path = input("복원할 백업 경로를 입력하세요: ").strip()
            if backup_path:
                restore_cache(backup_path)
        elif choice == "5":
            optimize_cache()
        elif choice == "6":
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다.")
