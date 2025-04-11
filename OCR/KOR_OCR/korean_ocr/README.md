# 한글 간판 OCR (PaddleOCR 파인튜닝 모델)

이 저장소는 한글 간판 텍스트 인식을 위해 파인튜닝된 PaddleOCR 모델을 사용하는 방법을 제공합니다.

## 프로젝트 개요

- 사용 환경: PaddleOCR, Python 3.8, PaddlePaddle GPU 2.6.2, CUDA 11.8, cuDNN 8.x
- 성과: 부분 문자열 일치율 기준 약 10% 향상

## 필요 환경

- Windows 10/11
- Python 3.8 이상
- Anaconda 또는 Miniconda
- CUDA 11.8 및 cuDNN 8.x (GPU 사용 시)
- NVIDIA GPU (외장 GPU 권장)

## 설치 방법

1. **Conda 환경 설정**
   ```bash
   # conda 환경 생성
   conda create -n paddleocr python=3.8
   conda activate paddleocr
   
   # PaddlePaddle GPU 버전 설치
   pip install paddlepaddle-gpu==2.6.2
   
   # PaddleOCR 설치
   pip install "paddleocr>=2.10.0"

PaddleOCR 코드 다운로드
bashgit clone https://github.com/PaddlePaddle/PaddleOCR.git
참고: CUDA 11.8 및 cuDNN 8.x는 별도로 설치해야 합니다. NVIDIA 개발자 사이트에서 다운로드할 수 있습니다.
파인튜닝된 모델 파일 설치

이 저장소의 korean_rec 폴더를 PaddleOCR/inference/ 경로에 복사합니다.


config.bat 파일 수정

config.bat 파일을 열고 PaddleOCR 설치 경로를 자신의 환경에 맞게 수정합니다.
⚠️ 주의: 경로명에 개인 사용자명이 포함되어 있을 수 있으니 반드시 확인하고 수정하세요.



사용 방법

명령 프롬프트에서 다음과 같이 실행합니다:
info.bat 이미지파일.jpg
또는
info.sh 이미지파일.jpg

OCR 결과는 ocr_results 폴더에 저장됩니다.

평가 방식 설명
부분 문자열 일치율 계산 방식:

정답 텍스트의 모든 단어가 OCR 결과에 포함되어 있으면 100%로 간주
예: 정답이 "서울모터스"이고, OCR 결과가 "서울모터스 73"이면 100% 일치
일부 단어만 포함된 경우 해당 비율을 계산

라이선스

이 프로젝트는 Apache License 2.0을 따릅니다.
PaddleOCR의 라이선스 정책을 준수합니다.