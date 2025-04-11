@echo off
REM 한글 OCR 실행 스크립트 (파인튜닝된 PaddleOCR 모델 사용)

REM 사용 방법 안내
echo ===== 한글 간판 OCR 시스템 =====
echo 사용법: info.sh [이미지파일경로]
echo 예시: info.sh test.jpg

REM 입력 인자 확인
if "%~1"=="" (
    echo 오류: 이미지 파일 경로를 입력해주세요.
    exit /b 1
)

REM 환경 설정
echo 환경 설정 중...

REM 설정 파일 존재 여부 확인
if exist config.bat (
    echo 설정 파일 로딩 중...
    call config.bat
) else (
    REM 기본 경로 설정
    echo 기본 설정 사용 중...
    REM PaddleOCR 경로 (사용자 환경에 맞게 수정 필요)
    set PADDLE_OCR_DIR=C:\Users\dhtkd\PaddleOCR
    set KOREAN_MODEL_DIR=%PADDLE_OCR_DIR%\inference\korean_rec\inference
    set KOREAN_DICT_PATH=%PADDLE_OCR_DIR%\ppocr\utils\dict\korean_dict.txt
)

REM 결과 저장 디렉토리 설정
set OUTPUT_DIR=%~dp0ocr_results
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Conda 환경 활성화
echo Conda 환경 활성화 중...
call conda activate paddleocr
if %ERRORLEVEL% neq 0 (
    echo 오류: paddleocr Conda 환경을 활성화할 수 없습니다. 
    echo Conda가 설치되어 있고 paddleocr 환경이 생성되어 있는지 확인하세요.
    exit /b 1
)

REM OCR 실행
echo OCR 처리 시작: %~1
python "%PADDLE_OCR_DIR%\tools\infer\predict_system.py" ^
  --image_dir="%~1" ^
  --rec_model_dir="%KOREAN_MODEL_DIR%" ^
  --rec_char_dict_path="%KOREAN_DICT_PATH%" ^
  --use_space_char=True ^
  --use_gpu=True ^
  --use_angle_cls=True ^
  --det_db_box_thresh=0.5 ^
  --det_db_thresh=0.3 ^
  --det_db_unclip_ratio=1.6 ^
  --drop_score=0.5 ^
  --draw_img_save_dir="%OUTPUT_DIR%"

REM 실행 결과 확인
if %ERRORLEVEL% neq 0 (
    echo OCR 처리 중 오류가 발생했습니다.
) else (
    echo OCR 처리가 완료되었습니다.
    echo 결과 이미지: %OUTPUT_DIR%
)

REM 완료
echo 처리 완료.