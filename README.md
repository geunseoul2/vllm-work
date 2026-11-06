# vLLM workload experiments

vLLM에서 혼합 길이 요청과 요청 순서에 따른 성능을 비교하는 실험 코드와 결과입니다.

## 구성

- `build_real_mixed_workload.py`: Dolly 데이터에서 짧은 요청과 긴 요청을 선정하고 workload 및 manifest 생성
- `open.sh`: Qwen3-8B 서버 실행 (Qwen3-0.6B draft model 사용)
- `run.sh`: 요청률 2, 4, 6, 8 req/s 벤치마크
- `SchedCodes/`: 스케줄링 및 speculative decoding 관련 실험 코드
- `real_mixed_workload/`: 생성된 workload와 샘플 manifest
- `data/`: 실험 입력 데이터
- `results/`, `results_dspark/`, `results_eagle/`: 저장된 실험 결과

## 실행

CUDA 및 실험 옵션을 지원하는 vLLM 환경이 필요합니다. 사용한 vLLM/CUDA 버전은 아직 고정하지 않았습니다. 설치 버전에 따라 CLI 옵션 지원 여부를 확인하세요.

```bash
pip install 'transformers>=4.51' huggingface_hub
python build_real_mixed_workload.py
bash open.sh
```

서버가 준비되면 별도 터미널에서 실행합니다.

```bash
bash run.sh real_mixed_workload dolly_long_first_fixed128
```

기본 workload는 짧은 요청 160개와 긴 요청 40개이며, fixed workload의 출력 길이는 128토큰입니다. GPU, 모델, 포트와 메모리 설정은 실행 전 스크립트에서 확인하세요.

## 데이터 출처

Dolly 기반 workload의 원본은 Databricks의 `databricks-dolly-15k`입니다.
출처: https://huggingface.co/datasets/databricks/databricks-dolly-15k
생성 스크립트에 명시된 데이터 라이선스는 CC BY-SA 3.0입니다. Dolly에서 파생된 데이터는 해당 라이선스와 원본 저작자 표시를 따릅니다. 이 안내는 프로젝트 코드 전체에 대한 라이선스 부여를 의미하지 않습니다.
