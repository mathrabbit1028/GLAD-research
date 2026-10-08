# GLAD Research

TU 그래프 데이터셋에서 그래프 수준 이상 탐지 모델을 같은 데이터 분할과 평가 절차로 비교하는 프로젝트입니다. 실험 설정, 모델 구현, 데이터 처리, 평가 및 결과 집계를 분리해 모델을 추가해도 공통 실행 코드를 수정하지 않도록 구성했습니다.

## 프로젝트 구조

```text
configs/
  config.yaml             공통 실행·평가 설정
  model/*.yaml            모델별 어댑터와 학습 설정
  dataset/*.yaml          데이터셋별 분할·배치 설정
data/TU/                  TU 원본 파일(raw); processed 파일은 로컬 생성
src/data/                데이터 로딩과 분할
src/models/              공통 모델 인터페이스와 모델별 어댑터
src/evaluation/          점수 수집과 평가 지표
src/utils/               설정, seed, 집계, 통계 검정, 표 생성
src/train.py             학습 및 체크포인트 저장
src/evaluate.py          체크포인트 기반 독립 평가
scripts/                 전체 조합 실행과 논문용 표 생성
outputs/                 체크포인트, 로그, 결과(Git 제외)
```

## 공통 실험 규칙

`configs/dataset/`의 각 파일이 TU 데이터셋 하나를 정의합니다. 현재 등록된 데이터셋은 MUTAG, DD, COX2, ER_MD, SW-620, MOLT-4, PC-3, MCF-7입니다. 다수 클래스를 정상, 소수 클래스를 이상으로 취급하며, seed마다 정상 그래프의 80%를 학습에 사용합니다. 나머지 정상 그래프는 테스트에 사용합니다. 중형 데이터셋은 테스트 정상 수 이하의 이상 그래프를 선택하고, 대형 데이터셋은 모든 이상 그래프를 사용합니다. 사용된 분할 인덱스는 체크포인트에 저장됩니다.

모든 모델 어댑터는 **그래프마다 하나의 점수**를 반환하며 높은 점수가 더 이상임을 뜻합니다. F1 판정 임계값은 학습 정상 그래프 점수의 95번째 백분위수로 정하고 체크포인트에 저장합니다. 테스트 데이터는 임계값 선택에 사용하지 않습니다. 평가는 AUC, F1, Accuracy, 그래프당 추론 시간, CUDA 최대 메모리를 기록합니다. CPU 실행에서는 CUDA 메모리 값이 비어 있습니다.

모델 구조, epoch, optimizer 같은 값은 해당 `configs/model/*.yaml`에서 지정합니다. 공통 규칙과 다른 실험을 할 때는 설정과 결과를 별도로 기록해야 합니다.

## 데이터 준비

`processed/`는 Git에서 제외합니다. TU 원본 파일을 `data/TU/<데이터셋>/raw/`에 준비한 뒤 아래 명령으로 PyTorch Geometric 처리 파일을 생성하세요. 기본 명령은 `configs/dataset/`에 등록된 전체 데이터셋을 처리하며, `--dataset MUTAG`처럼 지정 데이터셋만 만들 수도 있습니다. `--force`는 기존 처리 파일을 다시 생성합니다. 이 스크립트는 다운로드하지 않고, 필요한 raw 파일이 없으면 파일 경로를 표시하며 중단합니다.

```bash
python scripts/prepare_tu_data.py
python scripts/prepare_tu_data.py --dataset MUTAG
```

원본 데이터는 각 TU 데이터셋의 배포 조건을 확인해 별도로 준비해야 합니다.

## 설치와 단일 실행

프로젝트 루트에서 실행합니다. PyTorch와 PyTorch Geometric을 사용 장치에 맞게 설치한 뒤 필요한 패키지를 설치하세요.

```bash
pip install -r requirements.txt
python -m src.train dataset=mutag seed=0
python -m src.evaluate --checkpoint outputs/checkpoints/<모델명>_MUTAG_seed0.pt
```

학습 명령은 `configs/config.yaml`의 기본 모델을 사용합니다. 다른 모델은 `model=<configs/model의 파일명>`으로 선택합니다. 데이터셋도 `dataset=<configs/dataset의 파일명>`으로 선택합니다. 실행 장치는 `device=auto`(CUDA 우선), `device=cpu`, `device=cuda` 중 하나로 지정할 수 있습니다. 설정 값은 `model.training.learning_rate=0.0005`처럼 덮어쓸 수 있습니다. 학습이 끝나면 출력된 체크포인트 경로를 평가 명령에 전달하세요.

## 반복 실행과 결과

```bash
bash scripts/run_all_baselines.sh
bash scripts/generate_paper_tables.sh
```

첫 스크립트는 `configs/model/*.yaml`과 `configs/dataset/*.yaml`의 모든 조합을 실행합니다. 조합별 반복 횟수는 `configs/config.yaml`의 `trials` 값이며 seed는 0부터 시작합니다. 실행 로그는 `outputs/logs/`, 체크포인트는 `outputs/checkpoints/`에 저장됩니다. 평가가 완료되면 run별 JSON과 그래프별 점수 CSV가 `outputs/results/`에 생성됩니다. 집계 결과는 `outputs/results/results.csv`, LaTeX 표는 `outputs/results/table.tex`입니다. 두 모델의 결과가 같은 데이터셋과 seed로 정렬되어 있으면 `src/utils/stat_test.py`의 `compare_paired`로 대응 표본 검정을 수행할 수 있습니다.

전체 조합 실행은 데이터셋 규모와 모델 설정에 따라 오래 걸릴 수 있습니다. 실행 전에 YAML의 dataset, model, `trials`, 장치 설정을 확인하세요.

## 모델 추가

1. `src/models/base_model.py`의 `BaseAnomalyModel`을 구현합니다. `fit`은 정상 학습 그래프를 사용하고, `anomaly_score`는 그래프당 점수 하나를 반환해야 합니다. `state_dict`와 `load_state_dict`는 체크포인트 저장·복원을 담당합니다.
2. `configs/model/<모델명>.yaml`을 만들고 `name`, `adapter`(예: `src.models.<모듈>.<클래스>`), 모델별 학습 설정을 정의합니다. 설정 파일명은 CLI의 `model=` 값이 되고 `name`은 결과에 기록됩니다.
3. 단일 학습·평가를 먼저 실행해 점수 방향, 체크포인트 복원, 결과 스키마를 확인합니다. 등록된 모델 설정 파일은 전체 조합 스크립트가 자동으로 찾습니다.

새 모델이 같은 TU 분할과 평가기를 사용하면 `src/train.py`와 `src/evaluate.py`에 모델 이름별 분기를 추가할 필요가 없습니다.
