# PRIS-RHP: Patch-Review Interaction Semantics for Review Helpfulness Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Transformers-4.44-FFD21E?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/Model-BGE--large--en--v1.5-4B8BBE">
</p>


## 📌 개요 (Overview)
본 저장소는 **PRIS-RHP** (Patch-Review Interaction Semantics for Review Helpfulness Prediction)의 공식 구현체를 제공함. 
PRIS-RHP는 Steam 게임 리뷰의 유용성(`votes_up`)을 예측하기 위해 **리뷰 본문**, **공식 패치노트**, **리뷰어/리뷰 메타정보**, **hand-crafted 언어 피처**를 통합적으로 모델링.

PRIS-RHP는 리뷰와 패치노트를 모두 **BGE(BAAI/bge-large-en-v1.5) CLS 임베딩**으로 인코딩한 뒤, 두 임베딩의 **원본 표현과 element-wise 곱·차**를 함께 결합해 패치-리뷰 간 상호작용 의미를 명시적으로 포착한다. 이를 hand-crafted 피처 및 메타 피처와 이어붙여 **Deep Pyramid MLP**에 입력함으로써 로그 변환된 유용성 점수를 예측. *No Man's Sky* Steam 리뷰 데이터를 활용한 실험을 통해, 패치-리뷰 상호작용과 리뷰어 컨텍스트를 함께 활용하는 방식이 텍스트/메타 단일 기반 베이스라인 대비 일관된 성능 향상을 보임을 확인.

## ⚙️ 실행 환경 (Requirements)
- Python 3.10+
- numpy<2.0
- pandas>=1.5.0,<3.0
- scipy
- tqdm
- matplotlib
- torch>=2.1.0
- torchvision
- torchaudio
- transformers==4.44.0
- sentence-transformers==2.7.0
- textstat
- textblob
- vaderSentiment
- contractions
- scikit-learn

의존성 설치:
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

## 📁 저장소 구조 (Repository Structure)

```bash
├── data/
│   ├── raw/
│   │   ├── nomans_reviews_flat_en.csv     # Steam 리뷰 원본
│   │   └── official/                        # 공식 패치노트 14개 (.txt) + steamdb 메타(.xlsx)
│   └── processed/
│       └── nomans_processed.csv             # 전처리된 리뷰 CSV
│
├── model/
│   └── pris.py                              # PRIS-RHP 모델, train_model, predict 함수 정의
│
├── src/
│   ├── bge.py                               # BGEExtractor: 리뷰 CLS + 패치 청크  및 mean-pool
│   ├── data.py                              # DataProcessor: 필터링/HC/BGE/분할/스케일링
│   ├── config.yaml                          # 데이터·BGE·학습 하이퍼파라미터
│   ├── path.py                              # 경로 상수 (BASE/DATA/RAW/PROCESSED/MODEL/SRC)
│   └── utils.py                             # 시드, YAML I/O, parquet 로더, 평가지표
│
├── framework_CLS_ver1_final.ipynb           # 노트북 버전 (실험/디버깅용 참고 자료)
├── main.py                                  # 전처리 → 학습 → 평가를 한 번에 실행
├── data.zip                                 # 대용량 CSV 압축본 (clone 후 압축 해제 필요)
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 데이터 (Data)

*No Man's Sky*의 Steam 리뷰와 14개 공식 패치노트를 매칭한 데이터셋을 사용. 리뷰는 `txt_file_name` 컬럼으로 그 시점의 패치노트와 연결되며, `has_patch_notes == True`인 행만 학습에 사용. 모델이 실제로 사용하는 주요 컬럼은 다음과 같음.

| 컬럼 | 역할 | 설명 |
|---|---|---|
| `review` | 입력 (텍스트) | 리뷰 본문 |
| `votes_up` | 타겟 | 리뷰 유용성 (추천 수) → `log1p` 변환 후 예측 |
| `txt_file_name` | 매칭 키 | 리뷰와 연결되는 공식 패치노트 파일명 |
| `has_patch_notes` | 필터 | 패치 매칭 여부 (True인 행만 사용) |
| `author_num_reviews` | 메타 (수치형) | 리뷰어가 Steam에 남긴 총 리뷰 수 |
| `author_num_games_owned` | 메타 (수치형) | 리뷰어가 보유한 Steam 게임 수 |
| `author_playtime_forever` | 메타 (수치형) | 해당 게임의 전체 누적 플레이타임 (분) |
| `author_playtime_at_review` | 메타 (수치형) | 리뷰 작성 시점까지의 플레이타임 (분) |
| `timestamp_created` | 메타 (수치형) | 리뷰 작성 시각 (Unix timestamp) |
| `received_for_free` | 메타 (이진형) | 게임을 무료로 받았는지 여부 (0/1) |

## 🧠 모델 설명 (Model Description)

PRIS-RHP는 리뷰 의미, 그 시점의 패치노트 의미, 리뷰어 행동, 리뷰의 언어적 특성을 함께 활용하는 멀티모달 회귀 모델로, 다음 7단계 파이프라인으로 구성.

<img width="1280" height="720" alt="PRIS_Framework" src="https://github.com/user-attachments/assets/75d00665-242b-4ed9-a833-22cc6f5c5e51" />

### Step 1. 리뷰 인코딩
리뷰 본문을 `BAAI/bge-large-en-v1.5`에 입력해 `[CLS]` 토큰의 1024차원 임베딩 $r \in \mathbb{R}^{1024}$ 을 확보. 토큰은 최대 512로 truncation.

### Step 2. 패치노트 인코딩
각 리뷰는 `txt_file_name`을 통해 그 시점의 공식 패치노트 한 개와 매칭. 패치노트는 길이가 길어 510 토큰 단위로 청크 분할 → 각 청크의 CLS 임베딩 → mean pooling을 거쳐 단일 1024차원 임베딩 $p \in \mathbb{R}^{1024}$ 로 집계.

### Step 3. Hand-crafted 언어 피처 (4-dim)
정제된 리뷰 텍스트에서 단어 수(`hc_word_count`), Flesch 가독성(`hc_readability`), TextBlob 주관성(`hc_subjectivity`), VADER 감성(`hc_sentiment`) 4개 값을 추출해 $\text{hc} \in \mathbb{R}^{4}$ 를 구성.

### Step 4. 리뷰어/리뷰 메타정보 (6-dim)
유용성에 영향을 주는 리뷰어 행동 및 리뷰 컨텍스트를 6차원 벡터 $\text{meta} \in \mathbb{R}^{6}$ 로 구성.

- 수치형 5개 — `author_num_reviews`(Steam 총 리뷰 수), `author_num_games_owned`(보유 게임 수), `author_playtime_forever`(전체 플레이타임), `author_playtime_at_review`(리뷰 작성 시점 플레이타임), `timestamp_created`(리뷰 작성 시각)
- 이진형 1개 — `received_for_free`(무료 수령 여부)

수치형 피처의 경우 결측치 제거 후 로그 변환(log1p) 및 표준화(StandardScaler)를 적용하여 피처 정규화. 이진형 피처는 결측치를 0으로 고정하는 방식의 원-핫 인코딩 성격의 처리를 통해 모델 입력값으로 최적화함.

### Step 5. 다중 표현 융합
리뷰-패치 상호작용을 명시적으로 표현하기 위해 두 임베딩의 원본·곱·차를 결합하고, 여기에 HC와 메타를 이어붙여 최종 입력 벡터 $x \in \mathbb{R}^{4106}$ 를 생성 ($1024 \times 4 + 4 + 6$).

$$
x = \big[\; r \;\|\; p \;\|\; r \odot p \;\|\; r - p \;\|\; \text{hc} \;\|\; \text{meta} \;\big]
$$

$r \odot p$ 는 리뷰와 패치의 공통 의미를, $r - p$ 는 두 텍스트의 차이 의미를 포착.

### Step 6. Deep Pyramid MLP
$x$ 는 `4106 → 1024 → 512 → 256` 으로 점진 축소되는 3-layer Pyramid MLP에 입력. 각 블록은 `Linear → BatchNorm → GELU → Dropout` 에 residual skip(차원 불일치 시 1×1 Linear projection)을 더한 구조로, 그래디언트 흐름을 안정화.

### Step 7. 예측 헤드 및 학습
최종 256차원 표현은 `Linear(256 → 1)` 헤드를 거쳐 스칼라 예측값을 산출. 학습은 타겟 `y_log = log1p(votes_up)`, 손실 `SmoothL1Loss(β=0.5)`, 옵티마이저 `AdamW(lr=3e-4, wd=1e-4)`, gradient clipping(`max_norm=1.0`), dropout `0.2`로 진행하며, validation loss 기준 patience=10의 Early Stopping을 적용.

## 🚀 실행 방법 (How to Run)

### 환경 구성
가상환경 생성 및 의존성 설치:
```bash
conda create -n pris python=3.10
conda activate pris
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 데이터 준비
저장소 루트의 `data.zip`을 압축 해제하면 `data/raw/` 와 `data/processed/` 에 CSV 파일이 자동으로 배치.

```bash
# Linux / macOS
unzip data.zip

# Windows PowerShell
Expand-Archive -Path data.zip -DestinationPath .
```

압축 해제 후 디렉터리 구조:
- `data/raw/nomans_reviews_flat_en.csv` — Steam 리뷰 원본
- `data/raw/official/*.txt` — 공식 패치노트 14개 (저장소에 직접 포함)
- `data/processed/nomans_processed.csv` — 전처리된 리뷰 CSV

리뷰 CSV의 `txt_file_name` 컬럼이 `data/raw/official/` 내 파일명과 매칭되어야 하며, `has_patch_notes == True`인 행만 학습에 사용.

### 설정 (Configuration)
파일명, BGE 설정, 학습 하이퍼파라미터는 [src/config.yaml](src/config.yaml)을 통해 조정.
 
- `data` — 컬럼명, 메타 컬럼 목록, 최소 단어수, 분할 비율
- `bge` — BGE 체크포인트(`BAAI/bge-large-en-v1.5`), batch size, chunk size, max length
- `args` — 학습 하이퍼파라미터 (epochs, batch_size, learning_rate, dropout_rate, weight_decay, patience, loss_beta, emb/HC/meta 차원)

기본 하이퍼파라미터: `epochs=100`, `batch_size=128`, `lr=3e-4`, `dropout=0.2`, `weight_decay=1e-4`, `patience=10`, `loss_beta=0.5`

기본 데이터 분할 비율: `train : val : test = 7 : 1 : 2`

### 학습 및 평가
다음 명령어 하나로 (필요 시) 전처리, Early Stopping을 포함한 학습, 그리고 테스트셋 평가가 순차적으로 수행.
```bash
python main.py
```
테스트셋에서는 **RMSE**, **MSE**, **MAE**, **MAPE** 지표가 출력.

## 실험 결과 (Experimental Results)

PRIS-RHP는 *No Man's Sky*의 Steam 리뷰와 14개 공식 패치노트를 매칭한 데이터셋에서 평가. 텍스트 단일 기반, 메타 단일 기반, 단순 concatenation 융합 기반 베이스라인과 비교 실험을 수행.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">No Man's Sky (Steam)</th>
    </tr>
    <tr>
      <th>RMSE</th><th>MSE</th><th>MAE</th><th>MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>TNN</td><td>0.475</td><td>0.531</td><td>0.729</td><td>41.417</td></tr>
    <tr><td>HP-BERT</td><td>0.497</td><td>0.581</td><td>0.762</td><td>41.600</td></tr>
    <tr><td>DMMN</td><td>0.456</td><td>0.582</td><td>0.762</td><td>33.968</td></tr>
    <tr><td><b>PRIS-RHP</b></td><td><b>0.367</b></td><td><b>0.400</b></td><td><b>0.633</b></td><td><b>30.780</b></td></tr>
  </tbody>
</table>

## 📬 문의 (Contact)

프로젝트에 대한 질문이나 협업 제안은 아래로 연락주시기 바랍니다.

- **이름**: 박종화 (Jonghwa Park)
- **소속**: 한성대학교 빅데이터공학과
- **이메일**: [jong8620@hansung.ac.kr](mailto:jong8620@hansung.ac.kr)

_Last updated: **2026년 5월**_
