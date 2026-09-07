# SME First Datamining Competition

**GK2A 정지위성 관측만으로 전국 96개 지점의 14시 기온·습도를 예측한다.**

[![Kaggle](https://img.shields.io/badge/Kaggle-Leaderboard-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/sme-first-datamining-competition/leaderboard)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![LightGBM](https://img.shields.io/badge/model-Ridge%20%2B%20LightGBM-2C7BB6)](https://lightgbm.readthedocs.io)

- 대회 페이지 — <https://www.kaggle.com/competitions/sme-first-datamining-competition>
- 리더보드 — <https://www.kaggle.com/competitions/sme-first-datamining-competition/leaderboard>

---

## 과제

| | |
|---|---|
| **입력** | GK2A 정지위성 영상, 지점 좌표·고도, 날짜 |
| **출력** | 96개 지점의 14시 기온(TA)·습도(HM) |
| **점수** | `RMSE_TA + 0.1 x RMSE_HM` — 기온이 약 2/3 를 차지 |
| **제약** | ASOS 지상관측은 **학습 라벨로만** 사용 가능, 입력 금지 (규칙 2.2 / 2.3) |

추론 시점에 지상관측을 못 쓰기 때문에, 위성 화소값에서 지면 상태를 읽어내는
피처 설계가 전부다. GK2A DN 값은 밝기온도와 **부호가 반대**라 창 안의
최솟값이 물리적으로 가장 따뜻한 화소가 된다 — 이 부호를 뒤집으면 피처 의미가
통째로 무너진다.

## 파이프라인

```
기상청 API 허브 ──> collect ──> data/*.parquet ──> model ──> model.pkl
   GK2A + ASOS       수집         위성 피처 + 라벨    학습      Ridge + LightGBM
                                        │
                                     validate ──> 제출 노트북
                                      검증          submit
```

## 저장소 구조

```
sme/
├── core/            공통 모듈 — 나머지 전부가 여기에 의존한다
│   ├── kma_client.py    기상청 API 허브 클라이언트 (GK2A · ASOS)
│   ├── grid.py          GK2A 격자 <-> 위경도 변환, 지점 주변 화소 추출
│   ├── evaluation.py    피처 조립 + 평가 (파이프라인의 중심축)
│   ├── features.py      피처 조립 — 그룹 단위로 켜고 끄기
│   ├── stations.py      ASOS 지점 좌표 관리
│   ├── labels.py        ASOS 시간자료 -> 일 단위 레이블
│   └── paths.py         저장소 표준 경로
├── collect/         수집 — 위성, 레이블, 패치, 결측 보충
├── model/           학습 — train, 하이퍼파라미터 탐색, 시드 앙상블
├── experiments/     무엇이 성능을 만들었는지 확인한 실험들
│   ├── channels/        어느 위성 채널을 쓸 것인가 (8)
│   ├── features/        어떤 피처를 만들 것인가 (10)
│   ├── temporal/        시간·연도를 어떻게 쓸 것인가 (8)
│   └── strategy/        어떻게 검증할 것인가 (7)
├── validate/        제출 전 최종 검증 — 홀드아웃, 실전 하루, 리더보드 대조
├── submit/          제출물 생성 — 추론 셀, 노트북, 발표자료, 채점 리허설
└── tools/           진단 — API 키 점검, 응답 속도 측정

scripts/             장시간 수집·최종화 셸 스크립트 (17개)
notebooks/           제출 노트북 (캐글 추론용 / 구글폼 학습용)
data/                수집한 parquet + reference/ 지점 표
docs/                대회 가이드, 제출 절차, 참고 문헌
samples/             API 응답 샘플 (.nc / .bin)
outputs/             submission.csv
대회제출_20260823/    최종 제출 묶음 — 제출 당시 그대로 보존
```

## 설치

```sh
python3 -m venv .venv
./.venv/bin/pip install -e .
```

## 실행

모든 명령은 **저장소 루트에서** 실행한다 (`data/` 상대경로가 루트 기준).

```sh
# 수집
python -m sme.collect.satellite sat --start 20260601 --end 20260930
python -m sme.collect.labels 20260601 20260930 data/asos14_2026.parquet

# 학습 -> model.pkl
python -m sme.model.train

# 검증
python -m sme.validate.holdout
python -m sme.validate.live_day

# 제출물 생성
python -m sme.submit.build_train_notebook
python -m sme.submit.rehearsal "$KMA_API_KEY" 20260817 20260823

# 장시간 작업은 셸 스크립트로 (어디서 실행하든 루트로 이동한다)
./scripts/collect_hist.sh
./scripts/finalize.sh
```

## API 키 설정

키는 **코드에 넣지 않는다.** 환경변수로 전달한다.

```sh
export KMA_API_KEY=<키1>
export KMA_API_KEY2=<키2>
```

`sme.core.kma_client` 는 `KMA_API_KEY` 를 먼저 보고, 없으면 저장소 루트의
`apikey.txt`(한 줄에 키 하나)를 읽는다. `apikey.txt` 는 커밋 대상이 아니다.

동작 확인:

```sh
python -m sme.tools.key_check     # 두 키로 실제 채점 호출을 시험
python -m sme.tools.api_probe     # 타임아웃·재시도 조합별 성공률
```

## 저장소에 없는 것

| 항목 | 이유 | 복구 |
|---|---|---|
| `.venv/` | 464MB, 재생성 가능 | 위 **설치** 참고 |
| `model.pkl`, `model_holdout.pkl` | GitHub 파일당 100MB 제한 초과 | `python -m sme.model.train` |
| `apikey.txt` | 평문 자격증명 | 위 **API 키 설정** 참고 |
| `대회제출_20260823.zip` | `대회제출_20260823/` 와 내용 동일 | 폴더를 압축 |

> **주의** — `대회제출_20260823/1_캐글업로드/model.pkl` 은 실제로 제출한
> 모델이다. 재학습본이 이것과 비트 단위로 같다는 보장은 없으므로 별도로
> 보관해야 한다.
