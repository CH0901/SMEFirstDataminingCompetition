# SME First Datamining Competition

GK2A 정지위성 관측으로 전국 96개 지점의 14시 기온·습도를 맞추는 대회 코드.

- 대회: https://www.kaggle.com/competitions/sme-first-datamining-competition
- 리더보드: https://www.kaggle.com/competitions/sme-first-datamining-competition/leaderboard

## 과제

| | |
|---|---|
| 입력 | GK2A 위성 영상, 지점 좌표·고도, 날짜 |
| 출력 | 96개 지점의 14시 기온(TA)·습도(HM) |
| 점수 | `RMSE_TA + 0.1 * RMSE_HM` |
| 제약 | ASOS 지상관측은 학습 라벨로만 사용 가능 (규칙 2.2 / 2.3) |

추론할 때 지상관측을 못 쓰므로 위성 화소값에서 지면 상태를 읽어내는 게 관건이다.
GK2A DN 값은 밝기온도와 부호가 반대라(실측 상관 -0.46 ~ -0.56), 창 안의 최솟값이
물리적으로 가장 따뜻한 화소가 된다. 부호를 반대로 잡으면 피처 의미가 뒤집힌다.

모델은 Ridge + LightGBM 이고 시드 앙상블을 쓴다.

## 구조

```
sme/
├── core/            공통 모듈. 나머지가 여기에 의존한다
│   ├── kma_client.py    기상청 API 허브 클라이언트 (GK2A, ASOS)
│   ├── grid.py          GK2A 격자 <-> 위경도 변환, 지점 주변 화소 추출
│   ├── evaluation.py    피처 조립과 평가
│   ├── features.py      피처를 그룹 단위로 켜고 끄기
│   ├── stations.py      ASOS 지점 좌표
│   ├── labels.py        ASOS 시간자료를 일 단위 레이블로
│   └── paths.py         저장소 표준 경로
├── collect/         수집 (위성, 레이블, 패치, 결측 보충)
├── model/           학습, 하이퍼파라미터 탐색, 시드 앙상블
├── experiments/     성능에 무엇이 기여했는지 확인한 실험들
│   ├── channels/        어느 채널을 쓸 것인가 (8)
│   ├── features/        어떤 피처를 만들 것인가 (10)
│   ├── temporal/        시간·연도를 어떻게 쓸 것인가 (8)
│   └── strategy/        어떻게 검증할 것인가 (7)
├── validate/        홀드아웃, 실전 하루, 리더보드 대조
├── submit/          추론 셀, 노트북, 발표자료, 채점 리허설
└── tools/           API 키 점검, 응답 속도 측정

scripts/             장시간 수집·최종화 셸 스크립트 17개
notebooks/           제출 노트북 (캐글 추론용, 구글폼 학습용)
data/                수집한 parquet, reference/ 에 지점 표
docs/                대회 가이드, 제출 절차, 참고 문헌
samples/             API 응답 샘플 (.nc, .bin)
outputs/             submission.csv
대회제출_20260823/    최종 제출 묶음. 제출 당시 그대로 둔다
```

## 설치

```sh
python3 -m venv .venv
./.venv/bin/pip install -e .
```

## 실행

`data/` 를 상대경로로 참조하므로 저장소 루트에서 실행한다.

```sh
# 수집
python -m sme.collect.satellite sat --start 20260601 --end 20260930
python -m sme.collect.labels 20260601 20260930 data/asos14_2026.parquet

# 학습 (model.pkl 생성)
python -m sme.model.train

# 검증
python -m sme.validate.holdout
python -m sme.validate.live_day

# 제출물
python -m sme.submit.build_train_notebook
python -m sme.submit.rehearsal "$KMA_API_KEY" 20260817 20260823
```

셸 스크립트는 어디서 실행하든 루트로 이동한다.

```sh
./scripts/collect_hist.sh
./scripts/finalize.sh
```

## API 키

키는 코드에 넣지 않고 환경변수로 넘긴다.

```sh
export KMA_API_KEY=<키1>
export KMA_API_KEY2=<키2>
```

`sme.core.kma_client` 는 `KMA_API_KEY` 를 먼저 보고, 없으면 저장소 루트의
`apikey.txt` 를 읽는다 (한 줄에 키 하나). `apikey.txt` 는 커밋하지 않는다.

```sh
python -m sme.tools.key_check     # 두 키로 실제 채점 호출을 시험
python -m sme.tools.api_probe     # 타임아웃·재시도 조합별 성공률
```

## 저장소에 없는 것

| 항목 | 이유 | 복구 |
|---|---|---|
| `.venv/` | 464MB, 재생성 가능 | 위 설치 참고 |
| `model.pkl`, `model_holdout.pkl` | 파일당 100MB 제한 초과 | `python -m sme.model.train` |
| `apikey.txt` | 평문 자격증명 | 위 API 키 참고 |
| `대회제출_20260823.zip` | `대회제출_20260823/` 와 내용 동일 | 폴더를 압축 |

`대회제출_20260823/1_캐글업로드/model.pkl` 은 실제로 제출한 모델이다. 재학습본이
이것과 같다는 보장이 없으니 따로 보관해 둘 것.
