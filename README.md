# SME 데이터분석 경진대회 (2026)

시스템경영공학과 데이터분석 경진대회 작업물 백업. 기상청 API 허브의 GK2A 위성
자료와 ASOS 지상관측 자료로 기온을 예측하는 파이프라인.

## 구조

| 경로 | 내용 |
|---|---|
| `대회제출_20260823/` | 최종 제출 묶음 (캐글 업로드본, 구글폼 제출본, 소스코드, 참고자료) |
| `data/` | 수집한 위성/ASOS 파케이 파일 및 일자별 위성 캐시 |
| `kaggle/` | 대회에서 제공한 원본 자료 (샘플 제출, 관측소 목록, 노트북 템플릿) |
| `collect*.py`, `collect*.sh` | 위성·ASOS 자료 수집 |
| `kma.py` | 기상청 API 허브 클라이언트 |
| `stations.py`, `geo.py`, `features.py` | 관측소·지리·피처 생성 |
| `train.py`, `tune*.py` | 모델 학습 및 하이퍼파라미터 탐색 |
| `eval14.py`, `holdout_test.py`, `year_risk.py` | 검증 |
| `inference_cell.py`, `submission_notebook_filled.ipynb` | 추론 및 제출 노트북 |

## API 키 설정

키는 코드에 넣지 않고 환경변수로 전달한다.

```sh
export KMA_API_KEY=<키1>
export KMA_API_KEY2=<키2>
```

`kma.py` 는 `KMA_API_KEY` 환경변수를 먼저 보고, 없으면 같은 폴더의
`apikey.txt`(한 줄에 키 하나)를 읽는다. `apikey.txt` 는 커밋 대상이 아니다.

## 저장소에 없는 것

`.gitignore` 로 제외된 항목이며, 필요하면 아래 방법으로 복구한다.

- `.venv/` — `python3 -m venv .venv` 후 의존성 재설치
- `*.pkl` (model.pkl 105MB, model_holdout.pkl 98MB) — GitHub 파일당 100MB
  제한을 넘어 제외. `train.py` 로 재학습하여 생성.
  **단 `대회제출_20260823/1_캐글업로드/model.pkl` 은 실제 제출한 모델이므로
  재학습본과 동일하다는 보장이 없다. 별도 보관 필요.**
- `대회제출_20260823.zip` — `대회제출_20260823/` 폴더와 내용이 같아 제외
- `apikey.txt` — 위 "API 키 설정" 참고
