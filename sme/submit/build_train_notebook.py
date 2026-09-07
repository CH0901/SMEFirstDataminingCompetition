"""제출물 ② 학습 노트북 생성.

이 노트북은 혼자서 model.pkl 을 그대로 재현한다. 필요한 모듈 소스를
노트북 안에 직접 써 넣기 때문에 다른 파일이 없어도 돌아간다.
(수집된 데이터 parquet 만 있으면 된다.)
"""
from __future__ import annotations
import json, pathlib

def md(t): return dict(cell_type="markdown", metadata={}, source=t.splitlines(keepends=True))
def code(t): return dict(cell_type="code", metadata={}, execution_count=None,
                         outputs=[], source=t.splitlines(keepends=True))

def writefile(src, dest=None):
    """모듈 소스를 %%writefile 셀로 만든다 — 노트북만으로 재현되게.

    src 는 이 저장소 안의 경로, dest 는 노트북이 캐글에 써 넣을 경로.
    패키지 구조를 그대로 재현해야 모듈 사이의 import 가 맞는다.
    """
    dest = dest or src
    return code(f"%%writefile {dest}\n" + pathlib.Path(src).read_text())

cells = [
md("""# 14시 기온·습도 예측 — 학습 노트북

제1회 시스템경영공학과 데이터분석 경진대회 · 제출물 ②

**과제**: GK2A 정지위성 관측만으로 전국 96개 지점의 14시 기온(TA)·습도(HM)를
맞춘다. 점수는 `RMSE_TA + 0.1 x RMSE_HM` 이라 기온이 약 2/3 를 차지한다.

**제약**: ASOS 지상관측은 *학습 라벨로만* 쓸 수 있고 입력으로는 못 쓴다
(규칙 2.2/2.3). 그래서 추론 시점에 우리가 아는 것은 위성 영상, 지점 좌표·고도,
날짜뿐이다.

**인과성**: 대상은 14시(KST) = 05 UTC. 그 시각까지 관측된 영상만 입력이다.
당일 00·02·04·05 UTC(KST 09·11·13·14시)와 전날 18·21 UTC(KST 새벽 3·6시)를 쓴다.

이 노트북을 위에서 아래로 실행하면 제출한 `model.pkl` 이 그대로 나온다
(시드 고정 + LightGBM `deterministic=True`)."""),

md("""## 1. 왜 이 구조인가 — 타깃을 먼저 뜯어봤다

14시 기온의 분산을 세 성분으로 나눠 봤다.

| 성분 | 분산 | 비중 | 무엇으로 잡히나 |
|---|---:|---:|---|
| 날짜간 (그날의 기단) | 9.31 | 66% | 전국 편차·이웃 지점 피처 |
| 지점간 (고정 특성) | 1.31 | 9% | 위경도·고도·지점번호 |
| 날짜x지점 상호작용 | 3.61 | 25% | **여기가 진짜 어려운 부분** |

즉 성능의 3분의 2는 "오늘이 더운 날인가"를 맞히는 문제고, 그건 한 지점의
화소값이 아니라 **전국 패턴**에 들어있다. 이 관찰이 아래 피처 설계를 결정했다.

남은 상호작용 성분을 실측 ASOS 변수와 대조해 보면:

    습도 -0.476 · 일사 +0.383 · 지면온도 +0.373 · 전운량 -0.258 · 풍속 +0.053

습도·일사·지면온도가 지배적인데 셋 다 입력 금지다. 그래서 각각의 위성 대응물
(수증기 흡수차, 반사도, 적외 최저휘도)을 피처로 만들어 대신 쓴다."""),

md("""## 2. 물리 배경

**분리대기창(split-window)**. KMA 공식 GK2A 지표면온도 알고리즘(ATBD)은

    LST = A0 + A1*T13 + A2*(T13 - T15)

를 쓴다. IR105(10.35um)와 IR123(12.37um)은 파장이 가까워 지표 신호는 거의
같지만 **수증기 흡수량이 다르다**. 그래서 두 채널의 차이가 대기 수증기 효과를
지워주는 보정항이 된다. 보조변수로 위성 천정각이 들어가는데 경로 길이 효과라
`sec(theta)-1` 형태다.

**GK2A DN 은 밝기온도와 반대**. 실측 상관이 전부 음수(-0.46~-0.56)였다.
따라서 창 안의 **최솟값(min9)이 물리적으로 가장 따뜻한 화소** = 구름에 덜 가린
맑은 하늘 대리값이다. 이 부호를 잘못 잡으면 피처 의미가 뒤집힌다."""),

md("## 3. 모듈 정의\n\n재현을 위해 사용한 코드를 노트북 안에 그대로 써 넣는다."),
code("""import pathlib
for d in ("sme", "sme/core", "sme/model"):
    pathlib.Path(d).mkdir(exist_ok=True)
    (pathlib.Path(d) / "__init__.py").touch()
print("패키지 준비 완료")"""),
writefile("sme/core/paths.py"),
writefile("sme/core/grid.py"),
writefile("sme/core/evaluation.py"),

md("""## 4. 피처 조립

`evaluation.assemble()` 이 하는 일:

1. **창 통계** — 지점 주변 9x9(약 18km) 화소에서 중심값·평균·표준편차·최소·최대.
   중심 화소만 쓰면 구름 가장자리에서 값이 널뛴다. 표준편차 자체가 "구름이
   깨져 있는지"를 알려준다.
2. **시각별 전개** — 6개 관측 시각을 각각 별도 피처로 (`_h05`, `_p21` 등),
   그리고 시각 간 차분(`_d1h`, `_dnt`)으로 변화 속도를 담는다.
3. **분리대기창 보정항** — `SWD_* = IR105 - IR123`, `SWDmin_*`, 구름 지표 `CLD_*`.
4. **기하** — 위성 천정각의 `sec(theta)-1`, 연중일의 사인·코사인.
5. **전국 편차 `ANO_*`** — 그날 전국 평균 대비 이 지점의 편차. 타깃 분산의 66%가
   날짜 효과였으므로 이게 가장 큰 기여를 한다.
6. **이웃 `NB_*`** — 최근접 8개 지점의 평균.
7. **방향별 이웃 `DW_/DE_/DN_/DS_` 와 대비 `dEW_/dNS_`** — 8개 평균은 방향을
   지운다. 서쪽이 뜨겁고 동쪽이 차가우면 서풍 기단 유입이고, 동해안 푄도
   "산맥 서쪽 구름 / 동쪽 맑음"이라는 방향 구조다.
   실측: 기온 1.252 -> 1.238, 습도 7.309 -> 7.233."""),

code("""import numpy as np, pandas as pd, lightgbm as lgb, pickle
from sme.core import evaluation as E

CHANNELS = ["IR105", "IR123", "SW038"]
df = E.assemble(CHANNELS)

feats = E.BASE + [c for c in df.columns
                  if c.split("_")[0] in CHANNELS
                  or c.startswith(("SWD_", "SWDmin_", "CLD_"))]
feats = [f for f in dict.fromkeys(feats) if f in df.columns]
ano = [c for c in df.columns
       if c.startswith(("ANO_", "NB_", "DW_", "DE_", "DN_", "DS_", "dEW_", "dNS_"))]

print(f"표본 {len(df):,}행 · 날짜 {df['date'].nunique()}일 · 지점 {df['STN'].nunique()}개")
print(f"피처: 기본 {len(feats)}개 + 공간 {len(ano)}개")"""),

md("""## 5. 타깃마다 다른 피처를 쓴다

점수가 `RMSE_TA + 0.1*RMSE_HM` 이라 두 타깃을 각각 최적화하는 게 이득이다.
실측으로 갈린 두 가지:

- **연도(YEAR)**: 8월 말 기온은 연도 간 변동이 커서 넣어야 편향이 잡힌다
  (2026 검증 TA 1.506 -> 1.435, 편향 -0.54 -> +0.15). 그런데 습도는 연도 효과가
  계절과 뒤섞여 크게 나빠졌다 (HM 7.078 -> 9.654). 그래서 기온만 넣는다.
- **공간 피처**: 기온에는 크게 도움이 되지만 습도에서는 나빠졌다 (7.078 -> 7.340).

한 가지 시도해서 **기각한** 것도 남겨 둔다. 습도가 기온 잔차와 가장 강하게
연결(-0.476)되므로 *예측 습도*를 기온 입력으로 넣어봤다. 결과는 1.281 -> 1.283 로
효과가 없었다. 물리는 맞지만 우리 습도 예측 자체가 부정확해(RMSE 7%p) 그
신호가 노이즈에 묻힌 것이다."""),

code("""feats_by_target = {
    "TA14": feats + ano,
    "HM14": [f for f in feats if f != "YEAR"],   # 습도는 연도·공간 피처 제외
}
for t, f in feats_by_target.items():
    print(f"  {t}: {len(f)}개")"""),

md("""## 6. 선형 + GBM 잔차 구조

트리는 **학습 범위 밖으로 외삽하지 못한다**. 8월 자료로 학습한 GBM 단독 모델은
6월에도 "8월이니까 30도"를 그대로 내놓아 편향이 +4.3도였다. 그래서 Ridge 선형
모델을 앞에 두어 추세를 담당시키고, GBM 은 그 **잔차만** 학습한다. 계절 밖
편향이 사라졌을 뿐 아니라 계절 안 성능도 함께 좋아졌다.

지점번호(`STN_CAT`)는 GBM 에만 범주형으로 넣는다. 선형에 넣으면 번호의 크기를
비교하게 되는데 그건 아무 의미가 없다.

시드를 바꿔 학습한 모델 3개를 평균한다 (실측 2.282 -> 2.254). 시드가 고정되어
있고 `deterministic=True` 라서 다시 돌려도 같은 결과가 나온다 (규칙: 무작위성 제거)."""),

writefile("sme/model/train.py"),

code("""from sme.model import train as T
print("설정:", {k: T.PARAMS[k] for k in
      ("learning_rate","num_leaves","min_data_in_leaf","feature_fraction","lambda_l2")})
print(f"라운드 {T.ROUNDS} · Ridge alpha {T.RIDGE_ALPHA} · 시드 {T.SEEDS}")"""),

md("""### 하이퍼파라미터를 고른 근거

날짜 그룹 5-fold CV 로 골랐다. 초기값(leaves 31 / lr 0.04 / 600라운드)은 표본이
59일이던 시절 것이라 데이터가 늘어난 뒤로는 용량이 부족했다.

    기준선 2.484 -> leaves 63 2.467 -> +lr 0.02/1500r 2.458 -> leaves 127 2.454

라운드는 1500에서 포화했고(2500도 동일), 선형 규제는 강할수록 나았다.

**검증은 반드시 날짜로 묶어야 한다.** 같은 날 96개 지점은 같은 기단을 공유하므로
무작위 분할은 사실상 정답을 흘린다. 홀드아웃 1주(672행)는 노이즈가 커서
0.03 정도 차이는 구별되지 않았고, 그래서 판단은 21,000행 이상의 5-fold CV 로 했다."""),

md("## 7. 학습 실행\n\n`train.main()` 이 학습부터 `model.pkl` 저장까지 수행한다."),
code("T.main()"),

md("""## 8. 규칙 준수

| 규칙 | 내용 | 준수 |
|---|---|---|
| 2.1 | 기상청 API 허브 자료만 사용 | GK2A LE1B + ASOS(라벨) 만 사용 |
| 2.2 | ASOS 는 학습 라벨로만 | 입력 피처에 ASOS 파생물 없음 |
| 2.3 | 외부 자료 금지 | 지점 좌표·고도는 대회 제공 `station_list.csv` |
| 3 | 14시 이하 관측만 | 05 UTC 이하 + 전날 18/21 UTC |
| 재현성 | 무작위성 제거 | 시드 고정 · `deterministic=True` |

**기후 평년값은 싣지 않았다.** 운영진 답변("ASOS는 모델 학습시 label값으로만
활용 가능")에 따라 지점별 과거 평균기온을 추론에 쓰는 것은 금지 대상으로 보고
번들에서 제외했다. 위성이 결측이면 모델이 위경도·고도·연중일만으로 예측한다."""),

md("""## 9. 성능

실제 채점과 같은 조건 — 학습은 검증 주 **이전 날짜만**, 미래를 전혀 보지 않는다.

검증 주간 2026-08-16~22 (668행):

| 구성 | 점수 | 기온 | 습도 |
|---|---:|---:|---:|
| 7년치 + 연도 피처 | 2.401 | | |
| 타깃별 연도 분리 | 2.130 | | |
| + 전국 편차 | 2.009 | | |
| + 최근접 8개 이웃 | 1.983 | 1.252 | 7.309 |
| **+ 방향별 이웃 (최종)** | **1.961** | **1.238** | **7.233** |

### 정보의 한계

같은 주간에 대해 "가능한 최선"을 직접 재봤다.

| 방법 | 점수 |
|---|---:|
| 14시 ASOS 그대로 베끼기 | 0.000 |
| 13시 ASOS 그대로 베끼기 | 1.345 |
| 다른 95개 지점의 실제 14시 값으로 완벽 공간보간 | 2.173 |
| **우리 모델 (위성만)** | **1.961** |

세 번째 줄이 중요하다. **다른 모든 지점의 정답을 알아도** 공간보간만으로는
2.173 이다. 위성만 쓰는 우리 모델이 그보다 낫다는 것은, 위성이 지점별
국지 상태에 대해 이웃 관측에 없는 정보를 실제로 담고 있다는 뜻이다."""),
]

nb = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.11")),
    nbformat=4, nbformat_minor=5)
pathlib.Path("notebooks/train_notebook.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1))
print(f"notebooks/train_notebook.ipynb 생성 — 셀 {len(cells)}개 "
      f"(코드 {sum(1 for c in cells if c['cell_type']=='code')}개)")
