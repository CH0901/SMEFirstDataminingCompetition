# ═══════════════════════════════════════════════════════════════════
#  ↓↓↓ 자유 구현 영역 ↓↓↓
#
#  설계 원칙
#  1) 입력은 GK-2A LE1B 위성영상과 정적 지점정보(위경도·고도)뿐이다.
#     ASOS 는 학습 라벨로만 쓰였고, 추론에는 어떤 형태로도 들어가지 않는다
#     (기후 평년값·지점별 과거 평균기온 포함 — 운영진 답변에 따름).
#     위성이 결측이면 모델이 위경도·고도·연중일만으로 예측한다.
#     (규정: "위성 입력이 결측되거나 없는 경우에도 예측값이 존재해야 함")
#  2) 인과성 준수 — 대상 시각 14:00 KST = 05:00 UTC 까지의 영상만 쓴다.
#     당일 06 UTC 이후는 정답 시각 이후라 요청조차 하지 않는다.
#  3) 무작위성 없음 — 같은 입력이면 항상 같은 출력.
# ═══════════════════════════════════════════════════════════════════
import glob
import io
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# GK2A LE1B 파일은 확장자가 .nc 지만 실제 형식은 HDF5 다 (매직바이트 \x89HDF).
# 그래서 netCDF4 없이 h5py 만으로 그대로 읽을 수 있고, h5py 는 캐글 기본
# 환경에 들어 있다. 두 경로가 같은 배열을 준다는 것은 확인했다.
# 추가 설치에 기대지 않는다 — pip 는 네트워크가 막히면 그대로 실패한다.
import h5py

print(f"h5py {h5py.__version__} — GK2A(HDF5) 판독 준비 완료")

# ── 학습 산출물 로드 ──────────────────────────────────────────────
_hits = glob.glob("/kaggle/input/**/model.pkl", recursive=True)
if not _hits:
    sys.exit("model.pkl 을 찾을 수 없습니다. Add Input 을 확인하세요.")
BUNDLE = pickle.load(open(_hits[0], "rb"))

MODELS   = BUNDLE["models"]
LINEARS  = BUNDLE.get("linears", {})   # 외삽 담당 선형 성분
FEATURES = BUNDLE["features"]
GBM_FEATURES = BUNDLE.get("gbm_features", FEATURES)
# 연도 피처를 타깃마다 다르게 쓰므로 입력 집합도 타깃별로 갈린다
FEAT_BY_T = BUNDLE.get("features_by_target", {})
GBMF_BY_T = BUNDLE.get("gbm_features_by_target", {})
CHANNELS = BUNDLE["channels"]
WIN_STATS = BUNDLE["win_stats"]
STN_META = BUNDLE["stations"]
ROWS     = BUNDLE["pixel_rows"].astype(int)
COLS     = BUNDLE["pixel_cols"].astype(int)
H_TODAY  = BUNDLE["hours_utc"]       # [0, 2, 4, 5]  당일 05 UTC 이하
H_PREV   = BUNDLE["hours_utc_prev"]  # [18, 21]      전날
HALF     = 4                          # 9x9 창

# 번들의 지점 순서와 채점 지점 순서를 맞춘다
_order = {s: i for i, s in enumerate(STN_META["STN"].astype(int))}
_idx = np.array([_order[s] for s in STATIONS])
ROWS, COLS = ROWS[_idx], COLS[_idx]

print(f"모델 로드 완료 — 채널 {CHANNELS}, 피처 {len(FEATURES)}개")

# ── 위성 다운로드 (실패해도 죽지 않고, 오래 끌지도 않는다) ────────
#
# 기상청 API 허브는 실제로 몇 시간씩 죽는 일이 있다. 그때 재시도에 매달리면
# 노트북이 런타임 한도를 넘겨 채점 자체가 불가능해진다. 그래서
#   - 요청당 타임아웃을 짧게 잡고
#   - 전체 다운로드에 시간 예산을 두며
#   - 초반이 연속 실패하면 서버가 죽은 것으로 보고 즉시 포기한다.
# 포기해도 지점정보·연중일만으로 예측은 그대로 나간다.
HUB = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B"

# ── 타임아웃 설계 (실측 기반) ─────────────────────────────────────
# 이 API 는 실패할 때 HTTP 오류를 주지 않고 연결이 그대로 멈춘다.
# 성공하면 0.3초, 실패하면 영영 응답이 없다. 따라서 오래 기다리는 것은
# 순수한 낭비이고, 짧게 끊고 새 연결로 다시 붙는 편이 빠르면서 성공률도 높다.
#
#   실측 (10장 기준)
#     타임아웃  5s x 8회  ->  10/10 성공, 평균 15.2초
#     타임아웃  8s x 6회  ->   8/10 성공, 평균 24.4초
#     타임아웃 15s x 4회  ->   8/10 성공, 평균 34.0초
REQ_TIMEOUT = 5           # 초 — 정상 응답은 0.3초면 온다
MAX_TRIES = 10            # 무응답은 재연결로만 뚫린다
RETRY_SLEEP = 0.3         # 초 — 대기가 아니라 재시도가 이득이므로 짧게

BUDGET_SEC = 2400         # 초 — 다운로드 전체 예산 (40분)
BREAKER_N = 60            # 연속 실패가 이만큼이면 서버 장애로 판단
MAX_PASSES = 5            # 남은 것을 다시 훑는 최대 횟수

WORKERS = 3
MIN_INTERVAL = 0.2

_last = [0.0]
_t0 = time.monotonic()
_err_kinds = {}
_consec_fail = [0]
_dead = [False]


def _fetch(channel, stamp, tries=MAX_TRIES):
    """한 장 다운로드. 실패하면 None. 예산 초과·차단기 작동 시 즉시 None."""
    if _dead[0] or time.monotonic() - _t0 > BUDGET_SEC:
        return None
    url = f"{HUB}/{channel}/KO/data"
    for _ in range(tries):
        if _dead[0] or time.monotonic() - _t0 > BUDGET_SEC:
            return None
        try:
            wait = MIN_INTERVAL - (time.monotonic() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
            r = requests.get(url, params={"date": stamp, "authKey": API_KEY},
                             timeout=REQ_TIMEOUT)
            if r.status_code == 200 and len(r.content) > 100_000:
                _consec_fail[0] = 0   # 하나라도 성공하면 장애 판단을 되돌린다
                return r.content
            if len(r.content) < 500 and b'"status"' in r.content:
                # 권한 없음·한도 초과는 재시도해도 소용없다
                break
        except Exception as e:
            # 조용히 삼키지 않는다. 종류별로 세어 마지막에 보고한다.
            _err_kinds[type(e).__name__] = _err_kinds.get(type(e).__name__, 0) + 1
        time.sleep(RETRY_SLEEP)

    _consec_fail[0] += 1
    if _consec_fail[0] >= BREAKER_N and not _dead[0]:
        _dead[0] = True
        print(f"[차단기] 연속 {BREAKER_N}회 실패 — API 장애로 판단, "
              f"위성 수집을 중단하고 지점정보만으로 진행합니다.")
    return None


def _read_image(raw):
    """GK2A 파일 바이트 -> 화소 배열. 디스크에 쓰지 않고 메모리에서 연다."""
    try:
        with h5py.File(io.BytesIO(raw), "r") as f:
            return np.asarray(f["image_pixel_values"][:])
    except Exception as e:
        _err_kinds[f"read:{type(e).__name__}"] = (
            _err_kinds.get(f"read:{type(e).__name__}", 0) + 1)
        return None


def _windows(img):
    """지점별 9x9 창 통계. 격자 밖이면 NaN."""
    n, w = len(ROWS), 2 * HALF + 1
    out = np.full((n, w, w), np.nan, dtype=np.float32)
    H, W = img.shape
    for i in range(n):
        r0, c0 = ROWS[i] - HALF, COLS[i] - HALF
        if r0 >= 0 and c0 >= 0 and r0 + w <= H and c0 + w <= W:
            out[i] = img[r0:r0 + w, c0:c0 + w]
    flat = out.reshape(n, -1)
    c = w // 2
    with np.errstate(all="ignore"):
        return {"c": out[:, c, c],
                "m9": np.nanmean(flat, axis=1),
                "sd9": np.nanstd(flat, axis=1),
                "min9": np.nanmin(flat, axis=1),
                "max9": np.nanmax(flat, axis=1)}


# ── 관측 시각 (KST) ──────────────────────────────────────────────
# 학습 때 쓴 시각은 UTC 기준이지만, 대회 헬퍼 to_api_datetime() 은 KST 를
# 받아 UTC 로 바꿔준다. KST 로 환산하면 여섯 시각 모두 '대상일 당일' 이다.
#
#   학습 태그   UTC            KST(대상일)
#   h00         00             09:00
#   h02         02             11:00
#   h04         04             13:00
#   h05         05             14:00   <- 대상 시각과 동일
#   p18         전날 18        03:00
#   p21         전날 21        06:00
#
# 피처 이름(태그)은 학습 때와 똑같이 유지해야 하므로 그대로 쓴다.
_OBS_KST = ([(f"h{h:02d}", h + 9) for h in H_TODAY]
            + [(f"p{h:02d}", h + 9 - 24) for h in H_PREV])

_jobs = []
for target_dt in PRED_DATES:
    day = pd.Timestamp(target_dt).normalize()
    for tag, kst_hour in _OBS_KST:
        obs_dt = day + pd.Timedelta(hours=kst_hour)
        # 헬퍼가 obs > target 이면 예외를 던져 인과성을 보증한다
        api_date = to_api_datetime(obs_dt, target_dt)
        for ch in CHANNELS:
            _jobs.append((target_dt, api_date, tag, ch))

print(f"위성 요청 {len(_jobs)}건 ({len(PRED_DATES)}일 x "
      f"{len(_OBS_KST)}시각 x {len(CHANNELS)}채널)")
print(f"  관측 시각(KST): {[h for _, h in _OBS_KST]}시  -> 대상 14시 이하 확인됨")

def _download(jobs):
    """목록을 받아 성공한 것만 _stats 에 채우고, 실패 목록을 돌려준다."""
    left = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_fetch, ch, stamp): (d, stamp, tag, ch)
                for (d, stamp, tag, ch) in jobs}
        for fut in as_completed(futs):
            d, stamp, tag, ch = futs[fut]
            raw = fut.result()
            img = _read_image(raw) if raw is not None else None
            if img is None:
                left.append((d, stamp, tag, ch))
            else:
                _stats[(d, tag, ch)] = _windows(img)
    return left


_stats = {}   # (date, tag, channel) -> {stat: array}
_left = _download(_jobs)
print(f"1차: 성공 {len(_jobs)-len(_left)} / 실패 {len(_left)}")

# 남은 것을 여러 번 더 훑는다. 실패는 대부분 일시적 무응답이라 다시
# 붙으면 뚫린다. 예산이 남아 있고 진전이 있는 동안만 반복하며,
# 차단기가 걸렸다면 서버가 정말 죽은 것이므로 멈춘다.
_round = 1
while (_left and not _dead[0] and _round < MAX_PASSES
       and time.monotonic() - _t0 < BUDGET_SEC):
    _round += 1
    _before = len(_left)
    time.sleep(3)
    _left = _download(_left)
    print(f"{_round}차 재시도 후: 실패 {len(_left)}")
    if len(_left) == _before:      # 더 이상 줄지 않으면 그만
        break
print(f"다운로드 최종 성공 {len(_jobs)-len(_left)} / {len(_jobs)}")
if _err_kinds:
    print("  발생한 예외:", ", ".join(f"{k} x{v}" for k, v in
                                   sorted(_err_kinds.items(), key=lambda x: -x[1])))

# ── 피처 조립 (학습 때와 정확히 같은 이름·정의) ───────────────────
_TAGS = [f"h{h:02d}" for h in H_TODAY] + [f"p{h:02d}" for h in H_PREV]
_nan = lambda: np.full(len(STATIONS), np.nan)
rows = []
for d in PRED_DATES:
    rec = {"STN": np.asarray(STATIONS)}
    for ch in CHANNELS:
        for tag in _TAGS:
            got = _stats.get((d, tag, ch))
            for s in WIN_STATS:
                rec[f"{ch}_{s}_{tag}"] = got[s] if got else _nan()
        for s in WIN_STATS:
            base = f"{ch}_{s}"
            rec[f"{base}_d1h"] = rec[f"{base}_h05"] - rec[f"{base}_h04"]
            rec[f"{base}_dnt"] = rec[f"{base}_h05"] - rec[f"{base}_p21"]
    # 분리대기창 보정항 (KMA LST 알고리즘의 T13 - T15)
    if "IR105" in CHANNELS and "IR123" in CHANNELS:
        for tag in _TAGS:
            for s in ("c", "m9"):
                a, b = f"IR105_{s}_{tag}", f"IR123_{s}_{tag}"
                if a in rec and b in rec:
                    rec[f"SWD_{s}_{tag}"] = rec[a] - rec[b]
            # 청천 화소 기준 (DN 최솟값 = 물리적으로 가장 뜨거운 화소)
            a, b = f"IR105_min9_{tag}", f"IR123_min9_{tag}"
            if a in rec and b in rec:
                rec[f"SWDmin_{tag}"] = rec[a] - rec[b]
            # 구름량 대리 (창 안의 최대-최소)
            a, b = f"IR105_max9_{tag}", f"IR105_min9_{tag}"
            if a in rec and b in rec:
                rec[f"CLD_{tag}"] = rec[a] - rec[b]
    f = pd.DataFrame(rec)
    f["date"] = pd.Timestamp(d).normalize()   # 피처는 날짜 단위
    rows.append(f)

X = pd.concat(rows, ignore_index=True)
X = X.merge(STN_META[["STN", "LAT", "LON", "HT"]], on="STN", how="left")
X["DOY"] = X["date"].dt.dayofyear
X["DOY_SIN"] = np.sin(2 * np.pi * X["DOY"] / 365.25)
X["DOY_COS"] = np.cos(2 * np.pi * X["DOY"] / 365.25)
X["YEAR"] = X["date"].dt.year.astype(float)   # 온난화 추세 외삽용 (규칙 2.1③)

# 전국 대비 편차 — 그날 한반도 전체 대비 이 지점이 얼마나 특이한가.
# 학습 때와 같은 정의: 같은 날짜 96개 지점의 평균을 빼준다.
_nat_keys = sorted({c[4:] for c in
                    set(FEAT_BY_T.get("TA14", [])) | set(GBMF_BY_T.get("TA14", []))
                    if c.startswith("ANO_")})
if _nat_keys:
    _have = [c for c in _nat_keys if c in X.columns]
    _mean = X.groupby("date")[_have].transform("mean")
    X = pd.concat([X, pd.DataFrame({f"ANO_{c}": X[c] - _mean[c] for c in _have},
                                   index=X.index)], axis=1)

# 최근접 8개 지점의 위성 평균 (대상 시각). 학습 때와 같은 정의.
_nb_need = sorted({c[3:] for c in set(FEAT_BY_T.get("TA14", []))
                   if c.startswith("NB_")})
if _nb_need:
    _s = STN_META.set_index("STN")
    _la = _s.loc[STATIONS, "LAT"].to_numpy(); _lo = _s.loc[STATIONS, "LON"].to_numpy()
    _d = np.sqrt(((_la[:, None] - _la) * 111) ** 2
                 + ((_lo[:, None] - _lo) * 111 * np.cos(np.radians(_la[:, None]))) ** 2)
    np.fill_diagonal(_d, 1e9)
    _near = {STATIONS[i]: [STATIONS[j] for j in np.argsort(_d[i])[:8]]
             for i in range(len(STATIONS))}
    for _c in _nb_need:
        if _c not in X.columns:
            X[f"NB_{_c}"] = np.nan; continue
        _p = X.pivot_table(index="date", columns="STN", values=_c)
        _m = pd.DataFrame({s_: _p[[x for x in _near[s_] if x in _p.columns]].mean(axis=1)
                           for s_ in STATIONS if s_ in _p.columns})
        _st = _m.stack().rename(f"NB_{_c}").reset_index()
        _st.columns = ["date", "STN", f"NB_{_c}"]
        X = X.merge(_st, on=["date", "STN"], how="left")

    # 방향별 이웃 (동/서/남/북 각 3개) — 학습 때와 같은 정의
    _dy = (_la[:, None] - _la) * 111
    _dx = (_lo[:, None] - _lo) * 111 * np.cos(np.radians(_la[:, None]))
    _sect = {"W": (_dx < 0) & (np.abs(_dx) > np.abs(_dy)),
             "E": (_dx > 0) & (np.abs(_dx) > np.abs(_dy)),
             "S": (_dy < 0) & (np.abs(_dy) >= np.abs(_dx)),
             "N": (_dy > 0) & (np.abs(_dy) >= np.abs(_dx))}
    for _tag, _mask in _sect.items():
        _grp = {}
        for _i in range(len(STATIONS)):
            _cand = np.where(_mask[_i])[0]
            _grp[STATIONS[_i]] = ([STATIONS[j] for j in _cand[np.argsort(_d[_i, _cand])[:3]]]
                                  if len(_cand) else [])
        for _c in _nb_need:
            if _c not in X.columns:
                X[f"D{_tag}_{_c}"] = np.nan; continue
            _p = X.pivot_table(index="date", columns="STN", values=_c)
            _m = pd.DataFrame({
                s_: (_p[[x for x in _grp[s_] if x in _p.columns]].mean(axis=1)
                     if _grp.get(s_) else np.nan)
                for s_ in STATIONS if s_ in _p.columns})
            _st = _m.stack().rename(f"D{_tag}_{_c}").reset_index()
            _st.columns = ["date", "STN", f"D{_tag}_{_c}"]
            X = X.merge(_st, on=["date", "STN"], how="left")
    for _c in _nb_need:
        if f"DE_{_c}" in X and f"DW_{_c}" in X:
            X[f"dEW_{_c}"] = X[f"DE_{_c}"] - X[f"DW_{_c}"]
        if f"DN_{_c}" in X and f"DS_{_c}" in X:
            X[f"dNS_{_c}"] = X[f"DS_{_c}"] - X[f"DN_{_c}"]

# 위성 천정각의 경로 길이 효과 (물리식에 sec(theta)-1 로 들어간다).
# GK-2A 는 정지위성이라 지점별 상수이며 번들에 실린 값을 그대로 쓴다.
if "SEC_SATZEN" in FEATURES:
    _sz = BUNDLE["sat_zenith"]
    X = X.merge(pd.DataFrame({"STN": STN_META["STN"].astype(int).to_numpy(),
                              "SEC_SATZEN": 1.0 / np.cos(np.radians(_sz)) - 1.0}),
                on="STN", how="left")

# 지점번호를 학습 때와 같은 범주 목록으로 맞춘다 (순서가 다르면 코드가 어긋난다)
if "STN_CAT" in GBM_FEATURES:
    X["STN_CAT"] = pd.Categorical(X["STN"].astype(int),
                                  categories=BUNDLE["stn_categories"])

_need = set(FEATURES) | set(GBM_FEATURES)
for _v in list(FEAT_BY_T.values()) + list(GBMF_BY_T.values()):
    _need |= set(_v)
for c in _need:                              # 학습 때 없던 컬럼 방어
    if c not in X.columns:
        X[c] = np.nan

# ── 예측 ──────────────────────────────────────────────────────────
# 시드별 모델의 평균을 쓴다 (학습 때와 동일).
def _predict(key):
    lf = FEAT_BY_T.get(key, FEATURES)
    gf = GBMF_BY_T.get(key, GBM_FEATURES)
    Z = X[lf].astype(np.float64)                 # 선형 성분 입력
    G = X[gf]                                    # GBM 입력 (지점번호 포함)
    ms = MODELS[key]
    ms = ms if isinstance(ms, list) else [ms]
    p = np.mean([m.predict(G) for m in ms], axis=0)
    if key in LINEARS:                 # 잔차 학습이므로 선형분을 되더한다
        p = p + LINEARS[key].predict(Z)
    return p

ta = _predict("TA14")
hm = _predict("HM14")

# 위성이 결측인 행도 모델이 그대로 처리한다. LightGBM 은 결측을 학습 때
# 배운 방향으로 보내므로, 위성이 전혀 없으면 위경도·고도·연중일만으로
# 예측이 나온다. 기후 평년값 같은 ASOS 파생 자료로 대체하지 않는다
# (운영진: ASOS 는 학습 라벨로만 사용 가능).
_sat_cols = [c for c in FEATURES if c.split("_")[0] in CHANNELS]
_no_sat = int(X[_sat_cols].isna().all(axis=1).sum())
if _no_sat:
    print(f"위성 전무 {_no_sat}행 — 지점정보·연중일만으로 예측합니다")

# 물리적으로 불가능한 값 차단. 결측 제출은 실격이므로 NaN 도 막는다.
# 대체값은 예측 자체의 중앙값이라 외부 자료가 아니다.
ta = np.clip(np.nan_to_num(ta, nan=float(np.nanmedian(ta))), -30, 50)
hm = np.clip(np.nan_to_num(hm, nan=float(np.nanmedian(hm))), 0, 100)

pred = pd.DataFrame({
    "Date":   X["date"].dt.strftime("%Y%m%d").astype(int),
    "STN_ID": X["STN"].astype(int),
    "TA":     ta,
    "HM":     hm,
})
print(f"예측 완료 {len(pred)}행  TA {pred.TA.min():.1f}~{pred.TA.max():.1f}  "
      f"HM {pred.HM.min():.1f}~{pred.HM.max():.1f}")
