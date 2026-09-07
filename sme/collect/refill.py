"""결측 부분만 다시 채운다.

collect.py 는 '하루 파일이 있으면 통째로 건너뛰기' 라서, 하루 안에서 일부
(시각 x 채널) 만 실패한 경우를 메우지 못한다. API 가 나빴던 시간대에 받은
파일들이 그 상태다. 여기서는 결측 칸만 골라 다시 요청한다.
"""
import glob, os, sys, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sme.core import kma_client
from sme.core import stations
from sme.core.grid import extract_windows, lonlat_to_rowcol, window_features

folder, half = sys.argv[1], int(sys.argv[2])
CH = list(kma_client.DEFAULT_CHANNELS)
stn = stations.load(); lon, lat = stn.LON.to_numpy(), stn.LAT.to_numpy()
cache = {}
files = sorted(glob.glob(f"{folder}/*.parquet"))
print(f"{len(files)}일 점검")

fixed = still = 0
for k, path in enumerate(files, 1):
    d = pd.read_parquet(path)
    # 결측인 (시각, 채널) 조합 찾기
    todo = []
    for h in sorted(d.time_utc.dt.hour.unique()):
        row = d[d.time_utc.dt.hour == h]
        for ch in CH:
            col = f"{ch}_m9"
            if col not in d.columns or row[col].isna().all():
                todo.append((h, ch))
    if not todo:
        continue
    day = pd.Timestamp(os.path.basename(path)[:8])
    blobs = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(kma_client.fetch_gk2a, ch,
                          (day + pd.Timedelta(hours=h)).strftime("%Y%m%d%H%M")): (h, ch)
                for h, ch in todo}
        for f in as_completed(futs):
            try:
                blobs[futs[f]] = f.result()
            except Exception:
                pass
    if not blobs:
        still += len(todo); continue

    for (h, ch), raw in blobs.items():
        img, meta = kma_client.read_gk2a(raw)
        key = (img.shape, float(meta["pixel_size"]))
        if key not in cache:
            cache[key] = lonlat_to_rowcol(lon, lat, meta)
        r, c = cache[key]
        feats = window_features(extract_windows(img, r, c, half=half), ch)
        m = d.time_utc.dt.hour == h
        for name, vals in feats.items():
            if name not in d.columns:
                d[name] = np.nan
            d.loc[m, name] = vals
    d.to_parquet(path, index=False)
    fixed += len(blobs); still += len(todo) - len(blobs)
    if k % 20 == 0:
        print(f"  {k}/{len(files)}일  복구 {fixed}칸  미복구 {still}칸", flush=True)

print(f"[refill] 복구 {fixed}칸 · 미복구 {still}칸")
