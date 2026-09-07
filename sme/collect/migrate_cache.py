"""sat_days 에 잘못 들어간 새 채널 파일을 sat_final 로 옮긴다."""
import glob, os, shutil, pandas as pd
moved=0
for p in sorted(glob.glob("data/sat_days/*.parquet")):
    cols = pd.read_parquet(p).columns
    if any(c.startswith("IR105_") for c in cols):
        dst = "data/sat_final/" + os.path.basename(p)
        if not os.path.exists(dst):
            shutil.move(p, dst); moved += 1
print(f"[migrate] {moved}개 이동 · sat_final {len(glob.glob('data/sat_final/*.parquet'))}일")
