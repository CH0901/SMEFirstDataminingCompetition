#!/bin/zsh
cd "$(dirname "$0")/.."; PY=./.venv/bin/python
echo "═══ 최종 학습 (7년 + 연도 피처) ═══"
$PY -m sme.model.train 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn|Performance|out\["
echo "\n═══ 피처 대조 ═══"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warn"
import pickle
b=pickle.load(open("model.pkl","rb"))
F=set(b["gbm_features"]); CH=b["channels"]; WS=b["win_stats"]
tags=[f"h{h:02d}" for h in b["hours_utc"]]+[f"p{h:02d}" for h in b["hours_utc_prev"]]
made={f"{ch}_{s}_{t}" for ch in CH for t in tags for s in WS}
made|={f"{ch}_{s}_{d}" for ch in CH for s in WS for d in ("d1h","dnt")}
made|={f"SWD_{s}_{t}" for t in tags for s in ("c","m9")}
made|={f"SWDmin_{t}" for t in tags}|{f"CLD_{t}" for t in tags}
made|={"LAT","LON","HT","DOY_SIN","DOY_COS","SEC_SATZEN","YEAR","STN_CAT"}
m=F-made; print(f"  학습 {len(F)} / 추론 {len(made)} → {'OK' if not m else 'X '+str(sorted(m)[:4])}")
EOF
echo "\n═══ 노트북 갱신 ═══"
$PY - <<'EOF'
import json
nb=json.load(open("kaggle/submission_notebook_2.ipynb"))
i=[k for k,c in enumerate(nb["cells"]) if c["cell_type"]=="code"][1]
nb["cells"][i]["source"]=open("sme/submit/inference_cell.py").read().splitlines(keepends=True)
nb["cells"][i]["outputs"]=[]; nb["cells"][i]["execution_count"]=None
json.dump(nb,open("submission_notebook_filled.ipynb","w"),ensure_ascii=False,indent=1)
print("  갱신 완료")
EOF
echo "\n═══ 홀드아웃 종단검증 ═══"
$PY -m sme.validate.holdout 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn|Performance|out\[" | tail -2
cp model_holdout.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY2" 20250824 20250830 2>&1 | grep -E "최종 성공|예측 완료"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warn"
import pandas as pd, numpy as np, glob
s=pd.read_csv("/tmp/kgl/working/submission.csv")
a=pd.concat([pd.read_parquet(f) for f in glob.glob("data/asos_2025*.parquet")],ignore_index=True)
a["TM"]=pd.to_datetime(a.TM); a=a[a.TM.dt.hour==14]
a["ID"]=a.TM.dt.strftime("%Y%m%d")+"_"+a.STN.astype(int).astype(str)
m=s.merge(a[["ID","TA","HM"]],on="ID",suffixes=("_p","_t")).dropna()
rt=np.sqrt(((m.TA_p-m.TA_t)**2).mean()); rh=np.sqrt(((m.HM_p-m.HM_t)**2).mean())
print(f"\n  [최종] RMSE_TA {rt:.3f}  RMSE_HM {rh:.3f}  Score {rt+0.1*rh:.3f}   (직전 1.880)")
EOF
echo "\n═══ 리더보드 CSV ═══"
cp model.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY" 20260624 20260630 2>&1 | grep -E "최종 성공|예측 완료"
cp /tmp/kgl/working/submission.csv outputs/submission.csv
echo "  submission.csv 갱신"
echo "\n═══ 완료 $(date +%H:%M) ═══"
