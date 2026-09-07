#!/bin/zsh
# 튜닝·피처 반영 → 재학습 → 피처 대조 → 홀드아웃 종단검증 → 노트북 갱신
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
F="grep -vE NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn"

echo "═══ 1. 최종 모델 학습 ═══"
$PY -m sme.model.train 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn"

echo "\n═══ 2. 학습/추론 피처 대조 ═══"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warnings"
import pickle
b=pickle.load(open("model.pkl","rb"))
F=set(b["gbm_features"]); CH=b["channels"]; WS=b["win_stats"]
tags=[f"h{h:02d}" for h in b["hours_utc"]]+[f"p{h:02d}" for h in b["hours_utc_prev"]]
made={f"{ch}_{s}_{t}" for ch in CH for t in tags for s in WS}
made|={f"{ch}_{s}_{d}" for ch in CH for s in WS for d in ("d1h","dnt")}
made|={f"SWD_{s}_{t}" for t in tags for s in ("c","m9")}
made|={f"SWDmin_{t}" for t in tags} | {f"CLD_{t}" for t in tags}
made|={"LAT","LON","HT","DOY_SIN","DOY_COS","SEC_SATZEN","STN_CAT"}
miss=F-made
print(f"  학습 {len(F)} / 추론 {len(made)} → {'OK 일치' if not miss else 'X 불일치 '+str(sorted(miss)[:5])}")
EOF

echo "\n═══ 3. 노트북 갱신 ═══"
$PY - <<'EOF'
import json
nb=json.load(open("kaggle/submission_notebook_2.ipynb"))
i=[k for k,c in enumerate(nb["cells"]) if c["cell_type"]=="code"][1]
nb["cells"][i]["source"]=open("sme/submit/inference_cell.py").read().splitlines(keepends=True)
nb["cells"][i]["outputs"]=[]; nb["cells"][i]["execution_count"]=None
json.dump(nb, open("submission_notebook_filled.ipynb","w"), ensure_ascii=False, indent=1)
print("  submission_notebook_filled.ipynb 갱신")
EOF

echo "\n═══ 4. 홀드아웃 모델 학습 ═══"
$PY -m sme.validate.holdout 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn"

echo "\n═══ 5. 종단검증 (실제 추론 노트북 실행) ═══"
cp model_holdout.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY2" 20250824 20250830 2>&1 | grep -E "차:|최종 성공|예측 완료"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warnings"
import pandas as pd, numpy as np, glob
s=pd.read_csv("/tmp/kgl/working/submission.csv")
a=pd.concat([pd.read_parquet(f) for f in glob.glob("data/asos_2025*.parquet")],ignore_index=True)
a["TM"]=pd.to_datetime(a.TM); a=a[a.TM.dt.hour==14]
a["ID"]=a.TM.dt.strftime("%Y%m%d")+"_"+a.STN.astype(int).astype(str)
m=s.merge(a[["ID","TA","HM"]],on="ID",suffixes=("_p","_t")).dropna()
rt=np.sqrt(((m.TA_p-m.TA_t)**2).mean()); rh=np.sqrt(((m.HM_p-m.HM_t)**2).mean())
print(f"\n  [최종] RMSE_TA {rt:.3f}  RMSE_HM {rh:.3f}  Score {rt+0.1*rh:.3f}")
print(f"  (직전 아키텍처 1.910)")
EOF

echo "\n═══ 6. 리더보드 제출용 CSV ═══"
cp model.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY" 20260624 20260630 2>&1 | grep -E "최종 성공|예측 완료"
cp /tmp/kgl/working/submission.csv outputs/submission.csv
echo "  submission.csv 갱신"
echo "\n═══ 완료 $(date +%H:%M) ═══"
