#!/bin/zsh
cd "$(dirname "$0")/.."; PY=./.venv/bin/python
echo "═══ 2020 라벨 보충 ═══"
mv data/asos14_2020.parquet data/asos14_2020_partial.parquet 2>/dev/null
$PY -m sme.collect.labels 20200601 20200930 data/asos14_2020.parquet 2>&1 | grep -E "^\[labels\]"
echo "\n═══ 7년치 재학습 ═══"
$PY -m sme.model.train 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn"
echo "\n═══ 홀드아웃 종단검증 ═══"
$PY -m sme.validate.holdout 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn"
cp model_holdout.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY2" 20250824 20250830 2>&1 | grep -E "최종 성공|예측 완료"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warnings"
import pandas as pd, numpy as np, glob
s=pd.read_csv("/tmp/kgl/working/submission.csv")
a=pd.concat([pd.read_parquet(f) for f in glob.glob("data/asos_2025*.parquet")],ignore_index=True)
a["TM"]=pd.to_datetime(a.TM); a=a[a.TM.dt.hour==14]
a["ID"]=a.TM.dt.strftime("%Y%m%d")+"_"+a.STN.astype(int).astype(str)
m=s.merge(a[["ID","TA","HM"]],on="ID",suffixes=("_p","_t")).dropna()
rt=np.sqrt(((m.TA_p-m.TA_t)**2).mean()); rh=np.sqrt(((m.HM_p-m.HM_t)**2).mean())
print(f"\n  [7년치] RMSE_TA {rt:.3f}  RMSE_HM {rh:.3f}  Score {rt+0.1*rh:.3f}")
print(f"  (3년치 1.880)")
EOF
echo "\n═══ 노트북·CSV 갱신 ═══"
$PY - <<'EOF'
import json
nb=json.load(open("kaggle/submission_notebook_2.ipynb"))
i=[k for k,c in enumerate(nb["cells"]) if c["cell_type"]=="code"][1]
nb["cells"][i]["source"]=open("sme/submit/inference_cell.py").read().splitlines(keepends=True)
nb["cells"][i]["outputs"]=[]; nb["cells"][i]["execution_count"]=None
json.dump(nb, open("submission_notebook_filled.ipynb","w"), ensure_ascii=False, indent=1)
print("  노트북 갱신")
EOF
cp model.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY" 20260624 20260630 2>&1 | grep -E "최종 성공|예측 완료"
cp /tmp/kgl/working/submission.csv outputs/submission.csv
echo "  submission.csv 갱신"
echo "\n═══ 완료 $(date +%H:%M) ═══"
