#!/bin/zsh
cd "$(dirname "$0")/.."; PY=./.venv/bin/python
echo "═══ 최종 학습 ═══"
$PY -m sme.model.train 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn|Performance|out\[" | head -6
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
echo "\n═══ 실전 재현 검증 (정식 설정) ═══"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warn|Performance|out\[|df\["
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
ex=[c for c in df.columns if c.startswith(("ANO_","NB_"))]
FB={"TA14":feats+ex,"HM14":[f for f in feats if f!="YEAR"]}
df["STN_CAT"]=df["STN"].astype("category")
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
r={}
for tgt in ("TA14","HM14"):
    F=FB[tgt]; s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    ms=[lgb.train(dict(T.PARAMS,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
        lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),T.ROUNDS)
        for sd in (42,7,2024)]
    r[tgt]=np.mean([m.predict(te[F+["STN_CAT"]]) for m in ms],axis=0)+lin.predict(te[F].astype(np.float64))
rt=np.sqrt(np.nanmean((r["TA14"]-te.TA14.to_numpy())**2))
rh=np.sqrt(np.nanmean((r["HM14"]-te.HM14.to_numpy())**2))
print(f"  Score {rt+0.1*rh:.3f}   TA {rt:.3f}   HM {rh:.3f}   (직전 2.009)")
EOF
echo "\n═══ 리더보드 CSV ═══"
cp model.pkl /tmp/kgl/input/model/model.pkl
rm -f /tmp/kgl/working/submission.csv
$PY -u -m sme.submit.rehearsal "$KMA_API_KEY" 20260624 20260630 2>&1 | grep -E "최종 성공|예측 완료"
cp /tmp/kgl/working/submission.csv outputs/submission.csv 2>/dev/null && echo "  submission.csv 갱신"
echo "\n═══ 완료 $(date +%H:%M) ═══"
