#!/bin/zsh
cd "$(dirname "$0")/.."; PY=./.venv/bin/python
echo "═══ 최종 학습 (타깃별 연도 피처) ═══"
$PY -m sme.model.train 2>&1 | grep -vE "NotOpenSSL|warnings.warn|RuntimeWarning|ret = a|return X @|^  warn|Performance|out\["
echo "\n═══ 피처 대조 ═══"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warn"
import pickle
b=pickle.load(open("model.pkl","rb"))
CH=b["channels"]; WS=b["win_stats"]
tags=[f"h{h:02d}" for h in b["hours_utc"]]+[f"p{h:02d}" for h in b["hours_utc_prev"]]
made={f"{ch}_{s}_{t}" for ch in CH for t in tags for s in WS}
made|={f"{ch}_{s}_{d}" for ch in CH for s in WS for d in ("d1h","dnt")}
made|={f"SWD_{s}_{t}" for t in tags for s in ("c","m9")}
made|={f"SWDmin_{t}" for t in tags}|{f"CLD_{t}" for t in tags}
made|={"LAT","LON","HT","DOY_SIN","DOY_COS","SEC_SATZEN","YEAR","STN_CAT"}
ok=True
for k,v in b["gbm_features_by_target"].items():
    m=set(v)-made
    print(f"  {k}: {len(v)}개 · {'OK' if not m else 'X '+str(sorted(m)[:3])}")
    ok &= not m
print(f"  YEAR 포함: TA14={'YEAR' in b['features_by_target']['TA14']} "
      f"HM14={'YEAR' in b['features_by_target']['HM14']}")
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
echo "\n═══ 실전 재현 종단검증 (2026-08-16~22) ═══"
$PY - <<'EOF' 2>&1 | grep -vE "NotOpenSSL|warn|Performance|out\["
import pickle, numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
FB={"TA14":feats,"HM14":[f for f in feats if f!="YEAR"]}
df["STN_CAT"]=df["STN"].astype("category")
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
res={}
for tgt in ("TA14","HM14"):
    F=FB[tgt]; s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    ms=[lgb.train(dict(T.PARAMS,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
        lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),T.ROUNDS)
        for sd in (42,7,2024)]
    res[tgt]=np.mean([m.predict(te[F+["STN_CAT"]]) for m in ms],axis=0)+lin.predict(te[F].astype(np.float64))
rt=np.sqrt(np.nanmean((res["TA14"]-te.TA14.to_numpy())**2))
rh=np.sqrt(np.nanmean((res["HM14"]-te.HM14.to_numpy())**2))
print(f"  Score {rt+0.1*rh:.3f}   TA {rt:.3f}   HM {rh:.3f}   편향 {np.nanmean(res['TA14']-te.TA14.to_numpy()):+.2f}")
print(f"  (경량 검증 2.143 · 시드3+1500라운드로 재확인)")
EOF
echo "\n═══ 완료 $(date +%H:%M) ═══"
