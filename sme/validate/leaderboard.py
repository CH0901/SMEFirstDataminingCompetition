"""캐글에서 나온 값이 맞는지 로컬 데이터로 대조한다 (API 호출 없음)."""
import pickle, numpy as np, pandas as pd
from sme.core import evaluation as E
from sme.core import kma_client
b=pickle.load(open("model.pkl","rb"))
df=E.assemble(list(kma_client.DEFAULT_CHANNELS))
W=pd.date_range("2026-06-24","2026-06-30")
d=df[df.date.isin(W)].copy()
d["STN_CAT"]=pd.Categorical(d["STN"], categories=b["stn_categories"])
out={}
for t in ("TA14","HM14"):
    F=b["features_by_target"][t]; C=b["gbm_features_by_target"][t]
    miss=[c for c in F if c not in d.columns]
    for c in miss: d[c]=np.nan
    p=np.mean([m.predict(d[C]) for m in b["models"][t]],0)+b["linears"][t].predict(d[F].astype(np.float64))
    out[t]=p
r=pd.DataFrame({"ID":d.date.dt.strftime("%Y%m%d")+"_"+d.STN.astype(str),
                "TA":out["TA14"].round(2),"HM":out["HM14"].round(2)})
print(f"로컬 계산 {len(r)}행 · TA {r.TA.min():.1f}~{r.TA.max():.1f} · HM {r.HM.min():.1f}~{r.HM.max():.1f}\n")
kaggle={"20260624_100":(19.47,74.71),"20260624_101":(27.67,43.03),
        "20260624_102":(23.60,54.65),"20260624_104":(21.76,77.91),
        "20260624_105":(23.01,68.00)}
print(f"{'ID':16s}{'캐글TA':>8s}{'로컬TA':>8s}{'차':>7s}   {'캐글HM':>8s}{'로컬HM':>8s}{'차':>7s}")
print("-"*64)
ok=True
for k,(ta,hm) in kaggle.items():
    row=r[r.ID==k]
    if row.empty: print(f"{k:16s}  로컬에 없음"); ok=False; continue
    lt,lh=float(row.TA.iloc[0]),float(row.HM.iloc[0])
    print(f"{k:16s}{ta:8.2f}{lt:8.2f}{ta-lt:+7.2f}   {hm:8.2f}{lh:8.2f}{hm-lh:+7.2f}")
    if abs(ta-lt)>0.02 or abs(hm-lh)>0.02: ok=False
print("\n판정:", "일치 — 캐글 실행 정상" if ok else "불일치 — 확인 필요")
