import pickle, sys, numpy as np, pandas as pd
import eval14 as E, kma
df=E.assemble(list(kma.DEFAULT_CHANNELS))
W=pd.date_range("2026-06-24","2026-06-30")
d=df[df.date.isin(W)].copy()
kaggle={"20260624_100":(19.47,74.71),"20260624_101":(27.67,43.03),
        "20260624_102":(23.60,54.65),"20260624_104":(21.76,77.91),
        "20260624_105":(23.01,68.00)}
for path,name in [("/tmp/model_before823.pkl","8/23 추가 전 (535일)"),
                  ("model.pkl","8/23 추가 후 (536일, 현재)")]:
    b=pickle.load(open(path,"rb"))
    x=d.copy(); x["STN_CAT"]=pd.Categorical(x["STN"],categories=b["stn_categories"])
    o={}
    for t in ("TA14","HM14"):
        F=b["features_by_target"][t]; C=b["gbm_features_by_target"][t]
        for c in F:
            if c not in x.columns: x[c]=np.nan
        o[t]=np.mean([m.predict(x[C]) for m in b["models"][t]],0)+b["linears"][t].predict(x[F].astype(np.float64))
    r=pd.DataFrame({"ID":x.date.dt.strftime("%Y%m%d")+"_"+x.STN.astype(str),
                    "TA":o["TA14"],"HM":o["HM14"]})
    dif=[]
    for k,(ta,hm) in kaggle.items():
        w=r[r.ID==k]
        if len(w): dif.append(abs(float(w.TA.iloc[0])-ta))
    print(f"  {name:26s} 기온 최대차 {max(dif):.3f}   "
          + ("<= 일치!" if max(dif)<0.02 else ""))
