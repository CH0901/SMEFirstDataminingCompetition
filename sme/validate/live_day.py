"""8/23 실전 검증 — 학습에 없는 하루로 파이프라인 전체를 채점한다.

8/23 은 채점 기간(8/24~30)이 아니고 이미 지난 날이라 정답 조회가 자유롭다.
학습 데이터는 8/22 까지라 이 하루는 완전한 out-of-sample 이다.
추론 노트북이 실제 API 로 뽑은 예측을 그 정답과 대조한다.
"""
import numpy as np, pandas as pd, kma
sub=pd.read_csv("/tmp/kgl/working/submission.csv")
sub["date"]=sub.ID.str.split("_").str[0]
sub["STN"]=sub.ID.str.split("_").str[1].astype(int)
p=sub[sub.date=="20260823"]

obs=kma.fetch_asos_range("20260823","20260823")
obs["TM"]=pd.to_datetime(obs["TM"])
o=obs[obs.TM.dt.hour==14][["STN","TA","HM"]]
o=o[(o.TA>-90)&(o.HM>-9)]
m=p.merge(o,on="STN",suffixes=("_pred","_obs"))
print(f"대조 가능 지점 {len(m)}개\n")

rt=float(np.sqrt(np.mean((m.TA_pred-m.TA_obs)**2)))
rh=float(np.sqrt(np.mean((m.HM_pred-m.HM_obs)**2)))
bt=float((m.TA_pred-m.TA_obs).mean())
print(f"  점수  {rt+0.1*rh:.3f}")
print(f"  기온  RMSE {rt:.3f}   편향 {bt:+.2f}도")
print(f"  습도  RMSE {rh:.3f}\n")

CITY={108:"서울",143:"대구",159:"부산",184:"제주",112:"인천",133:"대전",
      156:"광주",105:"강릉",101:"춘천",146:"전주"}
print(f"{'도시':6s}{'예측':>7s}{'실제':>7s}{'오차':>7s}")
print("-"*27)
for s,n in CITY.items():
    r=m[m.STN==s]
    if len(r): 
        r=r.iloc[0]
        print(f"{n:6s}{r.TA_pred:7.1f}{r.TA_obs:7.1f}{r.TA_pred-r.TA_obs:+7.1f}")

w=m.assign(e=(m.TA_pred-m.TA_obs).abs()).nlargest(5,"e")
print(f"\n오차 큰 지점 5개")
for _,r in w.iterrows():
    print(f"  {int(r.STN):4d}  예측 {r.TA_pred:5.1f}  실제 {r.TA_obs:5.1f}  {r.TA_pred-r.TA_obs:+5.1f}")
