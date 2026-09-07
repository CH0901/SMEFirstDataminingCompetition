import os, requests, time, sys
K=os.environ.get("KMA_API_KEY2", "")
U="https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/IR087/KO/data"
def fetch(date, timeout, tries):
    t0=time.time(); n=0
    for _ in range(tries):
        n+=1
        try:
            r=requests.get(U, params={"date":date,"authKey":K}, timeout=timeout)
            if r.status_code==200 and len(r.content)>100000: return True,n,time.time()-t0
        except Exception: pass
        time.sleep(0.3)
    return False,n,time.time()-t0
for to,tries in ((8,6),(5,8),(15,4)):
    ok=tot=att=0
    for i in range(1,11):
        s,n,el=fetch(f"202508{i:02d}0500",to,tries); ok+=s; tot+=el; att+=n
    print(f"  타임아웃 {to:2d}s x {tries}회:  성공 {ok}/10  평균 {tot/10:5.1f}초  평균시도 {att/10:.1f}회", flush=True)
