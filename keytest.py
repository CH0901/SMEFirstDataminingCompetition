"""두 키를 채점 노트북이 실제로 쓰는 호출로 시험한다."""
import os, time, urllib.request, urllib.parse
KEYS={"키1": os.environ.get("KMA_API_KEY", ""),
      "키2": os.environ.get("KMA_API_KEY2", "")}
GK2A="https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{ch}/KO/data"
ASOS="https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"

def get(url, params, timeout=20):
    q=url+"?"+urllib.parse.urlencode(params)
    t=time.time()
    try:
        with urllib.request.urlopen(q, timeout=timeout) as r:
            b=r.read(); return r.status, len(b), time.time()-t, b[:120]
    except Exception as e:
        return None, 0, time.time()-t, str(e)[:120].encode()

for name,k in KEYS.items():
    print(f"\n{name}")
    for ch in ("IR105","IR123","SW038"):
        s,n,el,head=get(GK2A.format(ch=ch), dict(date="202608230500", authKey=k))
        ok = s==200 and n>100_000
        print(f"  GK2A {ch:6s} {'정상' if ok else '실패':4s}  {n/1e6:5.2f}MB  {el:4.1f}초"
              + ("" if ok else f"  <- {head[:80]!r}"))
    s,n,el,head=get(ASOS, dict(tm="202608230900", stn="0", help="0", authKey=k))
    body=head.decode("euc-kr","ignore")
    ok = s==200 and "#" in body and "ERR" not in body.upper()
    print(f"  ASOS 지상관측  {'정상' if ok else '실패':4s}  {n/1e3:5.1f}KB  {el:4.1f}초"
          + ("" if ok else f"  <- {body[:80]!r}"))
