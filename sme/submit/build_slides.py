"""정성평가 발표자료 생성."""
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

INK, SOFT, MUTED = C(0x13,0x20,0x2A), C(0x3C,0x4E,0x5A), C(0x6A,0x7B,0x85)
ACC, ADOPT, REJECT = C(0x0C,0x69,0x79), C(0x16,0x6B,0x51), C(0x9C,0x43,0x27)
BG, PANEL, RULE = C(0xFF,0xFF,0xFF), C(0xF1,0xF5,0xF7), C(0xDC,0xE4,0xE8)
KR, MONO = "Apple SD Gothic Neo", "Menlo"
W, H = 13.333, 7.5

prs = Presentation(); prs.slide_width, prs.slide_height = I(W), I(H)

def slide():
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    return s

def box(s, x, y, w, h, fill=None, line=None, lw=1):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(x), I(y), I(w), I(h))
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = fill
    else: sh.fill.background()
    if line: sh.line.color.rgb = line; sh.line.width = Pt(lw)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    return sh

def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.25):
    tb = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, r in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = r.get("ls", spacing)
        if r.get("space_before"): p.space_before = Pt(r["space_before"])
        run = p.add_run(); run.text = r["t"]
        f = run.font
        f.name = r.get("font", KR); f.size = Pt(r.get("sz", 16))
        f.bold = r.get("b", False); f.color.rgb = r.get("c", INK)
    return tb

def header(s, eyebrow, title, sub=None):
    text(s, .95, .62, 11.5, .3, [{"t":eyebrow,"sz":11,"b":True,"c":ACC,"font":MONO}])
    text(s, .95, 1.0, 11.5, .8, [{"t":title,"sz":32,"b":True,"c":INK,"ls":1.15}])
    box(s, .95, 1.92, 1.5, .035, fill=ACC)
    if sub:
        text(s, .95, 2.18, 10.8, .6, [{"t":sub,"sz":15,"c":SOFT,"ls":1.4}])

def table(s, x, y, w, rows, colw, header_row=True, rh=.42, fs=13):
    """rows: [[cell,...],...]  colw: 비율 리스트"""
    tot = sum(colw); cx = [x]
    for c in colw[:-1]: cx.append(cx[-1] + w*c/tot)
    cw = [w*c/tot for c in colw]
    yy = y
    for ri, row in enumerate(rows):
        if header_row and ri == 0:
            box(s, x, yy+rh-.03, w, .022, fill=RULE)
        elif ri % 2 == 1:
            box(s, x-.12, yy-.04, w+.24, rh, fill=PANEL)
        for ci, cell in enumerate(row):
            if isinstance(cell, dict): txt, opt = cell["t"], cell
            else: txt, opt = cell, {}
            al = PP_ALIGN.RIGHT if opt.get("r") else PP_ALIGN.LEFT
            fn = MONO if opt.get("m") or opt.get("r") else KR
            text(s, cx[ci], yy, cw[ci]-.1, rh,
                 [{"t":txt, "sz":opt.get("sz", 11 if (header_row and ri==0) else fs),
                   "b":opt.get("b", header_row and ri==0),
                   "c":opt.get("c", MUTED if (header_row and ri==0) else INK),
                   "font":MONO if (header_row and ri==0) else fn, "ls":1.0}],
                 align=al, anchor=MSO_ANCHOR.MIDDLE)
        yy += rh
    return yy

def bar(s, x, y, w, frac, color=ACC, h=.14):
    box(s, x, y, w, h, fill=RULE)
    if frac > 0: box(s, x, y, max(w*frac, .04), h, fill=color)

# ══════════════════ 1. 표지
s = slide()
box(s, 0, 0, .34, H, fill=ACC)
text(s, 1.3, 2.15, 10.5, .4, [{"t":"제1회 시스템경영공학과 데이터분석 경진대회","sz":13,"b":True,"c":MUTED,"font":MONO}])
text(s, 1.3, 2.72, 11, 1.5, [{"t":"위성 영상만으로\n지상 기온·습도 예측하기","sz":44,"b":True,"c":INK,"ls":1.18}])
box(s, 1.3, 4.5, 1.8, .04, fill=ACC)
text(s, 1.3, 4.9, 10, .9, [
  {"t":"천리안 2A호 적외 영상으로 전국 96개 관측소의 14시 기온·습도를 예측했다.","sz":16,"c":SOFT,"ls":1.5},
  {"t":"지상관측은 정답으로만 사용할 수 있었다.","sz":16,"c":SOFT,"ls":1.5}])
for i,(lab,val,col) in enumerate([("최종 점수","2.27",ADOPT),("순위","1위",ADOPT),("2위와 차이","0.13",INK)]):
    bx = 1.3 + i*2.5
    text(s, bx, 6.15, 2.2, .25, [{"t":lab,"sz":10,"b":True,"c":MUTED,"font":MONO}])
    text(s, bx, 6.45, 2.2, .5, [{"t":val,"sz":26,"b":True,"c":col,"font":MONO}])

# ══════════════════ 2. 문제 정의
s = slide()
header(s, "문제 정의", "무엇을 맞혀야 했는가")
rows = [["항목","내용"],
        ["예측 대상","전국 96개 관측소의 매일 14시 기온(°C)·상대습도(%)"],
        ["채점식","기온 RMSE + 0.1 × 습도 RMSE  →  기온이 약 3분의 2"],
        ["사용 가능","천리안 2A호 위성 영상, 관측소 위경도·고도, 날짜"],
        ["사용 금지","지상관측 자료를 입력으로 사용 (정답 라벨로만 허용)"],
        ["시각 제약","대상 시각인 14시 이전에 관측된 영상만"]]
table(s, .95, 2.75, 11.4, rows, [1.6, 9.8], rh=.62, fs=14)
box(s, .95, 6.25, 11.4, .82, fill=PANEL)
text(s, 1.25, 6.45, 10.8, .5, [{"t":"핵심 난점 — 추론 시점에 지상 상태를 알려주는 자료가 하나도 없다. 위성이 보는 것은 지표면과 구름의 복사이지, 지상 2m 높이의 공기가 아니다.","sz":14,"c":SOFT,"ls":1.4}])

# ══════════════════ 3. 접근 — 분산 분해
s = slide()
header(s, "접근 방식", "입력 변수를 늘리기 전에 예측 대상을 분해했다",
       "기온의 변동이 어디서 오는지 모르면 어떤 변수가 필요한지도 알 수 없다고 판단했다.")
rows = [["변동 요인","분산","비중","무엇으로 설명되는가"],
        ["날짜 간 — 그날의 기단",{"t":"9.31","r":1},{"t":"66%","r":1},"계절 · 연도 · 전국 위성 패턴"],
        ["관측소 간 — 고정 특성",{"t":"1.31","r":1},{"t":"9%","r":1},"위경도 · 고도 · 관측소 번호"],
        ["날짜 × 관측소 상호작용",{"t":"3.61","r":1},{"t":"25%","r":1},"가장 어려운 부분"]]
table(s, .95, 3.15, 11.4, rows, [3.4, 1.2, 1.1, 5.7], rh=.58, fs=15)
for i,f in enumerate([.66,.09,.25]):
    bar(s, 5.55, 3.88+i*.58, 1.0, f, ACC if i!=2 else REJECT, h=.12)
box(s, .95, 5.65, 11.4, 1.1, fill=PANEL)
text(s, 1.25, 5.9, 10.8, .7, [
  {"t":"3분의 2가 “오늘이 더운 날인가”였다.","sz":17,"b":True,"c":INK,"ls":1.3},
  {"t":"그 정보는 한 관측소 위 화소의 밝기가 아니라 전국 패턴에 들어 있다. 이 관찰이 이후 변수 설계를 전부 결정했다.","sz":14,"c":SOFT,"ls":1.4}])

# ══════════════════ 4. 핵심 변수 ①
s = slide()
header(s, "적용 ①", "절댓값이 아니라 전국 평균 대비 편차")
y = 2.85
for lab, txt, col in [
  ("가설", "변동의 66%가 날짜 요인이라면, 모델이 매일 “오늘 전국이 얼마나 더운가”를 다시 추정하느라 용량을 낭비하고 있을 것이다. 그 성분을 분리해 주면 남은 용량을 관측소별 차이에 쓸 수 있다.", SOFT),
  ("근거", "같은 날 96개 관측소는 같은 기단을 공유한다. 따라서 (그 관측소 값 − 그날 전국 평균)은 날짜 요인이 제거된 국지 신호가 된다.", SOFT),
  ("예상", "모든 위성 변수에 이 변환을 적용하면 기온 오차가 뚜렷하게 줄 것이다.", SOFT)]:
    text(s, .95, y, 1.1, .3, [{"t":lab,"sz":11,"b":True,"c":MUTED,"font":MONO}])
    text(s, 2.25, y, 10.1, .8, [{"t":txt,"sz":15,"c":col,"ls":1.45}])
    y += .95
box(s, .95, 5.85, 11.4, 1.1, fill=C(0xE3,0xF1,0xEA))
text(s, 1.25, 6.05, 4.5, .3, [{"t":"결과","sz":11,"b":True,"c":ADOPT,"font":MONO}])
text(s, 1.25, 6.35, 5.5, .5, [{"t":"점수 2.130 → 2.009","sz":22,"b":True,"c":ADOPT,"font":MONO}])
text(s, 6.9, 6.15, 5.3, .7, [{"t":"사후 제거 실험에서도 이 변수군을 빼면 기온 오차가 0.112 나빠져, 위성 관련 변수 중 가장 큰 기여였다.","sz":13,"c":SOFT,"ls":1.4}])

# ══════════════════ 5. 잔차의 지배 변수
s = slide()
header(s, "적용 ②", "남은 25%는 무엇에 반응하는가",
       "날짜 × 관측소 상호작용 성분과 지상관측 변수들의 상관을 측정했다.")
data = [("습도",-.476),("일사",.383),("지면온도",.373),("전운량",-.258),("강수",-.214),("풍속",.053)]
y = 3.05
for name, v in data:
    text(s, .95, y, 1.5, .32, [{"t":name,"sz":15,"c":INK,"ls":1.0}], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 2.45, y, 1.0, .32, [{"t":f"{v:+.3f}","sz":14,"c":INK,"font":MONO,"ls":1.0}], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    bar(s, 3.8, y+.09, 3.6*abs(v)/.5, 1.0, ACC if abs(v)>.3 else MUTED, h=.14)
    y += .45
box(s, 7.9, 2.95, 4.45, 2.75, fill=PANEL)
text(s, 8.2, 3.2, 3.85, 2.3, [
  {"t":"상위 세 변수는 모두 사용 금지","sz":15,"b":True,"c":INK,"ls":1.3},
  {"t":"셋 다 지표면 에너지가 현열(기온 상승)과 잠열(증발)로 어떻게 나뉘는지의 문제다. 습한 지표는 증발에 에너지를 쓰느라 덜 뜨거워진다.","sz":13,"c":SOFT,"ls":1.45,"space_before":8},
  {"t":"→ 각각의 위성 대응물을 대신 사용\n   수증기 흡수차 · 단파 반사도 · 적외 최저휘도","sz":13,"b":True,"c":ACC,"ls":1.45,"space_before":8}])
box(s, .95, 6.05, 6.6, .95, fill=PANEL)
text(s, 1.25, 6.25, 6.0, .6, [{"t":"부수 소득 — 풍속은 상관이 0.053으로 거의 무관했다. 위성으로 볼 수 없는 변수 중 잃는 것이 적다는 뜻이다.","sz":13,"c":SOFT,"ls":1.4}])

# ══════════════════ 6. 모델 구조
s = slide()
header(s, "적용 ③", "선형 모델을 앞에, 트리는 잔차만")
y = 2.85
for lab, txt in [
  ("가설", "부스팅 트리의 예측은 리프값의 조합이라 학습 데이터의 최대·최소를 넘지 못한다. 8월 자료가 대부분인 학습셋으로 6월을 예측하면 구조적으로 과대평가할 것이다."),
  ("근거", "실제로 6월 예측의 편향이 +4.3°C였다. “8월이니까 30도”를 6월에도 그대로 내놓고 있었다."),
  ("예상", "선형 회귀를 앞에 세워 추세를 담당시키고 트리는 그 잔차만 학습하게 하면, 선형항이 기울기를 따라 값을 이어주므로 계절 밖 편향이 사라질 것이다.")]:
    text(s, .95, y, 1.1, .3, [{"t":lab,"sz":11,"b":True,"c":MUTED,"font":MONO}])
    text(s, 2.25, y, 10.1, .8, [{"t":txt,"sz":15,"c":SOFT,"ls":1.45}])
    y += .95
box(s, .95, 5.85, 11.4, 1.15, fill=C(0xE3,0xF1,0xEA))
text(s, 1.25, 6.05, 10.8, .3, [{"t":"결과","sz":11,"b":True,"c":ADOPT,"font":MONO}])
text(s, 1.25, 6.35, 10.8, .6, [
  {"t":"계절 밖 편향이 제거됐을 뿐 아니라 계절 안 성능도 함께 좋아졌다. 예상하지 못한 효과였다.","sz":15,"b":True,"c":ADOPT,"ls":1.35},
  {"t":"부수 효과 — 두 채널의 차이(IR105 − IR123)를 쓰는 분리대기창 보정도 이때 살아났다. 트리는 변수의 차이 같은 선형 조합을 직접 표현하지 못하기 때문이다.","sz":13,"c":SOFT,"ls":1.4,"space_before":6}])

# ══════════════════ 7. 방위별 인근 관측소
s = slide()
header(s, "적용 ④", "인근 관측소 평균은 방위 정보를 지운다")
y = 2.8
for lab, txt in [
  ("가설", "가장 가까운 8개 관측소의 평균을 쓰면 “주변이 대체로 어떤가”는 담기지만 어느 쪽이 더 뜨거운가는 사라진다. 그런데 기상 현상에는 방위가 있다."),
  ("근거", "오차가 가장 큰 관측소가 동해(3.47) · 울진(3.02) · 포항(2.76) · 속초(2.33)로 전부 동해안이었다. 이 지역의 푄 현상은 “산맥 서쪽은 구름, 동쪽은 맑음”이라는 동서 구조를 갖는다."),
  ("예상", "동·서·남·북 45도 구획별로 가장 가까운 3개씩 따로 평균 내고 동서·남북 대비를 추가하면 동해안 오차가 줄 것이다.")]:
    text(s, .95, y, 1.1, .3, [{"t":lab,"sz":11,"b":True,"c":MUTED,"font":MONO}])
    text(s, 2.25, y, 10.1, .85, [{"t":txt,"sz":15,"c":SOFT,"ls":1.45}])
    y += 1.0
box(s, .95, 5.95, 11.4, 1.05, fill=C(0xE3,0xF1,0xEA))
text(s, 1.25, 6.15, 4.6, .3, [{"t":"결과","sz":11,"b":True,"c":ADOPT,"font":MONO}])
text(s, 1.25, 6.45, 6.5, .45, [{"t":"기온 1.252 → 1.238    습도 7.309 → 7.233","sz":19,"b":True,"c":ADOPT,"font":MONO}])
text(s, 8.0, 6.25, 4.2, .6, [{"t":"공간 관련 변수 중 유일하게 두 예측 대상을 동시에 개선했다.","sz":13,"c":SOFT,"ls":1.4}])

# ══════════════════ 8. 기각 요약
s = slide()
header(s, "검증", "기각한 가설이 채택한 것보다 많다",
       "각각은 물리적으로 그럴듯했고, 왜 실패했는지가 이 문제의 한계를 설명한다.")
rows = [["가설","기대한 근거","결과"],
        ["예측한 습도를 기온 예측에 투입","습도가 기온 잔차와 가장 강한 상관",{"t":"1.281 → 1.283","r":1,"c":REJECT}],
        ["수증기 채널(WV073) 추가","습도의 직접 관측 대리값",{"t":"6.643 → 6.614","r":1,"c":REJECT}],
        ["전국 평균 수준을 변수로 추가","변동의 66%가 날짜 요인",{"t":"1.251 → 1.290","r":1,"c":REJECT}],
        ["3일치 위성 이력 추가","토양 수분이 며칠간 지속",{"t":"1.251 → 1.423","r":1,"c":REJECT}],
        ["습도에 로짓 변환","0~100% 경계 반영",{"t":"기여 0.005","r":1,"c":MUTED}],
        ["14시 외 시각도 학습에 사용","표본 4배 증가",{"t":"구조 변경 위험","r":1,"c":MUTED}]]
table(s, .95, 3.15, 11.4, rows, [4.4, 4.4, 2.6], rh=.56, fs=14)

# ══════════════════ 9. 기각 심화 — 수증기
s = slide()
header(s, "기각 사례", "가장 반직관적이었던 결과")
text(s, .95, 2.7, 11.4, .5, [{"t":"습도를 예측하는데 수증기 전용 채널이 도움이 되지 않았다","sz":22,"b":True,"c":INK,"ls":1.3}])
y = 3.5
for lab, txt in [
  ("가설", "습도가 점수의 37%를 차지하는데 수증기 전용 채널을 한 번도 쓰지 않았다. 지금은 두 적외 채널의 차이가 수증기를 간접 보정할 뿐이다."),
  ("검증", "WV073(하층 수증기 7.3μm)과 IR133(이산화탄소 흡수대)을 111일치 수집해 시험했다."),
  ("결과", "습도 오차 6.643 → 6.614. 같은 검증 구간에서 이미 잡음으로 확인된 범위였다. 나머지 421일 추가 수집을 중단했다.")]:
    text(s, .95, y, 1.1, .3, [{"t":lab,"sz":11,"b":True,"c":MUTED,"font":MONO}])
    text(s, 2.25, y, 10.1, .75, [{"t":txt,"sz":15,"c":SOFT,"ls":1.45}])
    y += .88
box(s, .95, 6.1, 11.4, 1.0, fill=C(0xF6,0xE9,0xE3))
text(s, 1.25, 6.32, 10.8, .6, [
  {"t":"해석 — 7.3μm 흡수대가 보는 것은 대기 중·상층의 수증기이고,","sz":15,"b":True,"c":REJECT,"ls":1.35},
  {"t":"우리가 맞혀야 하는 것은 지상 2m의 상대습도다. 둘 사이의 연결이 생각보다 약했다.","sz":15,"b":True,"c":REJECT,"ls":1.35}])

# ══════════════════ 10. 달성 가능한 상한
s = slide()
header(s, "평가 방법", "이 문제에서 도달 가능한 상한을 직접 쟀다",
       "다른 팀과 비교해야만 우리 점수를 평가할 수 있는 것은 아니다.")
rows = [["방법","점수","성격"],
        ["14시 지상관측을 그대로 복사",{"t":"0.000","r":1},"규칙 위반 · 정답 복사"],
        ["13시 지상관측을 그대로 복사",{"t":"1.345","r":1},"규칙 위반 · 1시간 지속성"],
        ["다른 95개 관측소의 실제 14시 값으로 완벽 보간",{"t":"2.173","r":1},"합법적 기준선"],
        [{"t":"우리 모델 — 위성만 사용","b":True,"c":ADOPT},{"t":"1.961","r":1,"b":True,"c":ADOPT},{"t":"교차검증","c":ADOPT}]]
table(s, .95, 3.15, 11.4, rows, [5.8, 1.6, 4.0], rh=.56, fs=14)
box(s, .95, 5.7, 11.4, 1.35, fill=PANEL)
text(s, 1.25, 5.92, 10.8, 1.0, [
  {"t":"세 번째 줄이 핵심이다.","sz":16,"b":True,"c":INK,"ls":1.3},
  {"t":"다른 모든 관측소의 정답을 알아도 공간 보간만으로는 2.173이다. 위성만 쓰는 우리 모델이 그보다 낫다는 것은, 위성이 이웃 관측에 없는 국지 정보를 실제로 담고 있다는 직접 증거다.","sz":14,"c":SOFT,"ls":1.45,"space_before":5}])

# ══════════════════ 11. 오차 분해
s = slide()
header(s, "진단", "남은 오차는 전부 관측소별 차이였다",
       "“전국 평균 수준” 가설이 기각된 뒤, 오차를 두 성분으로 다시 나눴다.")
rows = [["성분","오차","의미"],
        ["전체 기온 오차",{"t":"1.251","r":1},"—"],
        ["  일별 평균을 못 맞힌 몫",{"t":"0.341","r":1},"이미 거의 해결됨"],
        ["  관측소별 차이를 못 맞힌 몫",{"t":"1.204","r":1},"오차의 거의 전부"],
        [{"t":"그날 전국 평균을 완벽히 안다면","c":ACC},{"t":"1.204","r":1,"c":ACC},{"t":"개선폭 0.047뿐","c":ACC}]]
table(s, .95, 3.15, 11.4, rows, [5.2, 1.6, 4.6], rh=.56, fs=14)
box(s, .95, 5.6, 11.4, 1.45, fill=PANEL)
text(s, 1.25, 5.82, 10.8, 1.1, [
  {"t":"변동의 66%가 날짜 요인이었지만, 그 성분은 이미 계절·연도·전국 위성 패턴으로 거의 다 설명되고 있었다.","sz":15,"b":True,"c":INK,"ls":1.35},
  {"t":"남은 오차를 지배하는 습도·일사·지면온도는 입력이 금지돼 있고, 2km 화소를 18km 범위로 평균 내면 해안선과 산지의 국지 구조가 뭉개진다. 오차가 가장 큰 관측소가 전부 푄·해풍 지역인 것이 그 증거다.","sz":14,"c":SOFT,"ls":1.45,"space_before":5}])

# ══════════════════ 12. 경과 + 최종
s = slide()
header(s, "경과", "개선의 순서")
steps = [("2.401","7년치 데이터 + 연도 변수",1.00),
         ("2.130","예측 대상별 변수 분리",.887),
         ("2.009","전국 평균 대비 편차",.837),
         ("1.983","가장 가까운 8개 관측소",.826),
         ("1.961","방위별 인근 관측소 — 최종",.817)]
y = 2.75
for i,(v,lab,f) in enumerate(steps):
    last = i == len(steps)-1
    if last: box(s, .85, y-.1, 11.6, .62, fill=C(0xE2,0xF0,0xF2))
    text(s, .95, y, 1.3, .38, [{"t":v,"sz":19,"b":True,"c":ACC if last else INK,"font":MONO,"ls":1.0}], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 2.5, y, 5.2, .38, [{"t":lab,"sz":15,"b":last,"c":INK if last else SOFT,"ls":1.0}], anchor=MSO_ANCHOR.MIDDLE)
    bar(s, 8.0, y+.12, 4.3*f, 1.0, ACC, h=.16)
    y += .68
text(s, .95, 6.25, 11.4, .3, [{"t":"검증 구간 2026-08-16~22. 학습은 그 이전 날짜만 사용해 미래를 전혀 보지 않았다.","sz":12,"c":MUTED}])
box(s, .95, 6.7, 11.4, .55, fill=PANEL)
text(s, 1.25, 6.82, 10.8, .35, [{"t":"최종 모델 — 선형 회귀 + 부스팅 트리 잔차 · 3개 시드 평균 · 51,005행 / 536일 / 96개 관측소 · 2019–2026","sz":13,"c":SOFT}])

# ══════════════════ 13. 한계
s = slide()
header(s, "한계", "우리 검증이 틀린 부분")
rows = [["검증","점수","조건"],
        ["2026-08-16~22 교차검증",{"t":"1.961","r":1},"7일 · 미래 미사용"],
        ["2026-08-23 단일 검증",{"t":"1.730","r":1},"1일 · 학습에 없던 날 · 실측 대조"],
        [{"t":"2026-08-26~30 최종 채점","b":True,"c":REJECT},{"t":"2.27","r":1,"b":True,"c":REJECT},{"t":"5일","c":REJECT}]]
table(s, .95, 2.85, 11.4, rows, [5.2, 1.6, 4.6], rh=.58, fs=14)
box(s, .95, 5.0, 11.4, 2.0, fill=C(0xF6,0xE9,0xE3))
text(s, 1.25, 5.25, 10.8, 1.6, [
  {"t":"우리 검증은 0.3 이상 낙관적이었다.","sz":18,"b":True,"c":REJECT,"ls":1.3},
  {"t":"8월 23일 검증은 하루짜리였다. 96개 관측소가 같은 기단을 공유하므로 실질 표본은 사실상 1일이고, 그날이 마침 쉬운 날이었을 가능성이 크다. 검증 과정에서 “하루 표본으로는 0.03 차이를 구별할 수 없다”고 여러 번 확인했으면서도, 최종 성능의 근거로는 그 하루를 앞세웠다.","sz":14,"c":SOFT,"ls":1.5,"space_before":8},
  {"t":"단일 구간 검증의 낙관 편향 — 다음에 가장 먼저 고칠 부분이다.","sz":14,"b":True,"c":REJECT,"ls":1.4,"space_before":8}])

out = "/Users/chaehyeon/Desktop/데이터분석대회_발표자료.pptx"
prs.save(out)
print(f"저장: {out}")
print(f"슬라이드 {len(prs.slides.__iter__.__self__._sldIdLst)}장")
