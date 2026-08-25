PROJECT_NAME = "AI Media Forensics Suite"

N1="#04070E"; N2="#0A1020"; N3="#111D35"; N4="#182B4A"
CP="#E0854A"; AM="#D4943A"; GD="#F0C040"
W="#E8ECF4"; M="#6B7D9A"; RD="#F04444"; GN="#2DD4A0"; YL="#FACC15"

_RAW = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif!important;color:$W$!important;-webkit-font-smoothing:antialiased!important}
h1,h2,h3,h4{font-family:'Inter',sans-serif!important;color:$W$!important;letter-spacing:-.03em;font-weight:700!important}
code,pre{font-family:'JetBrains Mono',monospace!important}
.stApp{background:$N1$!important;overflow-x:hidden}
.block-container{max-width:1180px!important;margin:0 auto!important;padding-top:2.2rem!important}

section[data-testid="stSidebar"]{background:linear-gradient(180deg,$N2$,$N1$)!important;border-right:1px solid rgba(224,133,74,.05)!important}
section[data-testid="stSidebar"] *{color:$M$!important}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:$W$!important}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]{border-radius:10px!important;margin:3px 10px!important;padding:9px 14px!important;transition:all .2s!important}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover{background:rgba(224,133,74,.08)!important;color:$W$!important}
section[data-testid="stSidebar"] [aria-current="page"]{background:rgba(224,133,74,.12)!important;border-left:3px solid $CP$!important;color:$W$!important}

/* WELCOME */
.welcome-page{text-align:center;padding:5rem 2rem!important;z-index:1!important;animation:fadeUp .6s ease-out!important}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.welcome-badge{font-family:'JetBrains Mono',monospace!important;font-size:.7rem!important;font-weight:700!important;letter-spacing:.2em!important;text-transform:uppercase!important;color:$CP$!important;background:rgba(224,133,74,.08)!important;border:1px solid rgba(224,133,74,.15)!important;padding:8px 20px!important;border-radius:100px!important;display:inline-block!important;margin-bottom:2rem!important}
.welcome-title{font-size:3.2rem!important;font-weight:900!important;line-height:1.1!important;margin-bottom:1.2rem!important;letter-spacing:-.04em!important}
.welcome-gradient{background:linear-gradient(135deg,$CP$,$GD$)!important;-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;background-clip:text!important}
.welcome-subtitle{font-size:1.05rem!important;color:$M$!important;max-width:520px!important;line-height:1.7!important;margin:0 auto 3rem!important}
.welcome-features{display:flex!important;gap:1.2rem!important;justify-content:center!important;flex-wrap:wrap!important;margin-bottom:2.5rem!important}
.wf-card{background:$N2$!important;border:1px solid rgba(224,133,74,.08)!important;border-radius:16px!important;padding:1.4rem!important;width:250px!important;text-align:left!important;transition:all .3s!important}
.wf-card:hover{border-color:rgba(224,133,74,.2)!important;transform:translateY(-4px)!important;box-shadow:0 12px 30px rgba(0,0,0,.3)!important}
.wf-icon{font-size:1.8rem!important;margin-bottom:.6rem!important}
.wf-name{font-weight:700!important;font-size:.92rem!important;margin-bottom:.3rem!important}
.wf-desc{font-size:.78rem!important;color:$M$!important;line-height:1.5!important}
.welcome-stats{display:flex!important;align-items:center!important;justify-content:center!important;gap:2rem!important;margin-bottom:2rem!important}
.ws-val{font-family:'JetBrains Mono',monospace!important;font-size:1.5rem!important;font-weight:700!important;color:$W$!important}
.ws-label{font-size:.72rem!important;color:$M$!important;text-transform:uppercase!important;letter-spacing:.08em!important;margin-top:.2rem!important}
.ws-div{width:1px!important;height:32px!important;background:rgba(255,255,255,.08)!important}

/* HEADER */
.brand-header{background:linear-gradient(135deg,$N2$,$N3$ 50%,$N2$)!important;padding:1.8rem 2.2rem!important;border-radius:20px!important;margin-bottom:1.8rem!important;border:1px solid rgba(224,133,74,.08)!important;box-shadow:0 8px 30px rgba(0,0,0,.4)!important;z-index:1!important;position:relative!important;overflow:hidden!important}
.brand-header::after{content:""!important;position:absolute!important;top:0!important;right:0!important;width:4px!important;height:100%!important;background:linear-gradient(180deg,$GD$,$CP$,$AM$)!important;border-radius:0 20px 20px 0!important}
.brand-eyebrow{color:$AM$!important;font-size:.6rem!important;font-weight:700!important;letter-spacing:.2em!important;text-transform:uppercase!important;margin:0 0 .3rem!important;display:flex!important;align-items:center!important;gap:8px!important}
.brand-eyebrow::before{content:""!important;width:6px!important;height:6px!important;border-radius:50%!important;background:$GN$!important;animation:dp 2s ease-in-out infinite!important}
@keyframes dp{0%,100%{opacity:1}50%{opacity:.3}}
.brand-header h1{color:$W$!important;margin:0!important;font-size:1.7rem!important;font-weight:900!important;letter-spacing:-.04em!important}
.brand-header p{color:$M$!important;margin:.4rem 0 0!important;font-size:.85rem!important;max-width:640px!important;line-height:1.6!important}

/* STEP WIZARD */
.step-wizard{display:flex!important;align-items:center!important;justify-content:center!important;padding:1.2rem 1.5rem!important;background:$N2$!important;border:1px solid rgba(224,133,74,.06)!important;border-radius:16px!important;margin-bottom:1.8rem!important;z-index:1!important;animation:fadeUp .3s ease-out!important}
.sw-step{display:flex!important;align-items:center!important;gap:6px!important;opacity:.3!important;transition:all .3s!important}
.sw-step.active{opacity:1!important}
.sw-step.done .sw-num{background:$GN$!important;color:$N1$!important}
.sw-num{width:30px!important;height:30px!important;border-radius:50%!important;display:flex!important;align-items:center!important;justify-content:center!important;font-family:'JetBrains Mono',monospace!important;font-size:.78rem!important;font-weight:700!important;background:$N4$!important;color:$M$!important;transition:all .3s!important;flex-shrink:0!important}
.sw-step.active .sw-num{background:linear-gradient(135deg,$CP$,$AM$)!important;color:#fff!important;box-shadow:0 0 0 3px rgba(224,133,74,.2)!important}
.sw-label{font-size:.75rem!important;font-weight:600!important;color:$M$!important;white-space:nowrap!important}
.sw-step.active .sw-label{color:$W$!important}
.sw-line{width:36px!important;height:2px!important;background:rgba(255,255,255,.06)!important;border-radius:1px!important;transition:background .3s!important;flex-shrink:0!important}
.sw-line.active{background:linear-gradient(90deg,$CP$,$AM$)!important}

/* STEP HEADER */
.step-header{display:flex!important;align-items:flex-start!important;gap:1rem!important;margin-bottom:1.5rem!important;padding:1rem 1.3rem!important;background:$N2$!important;border:1px solid rgba(224,133,74,.06)!important;border-radius:14px!important}
.sh-num{font-family:'JetBrains Mono',monospace!important;font-size:1.5rem!important;font-weight:800!important;color:$CP$!important;line-height:1!important;opacity:.5!important}
.sh-title{font-size:1.1rem!important;font-weight:700!important;color:$W$!important;margin-bottom:.15rem!important}
.sh-desc{font-size:.82rem!important;color:$M$!important;line-height:1.5!important}

/* VERDICT */
.verdict-banner{border-left:5px solid!important;border-radius:0 18px 18px 0!important;padding:1.5rem 1.8rem!important;margin-bottom:1.5rem!important;background:$N2$!important;position:relative!important;overflow:hidden!important;z-index:1!important;animation:fadeUp .4s ease-out!important}
.verdict-banner::before{content:""!important;position:absolute!important;top:0!important;left:0!important;width:100%!important;height:100%!important;background:linear-gradient(90deg,rgba(224,133,74,.04),transparent 40%)!important;pointer-events:none!important}
.vb-row{display:flex!important;justify-content:space-between!important;align-items:flex-start!important;margin-bottom:.4rem!important}
.vb-label{font-size:2rem!important;font-weight:900!important;letter-spacing:.04em!important;line-height:1.1!important}
.vb-time{font-family:'JetBrains Mono',monospace!important;font-size:.8rem!important;color:$M$!important;background:rgba(255,255,255,.04)!important;padding:5px 12px!important;border-radius:8px!important;border:1px solid rgba(255,255,255,.06)!important;display:flex!important;align-items:center!important;gap:5px!important}
.vb-time-icon{font-size:.65rem!important;opacity:.5!important}
.vb-prob{font-size:1.05rem!important;color:$M$!important;font-family:'JetBrains Mono',monospace!important;margin-bottom:.8rem!important}
.vb-bar-wrap{height:5px!important;background:rgba(255,255,255,.04)!important;border-radius:3px!important;overflow:hidden!important;margin-bottom:1rem!important}
.vb-bar{height:100%!important;border-radius:3px!important;transition:width 1.2s ease!important;box-shadow:0 0 12px currentColor!important}
.vb-meta{display:flex!important;flex-wrap:wrap!important;gap:6px!important}
.vb-tag{font-size:.76rem!important;color:$M$!important;font-family:'JetBrains Mono',monospace!important;background:rgba(255,255,255,.03)!important;padding:4px 12px!important;border-radius:7px!important;border:1px solid rgba(255,255,255,.05)!important}
.vb-tag b{color:$W$!important}

/* METRIC CARDS */
.metric-card{background:$N2$!important;border:1px solid rgba(224,133,74,.06)!important;border-radius:16px!important;padding:1.3rem 1.4rem!important;transition:all .3s!important;position:relative!important;overflow:hidden!important;z-index:1!important;animation:fadeUp .4s ease-out both!important}
.metric-card:nth-child(1){animation-delay:.05s!important}
.metric-card:nth-child(2){animation-delay:.12s!important}
.metric-card:nth-child(3){animation-delay:.19s!important}
.metric-card::before{content:""!important;position:absolute!important;top:0!important;left:0!important;right:0!important;height:3px!important;border-radius:16px 16px 0 0!important;background:linear-gradient(90deg,$CP$,$GD$)!important;opacity:0!important;transition:opacity .3s!important}
.metric-card:hover{border-color:rgba(224,133,74,.2)!important;transform:translateY(-4px)!important;box-shadow:0 14px 36px rgba(0,0,0,.3)!important}
.metric-card:hover::before{opacity:1!important}
.mc-top{display:flex!important;justify-content:space-between!important;align-items:center!important;margin-bottom:.4rem!important}
.mc-label{font-size:.7rem!important;text-transform:uppercase!important;letter-spacing:.12em!important;color:$M$!important;font-weight:700!important}
.mc-weight{font-family:'JetBrains Mono',monospace!important;font-size:.65rem!important;color:$M$!important;opacity:.4!important}
.mc-value{font-size:2rem!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;margin-bottom:.7rem!important;line-height:1!important}
.mc-bar{height:4px!important;background:rgba(255,255,255,.04)!important;border-radius:2px!important;overflow:hidden!important;margin-bottom:.5rem!important}
.mc-fill{height:100%!important;border-radius:2px!important;transition:width 1.2s ease!important;box-shadow:0 0 8px currentColor!important}
.mc-assess{font-size:.72rem!important;font-weight:600!important;color:$M$!important;font-family:'JetBrains Mono',monospace!important}

/* IMAGES & EVIDENCE */
.img-card-title{font-size:.8rem!important;font-weight:600!important;color:$M$!important;text-transform:uppercase!important;letter-spacing:.1em!important;margin-bottom:.7rem!important;display:flex!important;align-items:center!important;gap:8px!important}
.ict-dot{width:7px!important;height:7px!important;border-radius:50%!important;background:$CP$!important;display:inline-block!important;animation:dp 2s ease-in-out infinite!important}
.evidence-label{font-size:.72rem!important;font-weight:600!important;color:$CP$!important;font-family:'JetBrains Mono',monospace!important;letter-spacing:.08em!important;margin-bottom:.5rem!important;padding:5px 0!important;border-bottom:1px solid rgba(224,133,74,.1)!important}

/* PDF */
.pdf-hero{display:flex!important;align-items:center!important;gap:1.4rem!important;background:$N2$!important;border:1px solid rgba(224,133,74,.08)!important;border-radius:18px!important;padding:1.8rem!important;margin-bottom:1.5rem!important;animation:fadeUp .4s ease-out!important}
.pdf-hero-icon{font-size:2.8rem!important;flex-shrink:0!important}
.pdf-hero-title{font-size:1.15rem!important;font-weight:700!important;color:$W$!important;margin-bottom:.25rem!important}
.pdf-hero-desc{font-size:.82rem!important;color:$M$!important;line-height:1.5!important}
.report-features{display:grid!important;grid-template-columns:1fr 1fr!important;gap:.5rem!important;animation:fadeUp .4s ease-out .2s both!important}
.rf-item{font-size:.82rem!important;color:$M$!important;padding:7px 12px!important;background:$N2$!important;border:1px solid rgba(224,133,74,.04)!important;border-radius:9px!important;display:flex!important;align-items:center!important;gap:7px!important;transition:all .2s!important}
.rf-item:hover{border-color:rgba(224,133,74,.12)!important;background:rgba(224,133,74,.03)!important}
.rf-check{color:$GN$!important;font-weight:700!important}

/* TABS */
.stTabs [data-baseweb="tab-list"]{gap:4px!important;background:$N2$!important;padding:4px!important;border-radius:14px!important;border:1px solid rgba(224,133,74,.06)!important;z-index:1!important}
.stTabs [data-baseweb="tab"]{border-radius:10px!important;padding:10px 26px!important;font-weight:600!important;color:$M$!important;transition:all .2s!important;font-size:.9rem!important}
.stTabs [data-baseweb="tab"]:hover{color:$W$!important;background:rgba(224,133,74,.06)!important;transform:translateY(-1px)!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,$CP$,$AM$)!important;color:#fff!important;box-shadow:0 4px 20px rgba(224,133,74,.3)!important;transform:translateY(-1px)!important}

/* BUTTONS */
.stButton>button{font-weight:600!important;border-radius:12px!important;padding:.7rem 1.5rem!important;transition:all .25s!important;font-size:.92rem!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,$CP$,$AM$)!important;color:#fff!important;border:none!important;box-shadow:0 4px 20px rgba(224,133,74,.3)!important}
.stButton>button[kind="primary"]:hover{box-shadow:0 8px 30px rgba(224,133,74,.45)!important;transform:translateY(-2px)!important}
.stButton>button[kind="primary"]:active{transform:translateY(0)!important;box-shadow:0 2px 10px rgba(224,133,74,.3)!important}
.stButton>button[kind="secondary"]{background:$N2$!important;color:$W$!important;border:1px solid rgba(224,133,74,.1)!important}
.stButton>button[kind="secondary"]:hover{border-color:$CP$!important;background:rgba(224,133,74,.06)!important;box-shadow:0 4px 16px rgba(0,0,0,.2)!important;transform:translateY(-2px)!important}

/* UPLOAD */
/* TOP HEADER / TOOLBAR — was unstyled, showing as a jarring mismatched block against the dark theme */
header[data-testid="stHeader"]{background:$N1$!important;border-bottom:1px solid rgba(224,133,74,.06)!important}
[data-testid="stToolbar"]{background:transparent!important}
[data-testid="stToolbar"] button{color:$M$!important}
[data-testid="stToolbar"] svg{fill:$M$!important}
[data-testid="stDecoration"]{background:linear-gradient(90deg,$CP$,$GD$,$CP$)!important}
#MainMenu, footer{visibility:hidden!important}

/* UPLOAD — reduced from an oversized dropzone to a compact, tidy one */
[data-testid="stFileUploader"]{border:2px dashed rgba(224,133,74,.15)!important;border-radius:12px!important;background:$N2$!important;padding:.9rem 1.1rem!important;transition:all .3s!important;z-index:1!important}
[data-testid="stFileUploader"]:hover{border-color:$CP$!important;background:rgba(224,133,74,.03)!important}
[data-testid="stFileUploaderDropzone"]{min-height:0!important;padding:.4rem!important}
[data-testid="stFileUploaderDropzoneInstructions"] span{font-size:.82rem!important}

/* IMAGES */
[data-testid="stImage"] img{border-radius:14px!important;border:1px solid rgba(224,133,74,.06)!important;animation:imgIn .45s ease-out!important;transition:box-shadow .3s!important}
[data-testid="stImage"] img:hover{box-shadow:0 8px 28px rgba(0,0,0,.3)!important}
@keyframes imgIn{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}

/* STATUS / PROGRESS */
[data-testid="stStatusWidget"]{background:$N2$!important;border:1px solid rgba(224,133,74,.08)!important;border-radius:14px!important;padding:1rem 1.4rem!important;z-index:1!important}
.stProgress>div>div>div{background:linear-gradient(90deg,$CP$,$GD$)!important;border-radius:4px!important}

/* OTHER */
[data-testid="stExpander"]{border:1px solid rgba(224,133,74,.06)!important;border-radius:12px!important;background:$N2$!important;z-index:1!important}
[data-testid="stArrowVegaLiteChart"]{background:$N2$!important;border-radius:14px!important;padding:1rem!important;border:1px solid rgba(224,133,74,.06)!important;z-index:1!important}
[data-testid="stAlert"]{border-radius:12px!important;z-index:1!important}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:$N1$}
::-webkit-scrollbar-thumb{background:$N4$;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:$CP$}

@media(max-width:768px){
    .welcome-title{font-size:2.2rem!important}
    .welcome-subtitle{font-size:.9rem!important}
    .welcome-features{flex-direction:column!important;align-items:center!important}
    .wf-card{width:100%!important;max-width:320px!important}
    .brand-header h1{font-size:1.3rem!important}
    .vb-label{font-size:1.5rem!important}
    .vb-row{flex-direction:column!important;gap:.4rem!important}
    .vb-meta{flex-direction:column!important}
    .mc-value{font-size:1.6rem!important}
    .sw-label{font-size:.65rem!important}
    .sw-line{width:22px!important}
    .step-wizard{padding:.8rem!important}
    .step-header{flex-direction:column!important;gap:.6rem!important}
    .report-features{grid-template-columns:1fr!important}
    .pdf-hero{flex-direction:column!important;text-align:center!important}
}
@media(max-width:480px){
    .welcome-title{font-size:1.8rem!important}
    .vb-label{font-size:1.3rem!important}
    .mc-value{font-size:1.4rem!important}
    .stTabs [data-baseweb="tab"]{padding:8px 14px!important;font-size:.78rem!important}
}
</style>
"""

CUSTOM_CSS = _RAW.replace("$N1$",N1).replace("$N2$",N2).replace("$N3$",N3).replace("$N4$",N4).replace("$CP$",CP).replace("$AM$",AM).replace("$GD$",GD).replace("$W$",W).replace("$M$",M).replace("$RD$",RD).replace("$GN$",GN).replace("$YL$",YL)

BRAND_HEADER_HTML = """
<div class="brand-header">
    <p class="brand-eyebrow">Digital Forensics Laboratory</p>
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>"""

def risk_color(s):
    if s < .3: return GN
    if s < .6: return YL
    return RD