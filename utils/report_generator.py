"""
Forensic Examination Report — ReportLab
Professional forensic-grade PDF with images, case ID, certification.
"""

import os, io, tempfile, numpy as np, random
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)

# Colors
NAVY=colors.HexColor("#0B1120"); COPPER=colors.HexColor("#C87941")
AMBER=colors.HexColor("#D4943A"); GOLD=colors.HexColor("#D4AF37")
LBG=colors.HexColor("#F4F6F9"); W=colors.white; GRAY=colors.HexColor("#64748B")
LG=colors.HexColor("#E2E8F0"); RED=colors.HexColor("#EF4444")
GRN=colors.HexColor("#22C55E"); YEL=colors.HexColor("#EAB308")
DT=colors.HexColor("#1E293B"); STAMP=colors.HexColor("#C0392B")

def get_report_temp_path(fn="report.pdf"):
    return os.path.join(tempfile.gettempdir(), fn)

def _case_id():
    return f"DFAC-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

def _pil(bgr):
    if bgr is None: return None
    from PIL import Image as P
    if len(bgr.shape)==3 and bgr.shape[2]==3:
        return P.fromarray(bgr[:,:,::-1])
    return P.fromarray(bgr)

def _rlimg(bgr, mw=400, mh=260):
    p=_pil(bgr)
    if p is None: return None
    w,h=p.size; r=min(mw/w,mh/h,1.0)
    buf=io.BytesIO(); p.save(buf,format='PNG'); buf.seek(0)
    return RLImage(buf, width=w*r, height=h*r)

def _sty():
    s=getSampleStyleSheet()
    for n,kw in [
        ("CID",dict(fontSize=9,textColor=COPPER,fontName="Helvetica-Bold",spaceAfter=2,alignment=TA_RIGHT)),
        ("RT",dict(fontSize=22,textColor=NAVY,fontName="Helvetica-Bold",spaceAfter=4,leading=26)),
        ("RS",dict(fontSize=11,textColor=COPPER,fontName="Helvetica-Bold",spaceAfter=16)),
        ("SH",dict(fontSize=13,textColor=NAVY,fontName="Helvetica-Bold",spaceBefore=14,spaceAfter=8)),
        ("BD",dict(fontSize=9.5,textColor=DT,fontName="Helvetica",leading=14,alignment=TA_JUSTIFY)),
        ("BS",dict(fontSize=8.5,textColor=GRAY,fontName="Helvetica",leading=12)),
        ("CT",dict(fontSize=9.5,textColor=DT,fontName="Helvetica",leading=14,alignment=TA_JUSTIFY,spaceBefore=3,spaceAfter=3)),
    ]: s.add(ParagraphStyle(name=n,**kw))
    return s

def _vc(s):
    if s<.3: return GRN
    if s<.6: return YEL
    return RED

def _hf(c,d,cid):
    c.saveState(); w,h=A4
    # Header
    c.setStrokeColor(COPPER); c.setLineWidth(1.5)
    c.line(20*mm,h-15*mm,w-20*mm,h-15*mm)
    c.setFont("Helvetica",7); c.setFillColor(GRAY)
    c.drawString(20*mm,h-13*mm,"FORENSIC EXAMINATION REPORT  •  CONFIDENTIAL")
    c.drawRightString(w-20*mm,h-13*mm,f"Case {cid}")
    # Footer
    c.setStrokeColor(LG); c.setLineWidth(.5)
    c.line(20*mm,14*mm,w-20*mm,14*mm)
    c.setFont("Helvetica",7); c.setFillColor(GRAY)
    c.drawString(20*mm,10*mm,"AI Media Forensics Suite  •  Digital Forensics Laboratory")
    c.drawRightString(w-20*mm,10*mm,f"Page {d.page}")
    # Watermark
    c.saveState(); c.setFont("Helvetica-Bold",50)
    c.setFillColor(colors.Color(0,0,0,alpha=0.018))
    c.translate(w/2,h/2); c.rotate(45)
    c.drawCentredString(0,0,"CONFIDENTIAL")
    c.restoreState()
    c.restoreState()

def _mtbl(fn,at,cid,tm=None):
    d=[["Case ID:",cid],["File Analyzed:",fn],
       ["Analysis Type:","Deepfake Detection" if at=="deepfake" else "Media Authenticity Check"],
       ["Report Generated:",datetime.now().strftime("%d %B %Y, %I:%M:%S %p")]]
    if tm: d.append(["Analysis Duration:",f"{tm} seconds"])
    t=Table(d,colWidths=[120,340])
    t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),9),("TEXTCOLOR",(0,0),(0,-1),NAVY),
        ("TEXTCOLOR",(1,0),(1,-1),DT),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("LINEBELOW",(0,0),(-1,-1),.3,LG),
    ]))
    return t

def _vbn(s,txt):
    t=Table([[txt]],colWidths=[460])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),_vc(s)),("TEXTCOLOR",(0,0),(-1,-1),W),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),13),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),12),
        ("BOTTOMPADDING",(0,0),(-1,-1),12),
    ]))
    return t

def _imgs(imgs,keys,st,cols=2):
    el=[]
    av=[(k,imgs[k]) for k in keys if k in imgs and imgs[k] is not None]
    if not av: return [Paragraph("<i>No image available.</i>",st["BS"])]
    if cols==2 and len(av)>=2:
        i1=_rlimg(av[0][1],210,170); i2=_rlimg(av[1][1],210,170)
        if i1 and i2:
            l1=av[0][0].replace("_"," ").title(); l2=av[1][0].replace("_"," ").title()
            t=Table([[i1,i2],
                [Paragraph(f'<font size="8" color="#64748B">{l1}</font>',st["BS"]),
                 Paragraph(f'<font size="8" color="#64748B">{l2}</font>',st["BS"])]],
                colWidths=[230,230])
            t.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,0),"MIDDLE"),("TOPPADDING",(0,1),(-1,1),4)]))
            el.append(t)
        elif i1: el.append(i1)
    else:
        for k,a in av:
            im=_rlimg(a,380,240)
            if im:
                el.append(im)
                el.append(Paragraph(f'<font size="8" color="#64748B">{k.replace("_"," ").title()}</font>',st["BS"]))
    return el

def build_report(op,fn,at,res,images=None,exif_data=None):
    if images is None: images={}
    s=_sty(); cid=_case_id(); tm=res.get("time")
    doc=SimpleDocTemplate(op,pagesize=A4,topMargin=22*mm,bottomMargin=18*mm,
        leftMargin=20*mm,rightMargin=20*mm,title=f"Forensic Report — {cid}",author="AI Media Forensics Suite")
    story=[]

    # ── HEADER ──
    story.append(Paragraph(f"Case {cid}",s["CID"]))
    story.append(Spacer(1,4))
    story.append(Paragraph("Forensic Examination Report",s["RT"]))
    story.append(Paragraph("AI-Powered Deepfake & Media Authenticity Analysis",s["RS"]))
    story.append(Spacer(1,6))
    story.append(_mtbl(fn,at,cid,tm))
    story.append(Spacer(1,4))
    story.append(HRFlowable(width="100%",thickness=1.2,color=COPPER,spaceAfter=10))

    # ── 1. EVIDENCE ──
    story.append(Paragraph("1. Evidence Documentation",s["SH"]))
    story.append(Paragraph("The following image(s) constitute the digital evidence submitted for forensic examination.",s["BD"]))
    story.append(Spacer(1,6))
    if at=="deepfake":
        story.extend(_imgs(images,["original","face_crop"],s,2))
        if "gradcam" in images and images["gradcam"] is not None:
            story.append(Spacer(1,8))
            story.append(Paragraph("<b>Grad-CAM Attention Heatmap</b>",s["BD"]))
            story.append(Paragraph("Warm colors indicate regions the neural network weighted most heavily in classification.",s["BS"]))
            story.append(Spacer(1,4))
            gc=_rlimg(images["gradcam"],380,240)
            if gc: story.append(gc)
    else:
        story.extend(_imgs(images,["original","ela"],s,2))
    story.append(Spacer(1,6))

    # ── 2. VERDICT ──
    story.append(Paragraph("2. Examination Verdict",s["SH"]))
    if at=="deepfake":
        rk=res.get("fake_probability",0); lb=res.get("label","UNKNOWN")
        vt=f"{lb}  —  {rk*100:.1f}% Fake Probability"
    else:
        rk=res.get("overall_risk",0); vd=res.get("verdict","Unknown")
        vt=f"{vd}  —  {rk*100:.1f}% Risk Score"
    story.append(_vbn(rk,vt))
    story.append(Spacer(1,8))

    conf="HIGH" if rk>.8 or rk<.2 else "MODERATE" if rk>.6 or rk<.4 else "LOW"
    mode=res.get("mode","unknown"); mn=res.get("model_name","N/A")
    id2=[["Confidence Level:",conf],["Analysis Mode:",mode.upper()],["Model / Method:",mn]]
    if tm: id2.append(["Processing Time:",f"{tm} seconds"])
    it=Table(id2,colWidths=[130,330])
    it.setStyle(TableStyle([("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),9),("TEXTCOLOR",(0,0),(0,-1),NAVY),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),("LINEBELOW",(0,0),(-1,-1),.3,LG)]))
    story.append(it)

    if mode=="heuristic":
        story.append(Spacer(1,6))
        story.append(Paragraph(
            "<b>Note:</b> Analysis performed using frequency-domain heuristic fallback. "
            "Trained model weights were unavailable. Results are indicative signals only.",
            ParagraphStyle(name="HN",fontSize=9,textColor=AMBER,fontName="Helvetica",leading=13,
                           borderPadding=8,backColor=colors.HexColor("#FFFBEB"))))

    # ── 3. METRICS ──
    story.append(Paragraph("3. Detailed Analysis Metrics",s["SH"]))
    if at=="deepfake":
        if "frames_analyzed" in res:
            fd=[["Metric","Value"],["Total Frames Sampled",str(res["frames_analyzed"])],
                ["Frames Flagged FAKE",str(res["frames_flagged_fake"])],
                ["Fake Frame Ratio",f"{res['frames_flagged_fake']/max(1,res['frames_analyzed'])*100:.1f}%"],
                ["Average Fake Probability",f"{rk*100:.1f}%"]]
        else:
            fd=[["Metric","Value"],["Fake Probability",f"{rk*100:.1f}%"],["Classification",lb]]
        if "debug" in res:
            fd+=["","Frequency Analysis"]
            for k,v in res["debug"].items():
                fd.append([f"  {k.replace('_',' ').title()}",f"{v:.4f}"])
        ft=Table(fd,colWidths=[200,260])
        ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),W),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),9),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",(0,1),(0,-1),NAVY),("GRID",(0,0),(-1,-1),.4,LG),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[W,LBG]),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(ft)
    else:
        def _ass(v): return "Clean" if v<.3 else "Suspicious" if v<.6 else "Anomalous"
        sd=[["Forensic Signal","Score","Weight","Assessment"],
            ["Error Level Analysis (ELA)",f"{res['ela']['score']*100:.1f}%","45%",_ass(res['ela']['score'])],
            ["Metadata Forensics",f"{res['metadata']['score']*100:.1f}%","25%",_ass(res['metadata']['score'])],
            ["Noise Consistency",f"{res['noise']['score']*100:.1f}%","30%",_ass(res['noise']['score'])],
            ["WEIGHTED COMBINED",f"{rk*100:.1f}%","100%",
             "Low Risk" if rk<.3 else "Moderate" if rk<.6 else "High Risk"]]
        st2=Table(sd,colWidths=[160,70,60,170])
        st2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),W),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,0),(2,-1),"CENTER"),
            ("GRID",(0,0),(-1,-1),.4,LG),("ROWBACKGROUNDS",(0,1),(-1,-2),[W,LBG]),
            ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#F0F4FF")),
            ("FONTNAME",(0,-1),(-1,-1),"Helvetica-Bold"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.append(st2)
        story.append(Spacer(1,10))
        story.append(Paragraph("Metadata Findings",s["BD"]))
        for f in res.get("metadata",{}).get("findings",[]):
            story.append(Paragraph(f"&bull; {f}",s["BD"]))
        if res.get("metadata",{}).get("exif"):
            story.append(Spacer(1,8)); story.append(Paragraph("Raw EXIF Data",s["BD"]))
            er=[["Tag","Value"]]
            for k,v in res["metadata"]["exif"].items():
                er.append([str(k)[:40],str(v)[:60]])
            if len(er)>1:
                et=Table(er,colWidths=[150,310])
                et.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),W),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Courier"),
                    ("FONTSIZE",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),.3,LG),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[W,LBG]),
                    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
                story.append(et)

    # ── 4. METHODOLOGY ──
    story.append(Paragraph("4. Methodology",s["SH"]))
    if at=="deepfake":
        mt=("Examination performed using a pretrained CNN classifier "
            f"({mn}). Face regions detected via cascade classifier, isolated, preprocessed "
            "(resize, normalization), and classified. Grad-CAM generated for explainability."
            if mode=="model" else
            "Examination performed using frequency-domain heuristic: (1) HF/LF energy ratio "
            "in FFT spectrum, (2) local noise variance, (3) edge coefficient of variation, "
            "(4) histogram smoothness. Indicative signals only — not a trained classifier output.")
    else:
        mt=("Three independent forensic signals combined: <b>ELA</b> — resave at known JPEG "
            "quality, measure pixel differences for inconsistent recompression. <b>EXIF Forensics</b> "
            "— inspect for editing signatures, missing fields, timestamp issues. <b>Noise Consistency</b> "
            "— block-level noise variance to detect spliced regions with mismatched sensor noise. "
            "Weighted: ELA 45%, Metadata 25%, Noise 30%.")
    story.append(Paragraph(mt,s["BD"]))

    # ── 5. CERTIFICATION ──
    story.append(Spacer(1,12))
    story.append(Paragraph("5. Certification Statement",s["SH"]))
    story.append(HRFlowable(width="100%",thickness=.5,color=LG,spaceAfter=8))
    for ln in [
        "I certify that the forensic examination described in this report was conducted in "
        "accordance with established digital forensics procedures. The analysis was performed "
        "by an automated AI-powered system using the methodologies described in Section 4. "
        "All results are based on algorithmic analysis of the submitted digital evidence.",
        "",
        "This report is intended for informational and educational purposes. The automated "
        "system should not be considered a substitute for manual forensic examination by a "
        "qualified digital forensics examiner. Results represent probabilistic assessments "
        "and supporting forensic signals, not definitive proof of authenticity or manipulation."]:
        if ln: story.append(Paragraph(ln,s["CT"]))
        else: story.append(Spacer(1,4))

    story.append(Spacer(1,24))

    # Signature block
    sig=Table([
        ["","Examination System","","Date"],
        ["","AI Media Forensics Suite","",datetime.now().strftime("%d %B %Y")],
        ["","Automated Forensic Analysis","",f"Case {cid}"],
    ],colWidths=[15,155,15,120])
    sig.setStyle(TableStyle([
        ("FONTNAME",(1,0),(1,-1),"Helvetica-Bold"),("FONTNAME",(3,0),(3,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),9),("TEXTCOLOR",(0,0),(-1,-1),NAVY),
        ("LINEABOVE",(1,0),(1,0),1,NAVY),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(sig)

    # FORENSIC STAMP
    story.append(Spacer(1,16))
    stamp=Table([["FORENSIC LABORATORY"],["CERTIFIED REPORT"]],colWidths=[180])
    stamp.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(0,0),14),
        ("FONTSIZE",(0,1),(0,1),9),("TEXTCOLOR",(0,0),(-1,-1),STAMP),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("BOX",(0,0),(-1,-1),2.5,STAMP),
        ("INNERGRID",(0,0),(-1,-1),.5,STAMP),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(stamp)

    # DISCLAIMER
    story.append(Spacer(1,14))
    story.append(HRFlowable(width="100%",thickness=.5,color=LG,spaceAfter=6))
    story.append(Paragraph(
        "<i>DISCLAIMER: This report was generated by an automated academic/demonstration prototype. "
        "Results are forensic-indicative signals for educational and research purposes only. "
        "They do not constitute forensic-grade certification and should not be used as legal evidence "
        "without independent verification by a qualified digital forensics examiner.</i>",s["BS"]))

    def onp(c,d): _hf(c,d,cid)
    doc.build(story,onFirstPage=onp,onLaterPages=onp)
    return op