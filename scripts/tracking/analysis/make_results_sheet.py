"""Build tracking_experiments.xlsx (Google Sheets compatible) pre-filled with the runs scored so far."""
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.comments import Comment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter as L

BASE = '/home/teiszler/Videos/ddil_tracking/analysis/'
OUT = '/home/teiszler/Videos/ddil_tracking/tracking_experiments.xlsx'
HH = 540
FR = [0.2, 0.5, 1.0]
NROWS = 200  # data rows available in Runs / FP Events

F = 'Arial'
f_in = Font(name=F, color='0000FF')            # inputs: blue
f_calc = Font(name=F, color='000000')          # formulas: black
f_link = Font(name=F, color='008000')          # cross-sheet links: green
f_hdr = Font(name=F, bold=True, color='FFFFFF')
f_grp = Font(name=F, bold=True, color='000000')
f_title = Font(name=F, bold=True, size=14)
f_body = Font(name=F)
f_bold = Font(name=F, bold=True)
fill_hdr = PatternFill('solid', fgColor='305496')
fill_grp = {'info': PatternFill('solid', fgColor='D9E1F2'), 'video': PatternFill('solid', fgColor='E2EFDA'),
            'engine': PatternFill('solid', fgColor='FCE4D6'), 'notes': PatternFill('solid', fgColor='EDEDED')}
fill_input = PatternFill('solid', fgColor='FFFF99')
thin = Side(style='thin', color='BFBFBF')
box = Border(left=thin, right=thin, top=thin, bottom=thin)
wrap = Alignment(wrap_text=True, vertical='top')
center = Alignment(horizontal='center', vertical='center', wrap_text=True)


# ---------------------------------------------------------------- measured data
def run_metrics(csv):
    d = pd.read_csv(BASE + csv)
    x = d.x.interpolate(limit_direction='both').values
    y = d.y.interpolate(limit_direction='both').values
    e = np.hypot(x - 960, y - 540)
    g = (d.detected == 0).astype(int).values
    run = best = 0
    for v in g:
        run = run + 1 if v else 0; best = max(best, run)
    m = {'cov': d.detected.mean(), 'gap': best * 1001 / 30000,
         'rmse': np.sqrt((e ** 2).mean()), 'p95': np.percentile(e, 95),
         'inframe': ((x >= 0) & (x < 1920) & (y >= 0) & (y < 1080)).mean()}
    for f in FR:
        R = f * HH
        m[f'auc{f}'] = 100 * (1 - np.minimum(e, R).mean() / R)
        m[f'win{f}'] = (e <= R).mean()
    return m

# response delays from latency.py (yaw, pitch), seconds
DELAY = {'A': (1.03, 1.13), 'B': (1.17, 1.03), 'C': (1.27, 1.27),
         '1': (1.07, 1.23), '2': (1.13, 1.13), '3': (1.33, 1.37)}

CFG_RF = 'Offboard RF-DETR (remote)'
CFG_YOLO = 'Onboard YOLO26s quantized (Tachyon)'
CFG_OFFN = 'Offboard - normal (Nov 10)'
RUNS = [
    # id, date, video, start, config, model, location, host, notes, csv, key
    ('2026-11-09-A', '2026-11-09', 'nov09.MP4', '2:05', CFG_RF, 'RF-DETR', 'Offboard', 'remote', '', 'recheck/runA.csv', 'A'),
    ('2026-11-09-B', '2026-11-09', 'nov09.MP4', '4:45', CFG_YOLO, 'YOLO26s (quantized)', 'Onboard', 'Tachyon (heisenberg?)',
     'Gimbal stalled ~44-51 s (camera nearly still, robot drifted ~390 px).', 'recheck/runB.csv', 'B'),
    ('2026-11-09-C', '2026-11-09', 'nov09.MP4', '7:32', CFG_YOLO, 'YOLO26s (quantized)', 'Onboard', 'Tachyon (heisenberg?)',
     'Steady ~5.5 s oscillation on both axes (underdamped).', 'recheck/runC.csv', 'C'),
    ('2026-11-10-1', '2026-11-10', 'offboard_normal_runs1to3.MP4', '1:57', CFG_OFFN, '', 'Offboard', '',
     'Harsh low sun.', 'nov10/run1.csv', '1'),
    ('2026-11-10-2', '2026-11-10', 'offboard_normal_runs1to3.MP4', '4:07', CFG_OFFN, '', 'Offboard', '',
     'Harsh low sun.', 'nov10/run2.csv', '2'),
    ('2026-11-10-3', '2026-11-10', 'offboard_normal_runs1to3.MP4', '6:22', CFG_OFFN, '', 'Offboard', '',
     'Starts with robot ~400 px off centre; robot often backlit/dark; two ~5-6 s ground-truth gaps (robot on pad).',
     'nov10/run3.csv', '3'),
]

wb = Workbook()

# ---------------------------------------------------------------- Settings
st = wb.active; st.title = 'Settings'
st['A1'] = 'Settings'; st['A1'].font = f_title
rows = [
    ('Frame width (px)', 1920, 'Video resolution used for scoring.'),
    ('Frame height (px)', 1080, ''),
    ('Half-height H/2 (px)', '=B4/2', 'Radii are expressed as fractions of this, so scores do not depend on resolution.'),
    ('Radius r1 (x H/2)', 0.2, '"Tight centring" radius.'),
    ('Radius r2 (x H/2)', 0.5, 'Intermediate radius.'),
    ('Radius r3 (x H/2)', 1.0, 'Target at top/bottom edge of frame.'),
    ('Video frame rate (fps)', 29.97, ''),
    ('Ground-truth detector', 'track.py v2 (colour + local contrast)', 'Detector used to locate the robot in the video for scoring.'),
]
st['A2'] = 'Setting'; st['B2'] = 'Value'; st['C2'] = 'px equivalent'; st['D2'] = 'Note'
for c in 'ABCD':
    st[f'{c}2'].font = f_hdr; st[f'{c}2'].fill = fill_hdr
for i, (k, v, note) in enumerate(rows, start=3):
    st[f'A{i}'] = k; st[f'A{i}'].font = f_body
    st[f'B{i}'] = v
    st[f'B{i}'].font = f_calc if isinstance(v, str) and v.startswith('=') else f_in
    if not (isinstance(v, str) and v.startswith('=')):
        st[f'B{i}'].fill = fill_input
    st[f'D{i}'] = note; st[f'D{i}'].font = f_body
for i in (6, 7, 8):
    st[f'C{i}'] = f'=B{i}*$B$5'; st[f'C{i}'].font = f_calc; st[f'C{i}'].number_format = '0'
    st[f'B{i}'].number_format = '0.0'
st['B9'].number_format = '0.00'
st.column_dimensions['A'].width = 26; st.column_dimensions['B'].width = 36
st.column_dimensions['C'].width = 14; st.column_dimensions['D'].width = 80
st['A13'] = 'Legend'; st['A13'].font = f_bold
st['A14'] = 'Blue text on yellow'; st['A14'].font = f_in; st['A14'].fill = fill_input
st['B14'] = 'Input: type or paste values here'; st['B14'].font = f_body
st['A15'] = 'Black text'; st['A15'].font = f_calc
st['B15'] = 'Formula: do not overwrite'; st['B15'].font = f_body
st['A16'] = 'Green text'; st['A16'].font = f_link
st['B16'] = 'Formula pulling from another tab'; st['B16'].font = f_body

R1, R2, R3 = 'Settings!$B$6', 'Settings!$B$7', 'Settings!$B$8'

# ---------------------------------------------------------------- Runs
ws = wb.create_sheet('Runs')
cols = [
    # (group, header, width, kind, number_format, comment)
    ('info', 'Run ID', 15, 'in', '@', 'Unique ID, e.g. YYYY-MM-DD-<n>. Used by FP Events.'),
    ('info', 'Date', 11, 'in', 'yyyy-mm-dd', ''),
    ('info', 'Video file', 26, 'in', '@', ''),
    ('info', 'Start (m:ss)', 9, 'in', '@', 'Run start within the video.'),
    ('info', 'Duration (s)', 9, 'in', '0', ''),
    ('info', 'Configuration', 30, 'in', '@', 'Must match a name on the Summary tab to be aggregated.'),
    ('info', 'Detector model', 18, 'in', '@', ''),
    ('info', 'Inference location', 11, 'in', '@', 'Onboard / Offboard'),
    ('info', 'Compute host', 16, 'in', '@', ''),
    ('info', 'Model input res', 10, 'in', '@', 'e.g. 640x640'),
    ('info', 'socd throttle disabled?', 10, 'in', '@', 'Was the socd trip point raised to 101 on the onboard computer (Yes/No/Unknown)? Throttling can slow onboard inference.'),
    ('info', 'Conditions', 16, 'in', '@', 'Light, wind, altitude, etc.'),
    ('video', 'GT coverage', 9, 'in', '0.0%', 'Fraction of frames in which the ground-truth detector found the robot (rest interpolated).'),
    ('video', 'Longest GT gap (s)', 9, 'in', '0.0', ''),
    ('video', 'AUC @ r1', 9, 'in', '0.0', 'Area under the precision curve from 0 to r1 (0-100). = 100*(1 - mean(min(err, R))/R).'),
    ('video', 'AUC @ r2', 9, 'in', '0.0', 'Area under the precision curve from 0 to r2 (0-100).'),
    ('video', 'AUC @ r3', 9, 'in', '0.0', 'Area under the precision curve from 0 to r3 (0-100).'),
    ('video', 'Mean AUC', 9, 'calc', '0.0', 'Average of the three AUC radii.'),
    ('video', '% within r1', 9, 'in', '0.0%', 'Pass rate: fraction of frames with error <= r1.'),
    ('video', '% within r2', 9, 'in', '0.0%', ''),
    ('video', '% within r3', 9, 'in', '0.0%', ''),
    ('video', '% in frame', 9, 'in', '0.0%', 'Fraction of frames with the robot centre inside the image.'),
    ('video', 'RMSE (px)', 9, 'in', '0', ''),
    ('video', 'P95 error (px)', 9, 'in', '0', ''),
    ('video', 'Resp. delay yaw (s)', 9, 'in', '0.00', 'From latency.py: lag between error and camera pan (video-derived; includes controller dynamics).'),
    ('video', 'Resp. delay pitch (s)', 9, 'in', '0.00', ''),
    ('engine', 'Frames processed', 10, 'in', '#,##0', 'From detection-engine logs: number of frames the engine ran inference on.'),
    ('engine', 'Engine rate (Hz)', 9, 'calc', '0.0', 'Frames processed / duration.'),
    ('engine', 'Total detections', 10, 'in', '#,##0', 'All target-class detections the engine output (above its confidence threshold).'),
    ('engine', 'True-positive detections', 10, 'in', '#,##0', 'Detections that matched the robot (see Definitions).'),
    ('engine', 'False-positive detections', 10, 'calc', '#,##0', 'Total detections - true positives.'),
    ('engine', 'Missed frames', 10, 'in', '#,##0', 'Frames where the robot was visible but the engine produced no true positive.'),
    ('engine', 'FP per minute', 9, 'calc', '0.0', ''),
    ('engine', 'FP per 100 frames', 9, 'calc', '0.0', ''),
    ('engine', 'Precision', 9, 'calc', '0.0%', 'TP / total detections.'),
    ('engine', 'Recall', 9, 'calc', '0.0%', 'TP frames / (TP frames + missed frames). Assumes at most one TP per frame.'),
    ('engine', 'FP episodes', 9, 'link', '0', 'Count of rows for this run on the FP Events tab.'),
    ('engine', 'Time gimbal followed an FP (s)', 11, 'link', '0.0', 'Sum of FP Events durations where the gimbal steered toward the false target.'),
    ('notes', 'Notes', 50, 'in', '@', ''),
]
idx = {h: i + 1 for i, (_, h, *_r) in enumerate(cols)}
C = lambda h: L(idx[h])
groups = {'info': 'RUN INFO', 'video': 'VIDEO-BASED TRACKING METRICS (from auc.py / latency.py)',
          'engine': 'DETECTION-ENGINE METRICS (from engine logs)', 'notes': ''}
# group header row 1
start = 1
for gkey in ['info', 'video', 'engine', 'notes']:
    n = sum(1 for c in cols if c[0] == gkey)
    ws.cell(1, start, groups[gkey]).font = f_grp
    for j in range(start, start + n):
        ws.cell(1, j).fill = fill_grp[gkey]
    if n > 1:
        ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=start + n - 1)
    ws.cell(1, start).alignment = center
    start += n
for j, (g, h, w, kind, nf, com) in enumerate(cols, start=1):
    c = ws.cell(2, j, h); c.font = f_hdr; c.fill = fill_hdr; c.alignment = center; c.border = box
    if com:
        c.comment = Comment(com, 'tracking-sheet')
    ws.column_dimensions[L(j)].width = w
ws.row_dimensions[2].height = 45
# radius px in headers via formulas would break header text; keep labels and put px in a helper row note
FIRST, LAST = 3, 2 + NROWS


def formula(h, r):
    D = f'{C("Duration (s)")}{r}'
    if h == 'Mean AUC':
        a = [f'{C(x)}{r}' for x in ('AUC @ r1', 'AUC @ r2', 'AUC @ r3')]
        return f'=IF(COUNT({a[0]},{a[1]},{a[2]})=0,"",AVERAGE({a[0]},{a[1]},{a[2]}))'
    if h == 'Engine rate (Hz)':
        fp = f'{C("Frames processed")}{r}'
        return f'=IF(OR({fp}="",{D}=""),"",{fp}/{D})'
    if h == 'False-positive detections':
        t, tp = f'{C("Total detections")}{r}', f'{C("True-positive detections")}{r}'
        return f'=IF(OR({t}="",{tp}=""),"",{t}-{tp})'
    fpc = f'{C("False-positive detections")}{r}'
    if h == 'FP per minute':
        return f'=IF(OR({fpc}="",{D}=""),"",{fpc}/({D}/60))'
    if h == 'FP per 100 frames':
        fr = f'{C("Frames processed")}{r}'
        return f'=IF(OR({fpc}="",{fr}="",{fr}=0),"",100*{fpc}/{fr})'
    if h == 'Precision':
        t, tp = f'{C("Total detections")}{r}', f'{C("True-positive detections")}{r}'
        return f'=IF(OR({t}="",{tp}="",{t}=0),"",{tp}/{t})'
    if h == 'Recall':
        tp, m = f'{C("True-positive detections")}{r}', f'{C("Missed frames")}{r}'
        return f'=IF(OR({tp}="",{m}="",{tp}+{m}=0),"",{tp}/({tp}+{m}))'
    rid = f'$A{r}'
    if h == 'FP episodes':
        return f'=IF({rid}="","",COUNTIF(\'FP Events\'!$A$3:$A${LAST},{rid}))'
    if h == 'Time gimbal followed an FP (s)':
        return (f'=IF({rid}="","",SUMIFS(\'FP Events\'!$D$3:$D${LAST},\'FP Events\'!$A$3:$A${LAST},{rid},'
                f'\'FP Events\'!$F$3:$F${LAST},"Yes"))')
    raise KeyError(h)


for r in range(FIRST, LAST + 1):
    for j, (g, h, w, kind, nf, com) in enumerate(cols, start=1):
        c = ws.cell(r, j); c.number_format = nf; c.border = box
        if kind == 'in':
            c.font = f_in
        else:
            c.value = formula(h, r); c.font = f_link if kind == 'link' else f_calc

# pre-fill measured runs
for i, (rid, date, vid, startt, cfg, model, loc, host, note, csv, key) in enumerate(RUNS):
    r = FIRST + i; m = run_metrics(csv)
    vals = {'Run ID': rid, 'Date': pd.Timestamp(date).to_pydatetime(), 'Video file': vid, 'Start (m:ss)': startt,
            'Duration (s)': 60, 'Configuration': cfg, 'Detector model': model or None, 'Inference location': loc,
            'Compute host': host or None, 'socd throttle disabled?': 'Unknown',
            'Conditions': 'Harsh low sun' if date.endswith('10') else None,
            'GT coverage': m['cov'], 'Longest GT gap (s)': m['gap'],
            'AUC @ r1': m['auc0.2'], 'AUC @ r2': m['auc0.5'], 'AUC @ r3': m['auc1.0'],
            '% within r1': m['win0.2'], '% within r2': m['win0.5'], '% within r3': m['win1.0'],
            '% in frame': m['inframe'], 'RMSE (px)': m['rmse'], 'P95 error (px)': m['p95'],
            'Resp. delay yaw (s)': DELAY[key][0], 'Resp. delay pitch (s)': DELAY[key][1], 'Notes': note or None}
    for h, v in vals.items():
        if v is not None:
            ws.cell(r, idx[h], round(v, 4) if isinstance(v, float) else v)

ws.freeze_panes = 'B3'
for h in ('AUC @ r1', 'AUC @ r2', 'AUC @ r3', 'Mean AUC'):
    ws.conditional_formatting.add(f'{C(h)}{FIRST}:{C(h)}{LAST}', ColorScaleRule(
        start_type='num', start_value=0, start_color='F8696B', mid_type='num', mid_value=50, mid_color='FFEB84',
        end_type='num', end_value=100, end_color='63BE7B'))
dv_loc = DataValidation(type='list', formula1='"Onboard,Offboard"', allow_blank=True)
dv_yn = DataValidation(type='list', formula1='"Yes,No,Unknown"', allow_blank=True)
dv_cfg = DataValidation(type='list', formula1=f'Summary!$A$4:$A$23', allow_blank=True,
                        showErrorMessage=False)  # warn-free so new configs can be typed before adding to Summary
for dv, h in ((dv_loc, 'Inference location'), (dv_yn, 'socd throttle disabled?'), (dv_cfg, 'Configuration')):
    ws.add_data_validation(dv); dv.add(f'{C(h)}{FIRST}:{C(h)}{LAST}')

# ---------------------------------------------------------------- FP Events
fe = wb.create_sheet('FP Events')
fe['A1'] = ('Log of distinct false-positive episodes found in detection-engine output '
            '(one row per episode, e.g. engine boxed a landing pad for 2 s).')
fe['A1'].font = f_bold
fe.merge_cells('A1:H1')
fh = [('Run ID', 15, '@'), ('Start (s into run)', 10, '0.0'), ('End (s into run)', 10, '0.0'),
      ('Duration (s)', 10, '0.0'), ('Mistaken for', 18, '@'), ('Gimbal followed it?', 11, '@'),
      ('Source', 14, '@'), ('Notes', 60, '@')]
for j, (h, w, nf) in enumerate(fh, start=1):
    c = fe.cell(2, j, h); c.font = f_hdr; c.fill = fill_hdr; c.alignment = center; c.border = box
    fe.column_dimensions[L(j)].width = w
fe.row_dimensions[2].height = 32
for r in range(3, LAST + 1):
    for j, (h, w, nf) in enumerate(fh, start=1):
        c = fe.cell(r, j); c.number_format = nf; c.border = box; c.font = f_in
    fe[f'D{r}'] = f'=IF(OR(B{r}="",C{r}=""),"",C{r}-B{r})'; fe[f'D{r}'].font = f_calc
# example row (suspected, from video review)
for col, v in zip('ABCEFGH', ['2026-11-09-B', 29.5, 31.5, 'Landing pad', 'Yes', 'Video review',
                              'EXAMPLE / suspected: camera centred on the landing pad while the robot stood beside it. '
                              'Confirm against engine logs or delete this row.']):
    fe[f'{col}3'] = v
fe.freeze_panes = 'A3'
dv_run = DataValidation(type='list', formula1=f'Runs!$A${FIRST}:$A${LAST}', allow_blank=True)
dv_what = DataValidation(type='list', formula1='"Landing pad,Shadow,Vehicle,Person,Trailer/structure,Grass/glare,Other"',
                         allow_blank=True, showErrorMessage=False)
dv_yn2 = DataValidation(type='list', formula1='"Yes,No"', allow_blank=True)
dv_src = DataValidation(type='list', formula1='"Engine log,Video review"', allow_blank=True)
for dv, col in ((dv_run, 'A'), (dv_what, 'E'), (dv_yn2, 'F'), (dv_src, 'G')):
    fe.add_data_validation(dv); dv.add(f'{col}3:{col}{LAST}')

# ---------------------------------------------------------------- Summary
sm = wb.create_sheet('Summary', 1)
sm['A1'] = 'Summary by configuration (mean and sample SD across runs)'; sm['A1'].font = f_title
sm['A2'] = 'Type configuration names in column A (yellow); they must match the Configuration column on the Runs tab.'
sm['A2'].font = Font(name=F, italic=True)
sh = ['Configuration', 'Runs', 'AUC @ r1', 'SD', 'AUC @ r2', 'SD', 'AUC @ r3', 'SD', 'Mean AUC', 'SD',
      '% within r1', '% within r2', 'FP per minute', 'Precision', 'Recall', 'FP episodes', 'Time following FPs (s)']
for j, h in enumerate(sh, start=1):
    c = sm.cell(3, j, h); c.font = f_hdr; c.fill = fill_hdr; c.alignment = center; c.border = box
    sm.column_dimensions[L(j)].width = 34 if j == 1 else 10
sm.row_dimensions[3].height = 32
RG = lambda h: f'Runs!${C(h)}${FIRST}:${C(h)}${LAST}'
CFGR = RG('Configuration')


def avg(h, r):
    return f'=IFERROR(AVERAGEIFS({RG(h)},{CFGR},$A{r}),"")'


def sd(h, r):
    # sample SD via sums: SUMPRODUCT with comma-separated arrays treats text/"" as 0, so blanks are safe
    v = RG(h); n = f'COUNTIFS({CFGR},$A{r},{v},">-1E+300")'
    ss = f'SUMPRODUCT(--({CFGR}=$A{r}),{v},{v})'
    sm_ = f'SUMIFS({v},{CFGR},$A{r})'
    return f'=IFERROR(IF({n}<2,"",SQRT(MAX(0,({ss}-{sm_}^2/{n})/({n}-1)))),"")'


for r in range(4, 24):
    sm[f'A{r}'].font = f_in; sm[f'A{r}'].fill = fill_input
    sm[f'B{r}'] = f'=IF($A{r}="","",COUNTIFS({CFGR},$A{r},{RG("Run ID")},"<>"))'
    spec = [('C', 'AUC @ r1', avg, '0.0'), ('D', 'AUC @ r1', sd, '0.0'), ('E', 'AUC @ r2', avg, '0.0'),
            ('F', 'AUC @ r2', sd, '0.0'), ('G', 'AUC @ r3', avg, '0.0'), ('H', 'AUC @ r3', sd, '0.0'),
            ('I', 'Mean AUC', avg, '0.0'), ('J', 'Mean AUC', sd, '0.0'),
            ('K', '% within r1', avg, '0.0%'), ('L', '% within r2', avg, '0.0%'),
            ('M', 'FP per minute', avg, '0.0'), ('N', 'Precision', avg, '0.0%'), ('O', 'Recall', avg, '0.0%')]
    for col, h, fn, nf in spec:
        sm[f'{col}{r}'] = f'=IF($A{r}="","",{fn(h, r)[1:]})'
        sm[f'{col}{r}'].number_format = nf
    sm[f'P{r}'] = f'=IF($A{r}="","",SUMIFS({RG("FP episodes")},{CFGR},$A{r}))'
    sm[f'Q{r}'] = f'=IF($A{r}="","",SUMIFS({RG("Time gimbal followed an FP (s)")},{CFGR},$A{r}))'
    sm[f'Q{r}'].number_format = '0.0'
    for col in 'BCDEFGHIJKLMNOPQ':
        sm[f'{col}{r}'].font = f_link
    for col in 'ABCDEFGHIJKLMNOPQ':
        sm[f'{col}{r}'].border = box
for i, cfg in enumerate([CFG_RF, CFG_YOLO, CFG_OFFN]):
    sm[f'A{4 + i}'] = cfg
for col in 'CEGI':
    sm.conditional_formatting.add(f'{col}4:{col}23', ColorScaleRule(
        start_type='num', start_value=0, start_color='F8696B', mid_type='num', mid_value=50, mid_color='FFEB84',
        end_type='num', end_value=100, end_color='63BE7B'))
sm.freeze_panes = 'B4'
sm['A25'] = ('SD is the sample standard deviation across runs (blank with fewer than 2 runs). '
             'With 1-3 runs per configuration, differences of a few AUC points are within run-to-run variation.')
sm['A25'].font = Font(name=F, italic=True)

# ---------------------------------------------------------------- Definitions
df = wb.create_sheet('Definitions')
df['A1'] = 'Metric definitions'; df['A1'].font = f_title
defs = [
    ('Pointing error e(t)', 'Distance in pixels from the robot centre (ground-truth detector) to the image centre, per video frame. Frames where the ground-truth detector misses the robot are linearly interpolated.'),
    ('Radii r1, r2, r3', 'Expressed as fractions of the frame half-height (Settings tab). At 1080p: 0.2 = 108 px, 0.5 = 270 px, 1.0 = 540 px (target at top/bottom edge).'),
    ('AUC @ R (primary)', 'Area under the precision curve P(r) = fraction of frames with e <= r, for r from 0 to R, scaled 0-100. Equivalent to 100 x (1 - mean(min(e, R)) / R). Reaches 100 only if the target is exactly centred in every frame; it measures average centring quality, not whether the target stayed in frame.'),
    ('% within R', 'Pass rate: fraction of frames with e <= R.'),
    ('% in frame', 'Fraction of frames with the robot centre inside the image. A track-loss check.'),
    ('Response delay', 'Lag (s) maximising correlation between pointing error and camera pan rate (latency.py). Includes controller dynamics, so it is an upper bound on pipeline latency.'),
    ('True-positive detection', 'An engine detection matching the robot. Suggested rule: IoU >= 0.3 with the ground-truth robot box, or box centre within max(50 px, 1 robot length) of the ground-truth centre. At most one TP per frame; extra boxes on the robot count as FP.'),
    ('False-positive detection', 'Any engine detection that is not a TP (e.g. landing pad, shadow, vehicle). Count per detection; Total - TP.'),
    ('Missed frame', 'Frame processed by the engine where the robot was visible but no TP was output.'),
    ('FP episode', 'A contiguous stretch of FPs on the same wrong object (log on FP Events). Mark "Gimbal followed it?" = Yes if the gimbal visibly steered toward it; these are the most damaging FPs.'),
    ('What the engine logs need', 'Per processed frame: capture timestamp, inference start/end timestamps, every output box (class, confidence, x, y, w, h), and which box (if any) the controller used. This enables FP/TP counts, recall, and true pipeline latency.'),
]
df['A2'] = 'Metric'; df['B2'] = 'Definition'
for c in 'AB':
    df[f'{c}2'].font = f_hdr; df[f'{c}2'].fill = fill_hdr
for i, (k, v) in enumerate(defs, start=3):
    df[f'A{i}'] = k; df[f'A{i}'].font = f_bold; df[f'A{i}'].alignment = wrap
    df[f'B{i}'] = v; df[f'B{i}'].font = f_body; df[f'B{i}'].alignment = wrap
df.column_dimensions['A'].width = 28; df.column_dimensions['B'].width = 120

# tab order: Runs, Summary, FP Events, Definitions, Settings
wb._sheets = [wb['Runs'], wb['Summary'], wb['FP Events'], wb['Definitions'], wb['Settings']]
wb.active = 0
for s in wb.worksheets:
    s.sheet_view.zoomScale = 100
wb.save(OUT)
print('saved', OUT)
