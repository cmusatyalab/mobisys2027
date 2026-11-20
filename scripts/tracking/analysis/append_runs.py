"""Append scored runs to tracking_experiments.xlsx in place (keeps anything typed into the workbook).

Usage:
  python3 append_runs.py BATCH
where BATCH is a key in BATCHES below. Each run is skipped if its Run ID already exists.
"""
import sys, copy
import numpy as np, pandas as pd, openpyxl
from auc import metrics

BASE = '/home/teiszler/Videos/ddil_tracking/analysis/'
XLSX = '/home/teiszler/Videos/ddil_tracking/tracking_experiments.xlsx'
FPS = 30000 / 1001


def delays(track_csv, cam_csv):
    """(yaw, pitch) lag in s maximising corr(error, camera pan rate) -- same method as latency.py"""
    from scipy.ndimage import uniform_filter1d
    d = pd.read_csv(track_csv); c = pd.read_csv(cam_csv)
    ex = d.x.interpolate(limit_direction='both').values - 960
    ey = d.y.interpolate(limit_direction='both').values - 540
    px = -uniform_filter1d(c.dx.values, 9) * FPS; py = -uniform_filter1d(c.dy.values, 9) * FPS
    out = []
    for e, p in ((ex, px), (ey, py)):
        e0 = e - e.mean(); p0 = p - p.mean(); lags = np.arange(0, int(3 * FPS))
        cc = np.array([np.corrcoef(e0[:len(e0) - L], p0[L:])[0, 1] for L in lags])
        out.append((lags[cc.argmax()] / FPS, cc.max()))
    return out


# run id, video, start, config, location, key (csv suffix), notes
BATCHES = {
    'nov13a': dict(date='2026-11-13', video='nov13/P7251059.MP4', dir='nov13', runs=[
        ('2026-11-13-ambient1', '0:56', 'Offboard - ambient', 'Offboard', 'ambient1', ''),
        ('2026-11-13-ambient2', '3:54', 'Offboard - ambient', 'Offboard', 'ambient2',
         'Robot left frame ~42-46 s (gimbal swung away; exited bottom-left).'),
        ('2026-11-13-ambient3', '7:02', 'Offboard - ambient', 'Offboard', 'ambient3', ''),
        ('2026-11-13-delayed1', '9:44', 'Offboard - delayed', 'Offboard', 'delayed1',
         'Robot left top of frame ~2-3 s at start.'),
        ('2026-11-13-delayed2', '11:50', 'Offboard - delayed', 'Offboard', 'delayed2',
         'Robot left top of frame ~2-3 s at start.'),
        ('2026-11-13-delayed3', '13:53', 'Offboard - delayed', 'Offboard', 'delayed3',
         'Robot left top of frame ~2-4 s at start (2 episodes).'),
    ]),
    'nov13b': dict(date='2026-11-13', video='nov13/P7261060.MP4', dir='nov13', runs=[
        ('2026-11-13-limited1', '1:34', 'Offboard - limited', 'Offboard', 'limited1', ''),
        ('2026-11-13-limited2', '3:47', 'Offboard - limited', 'Offboard', 'limited2', ''),
        ('2026-11-13-limited3', '6:08', 'Offboard - limited', 'Offboard', 'limited3', ''),
        ('2026-11-13-onboard-none1', '10:05', 'Onboard - none', 'Onboard', 'none', ''),
        ('2026-11-13-onboard-none2', '12:43', 'Onboard - none', 'Onboard', 'none2', ''),
        ('2026-11-13-onboard-none3', '15:02', 'Onboard - none', 'Onboard', 'none3', ''),
    ]),
    'nov13c': dict(date='2026-11-13', video='nov13/P7291063.MP4', dir='nov13', runs=[
        ('2026-11-13-cop-delayed1', '0:41', 'COP - delayed', None, 'cop-delayed1',
         'Robot left frame ~9.2-14.5 s (gimbal swung right; exited left).'),
        ('2026-11-13-cop-delayed2', '3:33', 'COP - delayed', None, 'cop-delayed2', ''),
        ('2026-11-13-cop-delayed3', '6:17', 'COP - delayed', None, 'cop-delayed3', ''),
        ('2026-11-13-cop-limited1', '9:07', 'COP - limited', None, 'cop-limited1',
         'Gimbal lagged robot walking left ~44-52 s (up to ~400 px off centre).'),
    ]),
    'nov17a': dict(date='2026-11-17', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov17/offboard_amb.MP4', dir='nov17',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)',
               fpfollow={'offboard-ambient1': 32.0, 'offboard-ambient2': 6.9, 'offboard-ambient3': 6.7}, runs=[
        ('2026-11-17-offboard-ambient1', '0:42', 'Offboard - ambient', 'Offboard', 'offboard-ambient1', 'Gimbal switched to the chair ~30 s; robot out of frame ~35 s to end.'),
        ('2026-11-17-offboard-ambient2', '2:40', 'Offboard - ambient', 'Offboard', 'offboard-ambient2', ''),
        ('2026-11-17-offboard-ambient3', '4:42', 'Offboard - ambient', 'Offboard', 'offboard-ambient3', ''),
    ]),
    'nov17b': dict(date='2026-11-17', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov17/onboard_none.MP4', dir='nov17',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)',
               fpfollow={'onboard-none1': 17.4, 'onboard-none2': 31.2, 'onboard-none3': 1.8}, runs=[
        ('2026-11-17-onboard-none1', '0:48', 'Onboard - none', 'Onboard', 'onboard-none1', 'Gimbal switched to the chair ~48 s; robot out of frame after that.'),
        ('2026-11-17-onboard-none2', '2:57', 'Onboard - none', 'Onboard', 'onboard-none2', 'Robot walked into the chair ~11-22 s and a person moved the chair (~22 s); GT cannot separate robot from chair ~12-22 s (interpolated). Gimbal stayed on the chair; robot out of frame from ~40 s.'),
        ('2026-11-17-onboard-none3', '5:38', 'Onboard - none', 'Onboard', 'onboard-none3', ''),
    ]),
    'nov17c': dict(date='2026-11-17', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov17/cop_amb.MP4', dir='nov17',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)',
               fpfollow={'cop-ambient1': 0.0, 'cop-ambient2': 3.2, 'cop-ambient3': 3.0}, runs=[
        ('2026-11-17-cop-ambient1', '0:52', 'COP - ambient', None, 'cop-ambient1', ''),
        ('2026-11-17-cop-ambient2', '4:10', 'COP - ambient', None, 'cop-ambient2', ''),
        ('2026-11-17-cop-ambient3', '7:08', 'COP - ambient', None, 'cop-ambient3', ''),
    ]),
    'nov17d': dict(date='2026-11-17', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov17/offboard_del.MP4', dir='nov17',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)',
               fpfollow={'offboard-delayed1': 2.0, 'offboard-delayed2': 29.3, 'offboard-delayed3': 29.5}, runs=[
        ('2026-11-17-offboard-delayed1', '1:01', 'Offboard - delayed', 'Offboard', 'offboard-delayed1', ''),
        ('2026-11-17-offboard-delayed2', '3:05', 'Offboard - delayed', 'Offboard', 'offboard-delayed2', 'Gimbal centred the chair through much of 4-56 s while the robot stayed in frame off-centre.'),
        ('2026-11-17-offboard-delayed3', '5:25', 'Offboard - delayed', 'Offboard', 'offboard-delayed3', 'Gimbal moved to the chair ~6 s; robot out of frame ~7-27 s; gimbal back on robot ~40 s.'),
    ]),
    'nov17e': dict(date='2026-11-17', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov17/offboard_lim.MP4', dir='nov17',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)',
               fpfollow={'offboard-limited1': 20.1, 'offboard-limited2': 49.1, 'offboard-limited3': 19.2}, runs=[
        ('2026-11-17-offboard-limited1', '0:49', 'Offboard - limited', 'Offboard', 'offboard-limited1', 'Gimbal centred the chair for long stretches (5-59 s) with the robot off-centre in frame.'),
        ('2026-11-17-offboard-limited2', '3:10', 'Offboard - limited', 'Offboard', 'offboard-limited2', 'Gimbal locked onto the chair from ~6 s to the end; robot mostly out of frame.'),
        ('2026-11-17-offboard-limited3', '5:29', 'Offboard - limited', 'Offboard', 'offboard-limited3', 'Robot stood still for the first ~14 s. Gimbal moved to the chair ~17 s; robot out of frame ~15-30 s and after ~55 s.'),
    ]),
    'nov17f': dict(date='2026-11-17', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov17/cop_del1.MP4', dir='nov17',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)',
               fpfollow={'cop-delayed1': 0.6}, runs=[
        ('2026-11-17-cop-delayed1', '0:46', 'COP - delayed', None, 'cop-delayed1', 'Robot stood on a landing pad ~41-42 s (GT miss, interpolated).'),
    ]),
    'nov18_onboard_none_med': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/onboard_none_med.MP4', dir='nov18',
               conditions='Onboard YOLO (medium). Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-onboard-none-med1', '0:44', 'Onboard - none (med)', 'Onboard', 'onboard-none-med1', 'Gimbal centred a distractor for 1 long stretch(es), first from ~13 s.'),
        ('2026-11-18-onboard-none-med2', '2:44', 'Onboard - none (med)', 'Onboard', 'onboard-none-med2', 'Gimbal centred a distractor for 1 long stretch(es), first from ~19 s.'),
        ('2026-11-18-onboard-none-med3', '4:40', 'Onboard - none (med)', 'Onboard', 'onboard-none-med3', 'Gimbal centred a distractor for 1 long stretch(es), first from ~4 s.'),
    ]),
    'nov18_onboard_none_small': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/onboard_none_small.MP4', dir='nov18',
               conditions='Onboard YOLO (small). Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-onboard-none-small1', '0:32', 'Onboard - none (small)', 'Onboard', 'onboard-none-small1', 'Gimbal centred a distractor for 2 long stretch(es), first from ~9 s.'),
        ('2026-11-18-onboard-none-small2', '5:40', 'Onboard - none (small)', 'Onboard', 'onboard-none-small2', ''),
        ('2026-11-18-onboard-none-small3', '7:55', 'Onboard - none (small)', 'Onboard', 'onboard-none-small3', 'Gimbal centred a distractor for 2 long stretch(es), first from ~16 s.'),
    ]),
    'nov18_cop_amb_small': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/cop_amb_small.MP4', dir='nov18',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-cop-ambient1', '0:31', 'COP - ambient (Nov 18)', None, 'cop-ambient1', ''),
        ('2026-11-18-cop-ambient2', '2:33', 'COP - ambient (Nov 18)', None, 'cop-ambient2', 'Robot stood still on a landing pad ~44-50 s (GT kept via standstill rule).'),
        ('2026-11-18-cop-ambient3', '4:55', 'COP - ambient (Nov 18)', None, 'cop-ambient3', ''),
    ]),
    'nov18_cop_deg_small': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/cop_deg_small.MP4', dir='nov18',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-cop-degraded1', '0:46', 'COP - degraded (Nov 18)', None, 'cop-degraded1', ''),
        ('2026-11-18-cop-degraded2', '2:56', 'COP - degraded (Nov 18)', None, 'cop-degraded2', ''),
        ('2026-11-18-cop-degraded3', '4:58', 'COP - degraded (Nov 18)', None, 'cop-degraded3', ''),
    ]),
    'nov18_cop_del_small': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/cop_del_small.MP4', dir='nov18',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-cop-delayed1', '0:27', 'COP - delayed (Nov 18)', None, 'cop-delayed1', ''),
        ('2026-11-18-cop-delayed2', '3:03', 'COP - delayed (Nov 18)', None, 'cop-delayed2', 'Gimbal centred a distractor for 1 long stretch(es), first from ~10 s.'),
        ('2026-11-18-cop-delayed3', '5:10', 'COP - delayed (Nov 18)', None, 'cop-delayed3', ''),
    ]),
    'nov18_cop_int_small': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/cop_int_small.MP4', dir='nov18',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-cop-intermittent1', '0:41', 'COP - intermittent (Nov 18)', None, 'cop-intermittent1', ''),
        ('2026-11-18-cop-intermittent2', '3:24', 'COP - intermittent (Nov 18)', None, 'cop-intermittent2', ''),
        ('2026-11-18-cop-intermittent3', '5:54', 'COP - intermittent (Nov 18)', None, 'cop-intermittent3', ''),
    ]),
    'nov18_cop_lim_small': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/cop_lim_small.MP4', dir='nov18',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-cop-limited1', '0:47', 'COP - limited (Nov 18)', None, 'cop-limited1', ''),
        ('2026-11-18-cop-limited2', '3:13', 'COP - limited (Nov 18)', None, 'cop-limited2', ''),
        ('2026-11-18-cop-limited3', '5:44', 'COP - limited (Nov 18)', None, 'cop-limited3', ''),
    ]),
    'nov18_cop_intplusden': dict(date='2026-11-18', xlsx='/home/teiszler/Videos/ddil_tracking/tracking_nov17.xlsx', video='nov18/cop_intplusden.MP4', dir='nov18',
               conditions='Distractors: white chair + gray bag. GT: RF-DETR 2XL (81-class) at 1080p + static/chair rejection (nov17/gt.py)', runs=[
        ('2026-11-18-cop-intdrip1', '0:31', 'COP (intermittent, drip)', None, 'cop-intdrip1', ''),
        ('2026-11-18-cop-intdrip2', '3:12', 'COP (intermittent, drip)', None, 'cop-intdrip2', 'Gimbal briefly on chair ~52 s with robot out of view.'),
        ('2026-11-18-cop-intdrip3', '5:18', 'COP (intermittent, drip)', None, 'cop-intdrip3', ''),
        ('2026-11-18-cop-denied1', '8:31', 'COP (denied)', None, 'cop-denied1', 'Gimbal centred a distractor for 1 long stretch(es), first from ~22 s.'),
        ('2026-11-18-cop-denied2', '10:36', 'COP (denied)', None, 'cop-denied2', ''),
        ('2026-11-18-cop-denied3', '12:40', 'COP (denied)', None, 'cop-denied3', ''),
    ]),
}
NOTES_EXTRA = {}   # run id -> extra note text, filled by caller if needed


def main(batch):
    b = BATCHES[batch]
    xlsx = b.get('xlsx', XLSX)
    wb = openpyxl.load_workbook(xlsx)
    ws = wb['Runs']
    col = {c.value: c.column for c in ws[2]}
    ids = {ws.cell(r, 1).value: r for r in range(3, 203)}
    free = [r for r in range(3, 203) if ws.cell(r, 1).value in (None, '')]
    added = []
    for rid, start, cfg, loc, key, note in b['runs']:
        if rid in ids:
            print('skip (exists):', rid); continue
        d = pd.read_csv(f"{BASE}{b['dir']}/run{key}.csv")
        m = metrics(d)
        (dy, ry), (dp, rp) = delays(f"{BASE}{b['dir']}/run{key}.csv", f"{BASE}{b['dir']}/cam{key}.csv")
        r = free.pop(0)
        oof = m['out of frame s']
        full_note = note
        if oof > 0:
            full_note = (note + ' ' if note else '') + f'Out of frame {oof:.1f} s (counted as miss at all radii).'
        if min(ry, rp) < 0.6:
            full_note += ' Response delay unreliable (weak correlation, e.g. due to track loss).'
        vals = {
            'Run ID': rid, 'Date': pd.Timestamp(b['date']).to_pydatetime(), 'Video file': b['video'].split('/')[-1],
            'Start (m:ss)': start, 'Duration (s)': 60, 'Configuration': cfg, 'Inference location': loc,
            'socd throttle disabled?': 'Unknown',
            'GT coverage': m['detected %'] / 100, 'Longest GT gap (s)': m['longest gap s'],
            'AUC @ r1': m['AUC@0.2'], 'AUC @ r2': m['AUC@0.5'], 'AUC @ r3': m['AUC@1.0'],
            '% within r1': m['% within 0.2'] / 100, '% within r2': m['% within 0.5'] / 100,
            '% within r3': m['% within 1.0'] / 100, '% in frame': m['% in frame'] / 100,
            'RMSE (px)': m['RMSE px (in frame)'], 'P95 error (px)': m['p95 px (in frame)'],
            'Resp. delay yaw (s)': dy if ry >= 0.6 else None, 'Resp. delay pitch (s)': dp if rp >= 0.6 else None,
            'Notes': full_note.strip() or None,
            'Conditions': b.get('conditions'),
            'Time gimbal followed an FP (s)': b.get('fpfollow', {}).get(key),
        }
        for h, v in vals.items():
            cur = ws.cell(r, col[h]).value
            if isinstance(cur, str) and cur.startswith('='):
                continue   # never overwrite a template formula (e.g. FP time sums FP Events)
            if v is not None:
                ws.cell(r, col[h]).value = round(v, 4) if isinstance(v, float) else v
        added.append((r, rid))
    # register new configurations on Summary (first empty slot)
    sm = wb['Summary']
    have = {sm[f'A{i}'].value for i in range(4, 24)}
    for cfg in dict.fromkeys(x[2] for x in b['runs']):
        if cfg not in have:
            slot = next(i for i in range(4, 24) if sm[f'A{i}'].value in (None, ''))
            sm[f'A{slot}'] = cfg; print('added config to Summary:', cfg)
    wb.save(xlsx)
    print('appended:', added)


if __name__ == '__main__':
    main(sys.argv[1])
