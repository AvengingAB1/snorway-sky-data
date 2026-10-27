#!/usr/bin/env python3
"""
extract_meps_elements.py - MEPS 2.5 km 10 m WIND and hourly PRECIP rendered into the
same per-frame WebP overlay JSON the app's Wind/Precip layer already consumes
(elements_latest.json: north/south/west/east box, base64 WebP per frame, legend strips).

Iceland's overlays are warped vedur images; here we draw them from MET Norway's numeric
fields with fixed colour bands (so one legend serves every frame). Image rows are spaced
in Web-Mercator Y, so Leaflet's imageOverlay (linear in projected space) registers exactly.

Usage: python extract_meps_elements.py [--run YYYYMMDDTHH] [--out elements_latest.json]
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import io
import json
import math
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import extract_meps as M

BOX = dict(south=57.05, north=66.2, west=0.11, east=15.63)    # = MEPS cloud grid box
IMG_W, IMG_H = 623, 772                                        # same pixel density as before
ALPHA = 190

# Wind speed bands (m/s, lower edges) and colours: calm blue -> green -> yellow -> red -> purple.
WIND_EDGES = [0, 3, 5, 8, 11, 14, 17, 21, 25, 29]
WIND_RGB = [(70, 110, 170), (60, 150, 190), (70, 180, 140), (120, 200, 90), (220, 220, 70),
            (240, 170, 50), (235, 110, 40), (215, 50, 50), (170, 30, 110), (120, 40, 160)]
# Hourly precipitation bands (mm/h, lower edges); below 0.1 = transparent.
PRECIP_EDGES = [0.1, 0.3, 0.5, 1, 2, 5, 10]
PRECIP_RGB = [(170, 210, 245), (120, 180, 240), (70, 140, 230), (40, 95, 205),
              (60, 50, 180), (140, 40, 170), (210, 40, 120)]


def pixel_latlon():
    def ym(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    yN, yS = ym(BOX['north']), ym(BOX['south'])
    rows = [(yN - (r + 0.5) / IMG_H * (yN - yS)) for r in range(IMG_H)]
    lats = np.degrees(np.arctan(np.sinh(np.array(rows))))
    lons = BOX['west'] + (np.arange(IMG_W) + 0.5) / IMG_W * (BOX['east'] - BOX['west'])
    return np.meshgrid(lats, lons, indexing='ij')


def band_index(v, edges):
    return np.digitize(v, edges) - 1          # -1 = below first edge


def colourise(v, edges, rgb, transparent_below=True):
    idx = band_index(v, edges)
    out = np.zeros(v.shape + (4,), np.uint8)
    pal = np.array(rgb, np.uint8)
    ok = idx >= 0 if transparent_below else np.ones(v.shape, bool)
    idx = np.clip(idx, 0, len(rgb) - 1)
    out[..., :3] = pal[idx]
    out[..., 3] = np.where(ok & ~np.isnan(v), ALPHA, 0)
    return out


def draw_arrows(img, spd, dirn, step=40):
    d = ImageDraw.Draw(img)
    for r in range(step // 2, IMG_H, step):
        for c in range(step // 2, IMG_W, step):
            s, frm = spd[r, c], dirn[r, c]
            if np.isnan(s) or np.isnan(frm) or s < 1:
                continue
            to = math.radians(frm + 180.0)        # meteorological FROM -> arrow points TO
            L = 9 + min(s, 25) * 0.35
            dx, dy = math.sin(to) * L, -math.cos(to) * L
            x0, y0, x1, y1 = c - dx / 2, r - dy / 2, c + dx / 2, r + dy / 2
            d.line([(x0, y0), (x1, y1)], fill=(20, 20, 30, 230), width=2)
            for a in (150, -150):
                ha = to + math.radians(a)
                d.line([(x1, y1), (x1 + math.sin(ha) * 4, y1 - math.cos(ha) * 4)], fill=(20, 20, 30, 230), width=2)


def webp_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='WEBP', lossless=True, method=4)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def legend_b64(edges, rgb, fmt):
    w, cell, h = 34, 34, 14
    img = Image.new('RGBA', (cell * len(rgb), h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i, col in enumerate(rgb):
        d.rectangle([i * cell, 0, (i + 1) * cell - 1, h - 1], fill=col + (255,))
        d.text((i * cell + 2, h + 1), fmt(edges[i]), fill=(25, 25, 25, 255), font=font)   # dark: the app shows legends on a light strip
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def build(run):
    ds = M.open_run(run)
    box = (BOX['south'], BOX['north'], BOX['west'], BOX['east'])
    rect = M.sub_rect(ds, box)
    if not M.run_complete(ds, rect):
        ds.close()
        return None
    y0, y1, x0, x1 = rect
    la = np.asarray(ds.variables['latitude'][y0:y1, x0:x1])
    lo = np.asarray(ds.variables['longitude'][y0:y1, x0:x1])
    plat, plon = pixel_latlon()
    tree = cKDTree(M.unit(la, lo).reshape(-1, 3))
    dist, idx = tree.query(M.unit(plat, plon).reshape(-1, 3))
    far = (dist * 6371.0 > 3.0).reshape(IMG_H, IMG_W)
    iy, ix = np.unravel_index(idx, la.shape)

    def read(var):
        t = time.time()
        a = M._retry(lambda: ds.variables[var][:, 0, y0:y1, x0:x1], 'read ' + var)
        a = np.ma.filled(np.ma.asarray(a, dtype='float32'), np.nan)
        g = a[:, iy, ix].reshape(a.shape[0], IMG_H, IMG_W)
        g[:, far] = np.nan
        print('  %s %s %.1fs' % (var, a.shape, time.time() - t))
        return g
    spd, dirn, acc = read('wind_speed'), read('wind_direction'), read('precipitation_amount_acc')
    tsec = np.asarray(ds.variables['time'][:], dtype='float64')
    ds.close()

    rt = dt.datetime.strptime(run, '%Y%m%dT%H').replace(tzinfo=dt.timezone.utc)
    wind, precip, times, frames = [], [], [], []
    for i in range(len(tsec)):
        if np.isnan(spd[i]).all():
            continue
        vt = dt.datetime.fromtimestamp(tsec[i], dt.timezone.utc)
        times.append(vt.strftime('%Y-%m-%dT%H:%MZ'))
        frames.append(int(round((vt - rt).total_seconds() / 3600)))
        wimg = Image.fromarray(colourise(spd[i], WIND_EDGES, WIND_RGB, transparent_below=False), 'RGBA')
        draw_arrows(wimg, spd[i], dirn[i])
        wind.append(webp_b64(wimg))
        if i == 0 or np.isnan(acc[i - 1]).all():
            precip.append(None)                     # hourly amount needs the previous step
        else:
            hourly = np.clip(acc[i] - acc[i - 1], 0, None)
            precip.append(webp_b64(Image.fromarray(colourise(hourly, PRECIP_EDGES, PRECIP_RGB), 'RGBA')))
    n = len(times)
    wl = legend_b64(WIND_EDGES, WIND_RGB, lambda v: '%d' % v)
    pl = legend_b64(PRECIP_EDGES, PRECIP_RGB, lambda v: ('%g' % v))
    return {
        'source': 'metno_meps_2p5km_elements', 'region': 'nnorway',
        'run': rt.strftime('%y%m%d_%H%M'),
        'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'north': BOX['north'], 'south': BOX['south'], 'west': BOX['west'], 'east': BOX['east'],
        'img_w': IMG_W, 'img_h': IMG_H, 'fmt': 'webp', 'mercator_rows': True,
        'frames': frames, 'times': times, 'wind': wind, 'precip': precip,
        # Fixed colour bands: one legend strip (frame 0); the app falls back to the nearest
        # frame that has a strip, so the other entries can stay null.
        'wind_legends': [wl] + [None] * (n - 1), 'precip_legends': [pl] + [None] * (n - 1),
        'credit': 'MET Norway (MEPS), NLOD 2.0 / CC BY 4.0',
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run')
    ap.add_argument('--out', default='elements_latest.json')
    a = ap.parse_args(argv)
    cands = [a.run] if a.run else list(reversed(M.list_runs()))[:3]
    res = None
    for run in cands:
        print('run=%s' % run)
        res = build(run)
        if res:
            break
        print('  run %s incomplete, trying previous' % run)
    if not res:
        raise SystemExit('no complete MEPS run found')
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, separators=(',', ':'))
    print('wrote %s: run=%s %d frames' % (a.out, res['run'], len(res['times'])))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
