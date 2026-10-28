#!/usr/bin/env python3
"""
extract_meps.py - MET Norway MEPS 2.5 km (HARMONIE-AROME / MetCoOp) low/mid/high
cloud cover -> compact JSON grid for the Sunsetglow app (Norway regions).

Unlike the Iceland extractor (which scrapes vedur PNGs), MEPS is published as
numeric NetCDF on thredds.met.no, so we read the real model fields over OPeNDAP:
only the sub-rectangle covering the region box is downloaded (a few tens of MB
per run, one request per layer). Licence: NLOD 2.0 / CC BY 4.0, credit MET Norway.

Output schema = harmonie_latest.json (row 0 = north, col 0 = west, UTC times,
run id 'YYMMDD_HHMM') plus `div`: stored integer / div = cloud fraction 0..1.
Iceland uses oktas (div 8); MEPS is continuous, stored in 5 % steps (div 20).

Usage:  python extract_meps.py [--run YYYYMMDDTHH] [--box lat0,lat1,lon0,lon1]
                               [--dlat 0.03] [--dlon 0.075] [--out FILE]
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import gzip
import json
import re
import sys
import time

import numpy as np
import requests
import netCDF4
from scipy.spatial import cKDTree

CATALOG = 'https://thredds.met.no/thredds/catalog/mepslatest/catalog.xml'
DODS = 'https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_%sZ.ncml'
LAYERS = {'low': 'low_type_cloud_area_fraction',
          'mid': 'medium_type_cloud_area_fraction',
          'high': 'high_type_cloud_area_fraction'}
DIV = 20                 # 5 % steps
NODATA = 255
# Visibility (fog layer): metres in VIS_STEP steps up to the app's 5000 m slider maximum;
# VIS_CLEAR = clearer than that (no fog at any threshold). One byte per cell.
VIS_VAR = 'visibility_in_air'
VIS_STEP = 50
VIS_CLEAR = 101
UA = {'User-Agent': 'sunsetglow-data/1.0 github.com/avengingab1'}

# South Norway: Husby/Nesna (N) .. Aalborg (S), Le Havre longitude (W) .. Falun (E): the whole
# south of Norway plus the North Sea up-sun, west Sweden and north Denmark.
DEFAULT_BOX = (57.05, 66.2, 0.11, 15.63)


def _retry(fn, what, tries=4, wait=15):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                       # network / DAP hiccup
            if k == tries - 1:
                raise
            print('  %s failed (%s), retry in %ds' % (what, str(e)[:120], wait))
            time.sleep(wait)


class SourceUnreachable(Exception):
    """thredds.met.no did not answer, or answered only with server errors (HTTP 5xx, e.g.
    'Storage ... is not active' / 'Cannot clear an unstored lazy reference', 25-26 Oct 2026):
    transient outage, skip this cycle, don't fail."""


STALE_HOURS = 12         # published run older than this => a source outage fails loudly again


def list_runs():
    """Existing MEPS runs, oldest -> newest. Probes each 3-hourly cycle's tiny OPeNDAP .dds
    (~1 s) newest-first instead of reading the THREDDS catalog listing, which intermittently
    takes > 60 s and used to fail whole runs (22-23 Oct 2026). The catalog is only a fallback."""
    now = dt.datetime.now(dt.timezone.utc)
    found, answered = [], False
    for back in range(0, 7):                         # cycles in the last 18 h
        c = now - dt.timedelta(hours=3 * back)
        run = c.replace(hour=c.hour - c.hour % 3).strftime('%Y%m%dT%H')
        for attempt in range(2):
            try:
                r = requests.get(DODS % run + '.dds', timeout=(10, 30), headers=UA)
                if r.status_code >= 500:             # server error is not an answer about the run
                    print('  probe %s: HTTP %d %s' % (run, r.status_code, r.text[:100].replace('\n', ' ')))
                    if attempt == 0:
                        time.sleep(5)
                        continue
                    break
                answered = True
                if r.status_code == 200:
                    found.append(run)
                break
            except requests.RequestException as e:
                print('  probe %s failed (%s)%s' % (run, str(e)[:80], ', retry' if attempt == 0 else ''))
                time.sleep(5)
        if len(found) >= 3:
            break
    if found:
        return sorted(found)
    try:
        txt = _retry(lambda: requests.get(CATALOG, timeout=60, headers=UA).text, 'catalog', tries=2)
        runs = sorted(set(re.findall(r'meps_det_2_5km_(\d{8}T\d{2})Z\.ncml', txt)))
    except Exception as e:
        if not answered:
            raise SourceUnreachable(str(e)[:200])
        raise
    if not runs:          # no run at all in 18 h (probes 5xx/404, catalog empty/404: 26 Oct 2026)
        raise SourceUnreachable('no MEPS run listed (catalog %s)' % ('empty' if answered else 'unavailable'))
    return runs


def open_run(run):
    return _retry(lambda: netCDF4.Dataset(DODS % run), 'open ' + run)


def sub_rect(ds, box, pad=4):
    """Index rectangle of the LCC grid covering the lat/lon box (strided scan, then pad)."""
    lat0, lat1, lon0, lon1 = box
    s = 8
    la = np.asarray(ds.variables['latitude'][::s, ::s])
    lo = np.asarray(ds.variables['longitude'][::s, ::s])
    m = (la >= lat0 - 0.3) & (la <= lat1 + 0.3) & (lo >= lon0 - 0.8) & (lo <= lon1 + 0.8)
    ys, xs = np.where(m)
    if not len(ys):
        raise SystemExit('box outside MEPS domain')
    ny, nx = ds.variables['latitude'].shape
    y0 = max(0, ys.min() * s - s - pad); y1 = min(ny, ys.max() * s + s + pad + 1)
    x0 = max(0, xs.min() * s - s - pad); x1 = min(nx, xs.max() * s + s + pad + 1)
    return int(y0), int(y1), int(x0), int(x1)


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def run_complete(ds, rect):
    """Newest run may still be filling: last step all-masked => incomplete."""
    y0, y1, x0, x1 = rect
    last = ds.variables[LAYERS['low']][-1, 0, y0:y1:10, x0:x1:10]
    return not np.ma.getmaskarray(last).all()


def build(run, box, dlat, dlon):
    ds = open_run(run)
    rect = sub_rect(ds, box)
    y0, y1, x0, x1 = rect
    if not run_complete(ds, rect):
        ds.close()
        return None
    la = np.asarray(ds.variables['latitude'][y0:y1, x0:x1])
    lo = np.asarray(ds.variables['longitude'][y0:y1, x0:x1])
    lat0, lat1, lon0, lon1 = box
    out_lats = np.round(np.arange(lat1, lat0 - 1e-9, -dlat), 5)      # N -> S
    out_lons = np.round(np.arange(lon0, lon1 + 1e-9, dlon), 5)       # W -> E
    glat, glon = np.meshgrid(out_lats, out_lons, indexing='ij')
    tree = cKDTree(unit(la, lo).reshape(-1, 3))
    dist, idx = tree.query(unit(glat, glon).reshape(-1, 3))
    far = dist * 6371.0 > 3.0                       # > 3 km from any model cell: outside
    iy, ix = np.unravel_index(idx, la.shape)

    tsec = np.asarray(ds.variables['time'][:], dtype='float64')
    rt = dt.datetime.strptime(run, '%Y%m%dT%H').replace(tzinfo=dt.timezone.utc)
    out = {}
    for key, var in LAYERS.items():
        t = time.time()
        a = _retry(lambda: ds.variables[var][:, 0, y0:y1, x0:x1], 'read ' + var)
        a = np.ma.filled(np.ma.asarray(a, dtype='float32'), np.nan)
        g = a[:, iy, ix]                             # (T, ncell)
        q = np.where(np.isnan(g), NODATA, np.clip(np.rint(g * DIV), 0, DIV)).astype(np.uint8)
        q[:, far] = NODATA
        out[key] = q.reshape(len(tsec), len(out_lats), len(out_lons))
        print('  %s %s read+resample %.1fs' % (key, a.shape, time.time() - t))
    try:                                             # optional: never block the cloud feed
        t = time.time()
        a = _retry(lambda: ds.variables[VIS_VAR][:, 0, y0:y1, x0:x1], 'read ' + VIS_VAR)
        a = np.ma.filled(np.ma.asarray(a, dtype='float32'), np.nan)
        g = a[:, iy, ix]
        q = np.where(np.isnan(g), NODATA, np.minimum(np.rint(g / VIS_STEP), VIS_CLEAR)).astype(np.uint8)
        q[:, far] = NODATA
        out['vis'] = q.reshape(len(tsec), len(out_lats), len(out_lons))
        print('  vis %s read+resample %.1fs' % (a.shape, time.time() - t))
    except Exception as e:
        print('  WARNING: %s unavailable (%s); fog layer falls back to Open-Meteo' % (VIS_VAR, str(e)[:120]))
    ds.close()

    # Drop steps that are entirely nodata (run still filling at the tail).
    keep = [i for i in range(len(tsec)) if (out['low'][i] != NODATA).any()]
    times, frames = [], []
    for i in keep:
        vt = dt.datetime.fromtimestamp(tsec[i], dt.timezone.utc)
        times.append(vt.strftime('%Y-%m-%dT%H:%MZ'))
        frames.append(int(round((vt - rt).total_seconds() / 3600)))
    res = {
        'source': 'metno_meps_2p5km', 'region': 'nnorway',
        'scale': 'fraction_div', 'div': DIV, 'nodata': NODATA,
        'run': rt.strftime('%y%m%d_%H%M'),
        'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'lat0': float(out_lats[0]), 'lat1': float(out_lats[-1]),
        'lon0': float(out_lons[0]), 'lon1': float(out_lons[-1]),
        'nlat': len(out_lats), 'nlon': len(out_lons), 'dlat': dlat, 'dlon': dlon,
        'frames': frames, 'times': times,
        'low': [out['low'][i].tolist() for i in keep],
        'mid': [out['mid'][i].tolist() for i in keep],
        'high': [out['high'][i].tolist() for i in keep],
        'credit': 'MET Norway (MEPS), NLOD 2.0 / CC BY 4.0',
    }
    if 'vis' in out:
        # One base64 string of nlat*nlon bytes per frame (row-major, row 0 = north): ~3x smaller
        # than nested JSON arrays in the app's IndexedDB history and in memory.
        res['vis'] = [base64.b64encode(np.ascontiguousarray(out['vis'][i]).tobytes()).decode('ascii') for i in keep]
        res['vis_enc'] = 'u8-b64'
        res['vis_step'] = VIS_STEP          # metres per stored unit
        res['vis_clear'] = VIS_CLEAR        # stored value meaning "clearer than vis_clear-1 units"
    return res


def outage(published, why):
    """Source outage: exit 4 (workflow keeps the published run, warns, Worker re-dispatches),
    unless the published run is already older than STALE_HOURS - then fail so it gets noticed."""
    print(why)
    try:
        age = (dt.datetime.now(dt.timezone.utc) -
               dt.datetime.strptime(published, '%y%m%d_%H%M').replace(tzinfo=dt.timezone.utc))
    except (TypeError, ValueError):
        age = None
    if age is None or age > dt.timedelta(hours=STALE_HOURS):
        raise SystemExit('source outage and published run %s is %s old (limit %d h)' % (
            published, 'unknown' if age is None else str(age).split('.')[0], STALE_HOURS))
    print('keeping the published run %s (%s old)' % (published, str(age).split('.')[0]))
    return 4


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='YYYYMMDDTHH (default newest complete)')
    ap.add_argument('--box', default=','.join(map(str, DEFAULT_BOX)))
    ap.add_argument('--dlat', type=float, default=0.03)
    ap.add_argument('--dlon', type=float, default=0.055)   # ~2.9 km at 61 N (North Norway: 0.075 = 2.7 km at 69 N)
    ap.add_argument('--out', default='harmonie_latest.json')
    ap.add_argument('--skip-if-run', help="exit 3 without writing if newest complete run == this id ('YYMMDD_HHMM')")
    a = ap.parse_args(argv)
    box = tuple(float(x) for x in a.box.split(','))
    try:
        cands = [a.run] if a.run else list(reversed(list_runs()))[:3]
    except SourceUnreachable as e:
        return outage(a.skip_if_run, 'thredds.met.no unreachable (%s)' % e)
    res, dap_errors = None, []
    for run in cands:
        rid = dt.datetime.strptime(run, '%Y%m%dT%H').strftime('%y%m%d_%H%M')
        if a.skip_if_run and rid == a.skip_if_run:
            print('run %s already published; nothing to do' % rid)
            return 3
        print('run=%s box=%s' % (run, box))
        try:
            res = build(run, box, a.dlat, a.dlon)
        except (OSError, RuntimeError) as e:         # netCDF4 DAP server error after retries
            dap_errors.append('%s: %s' % (run, str(e)[:120]))
            print('  run %s unreadable (%s), trying previous' % (run, str(e)[:120]))
            continue
        if res:
            break
        print('  run %s still incomplete, trying previous' % run)
    if not res:
        if dap_errors and not a.run:
            return outage(a.skip_if_run, 'thredds.met.no server errors: ' + '; '.join(dap_errors))
        raise SystemExit('no complete MEPS run found')
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, separators=(',', ':'))
    raw = open(a.out, 'rb').read()
    print('wrote %s: run=%s %dx%d x %d frames, %.1f MB (gzip %.1f MB)' % (
        a.out, res['run'], res['nlat'], res['nlon'], len(res['times']),
        len(raw) / 1e6, len(gzip.compress(raw, 6)) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
