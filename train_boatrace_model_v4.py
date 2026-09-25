# -*- coding: utf-8 -*-
"""
BOAT RACE v4.0.1 calibrated shadow prediction model trainer

Purpose
-------
Train portable boat-level logistic models from the rolling SQLite history built by
build_boatrace_history_v3_3.py.  The generated coefficients can be scored directly
inside Google Apps Script, so no always-on prediction server is required.

Leakage control
---------------
For every race date, features are generated only from results dated BEFORE that day.
Rows from the current day are added to the history state only after every race on that
calendar day has been featurized.  This intentionally matches the production history
CSV, which is built through the previous day.

Models
------
BASE    : history + venue + boat/course proxy.  Usable as soon as the race list exists.
WEATHER : BASE + current wind + venue/course/wind historical condition stats.
          Intended for the live stage after wind data is available.

The v4.0.1 model does not use motor, national/local official rating fields, or exhibition
metrics because the 5-year K-data cache does not contain leakage-safe historical values
for those fields.  They can be added later after the live shadow log has accumulated.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from sklearn.linear_model import SGDClassifier

VERSION = "4.0.1-shadow-softmax-temp"

VENUE_CODES = [f"{i:02d}" for i in range(1, 25)]
WIND_DIRS = ["無風", "北東", "南東", "南西", "北西", "北", "東", "南", "西"]
WIND_BANDS = ["0m", "1-2m", "3-4m", "5m以上"]

COMMON_NUMERIC_FEATURES = [
    "r30_logstarts", "r30_win",
    "r90_logstarts", "r90_win",
    "r365_logstarts", "r365_win",
    "r5_logstarts", "r5_win", "r5_top2", "r5_top3", "r5_avg_st", "r5_std_st",
    "rc5_logstarts", "rc5_win", "rc5_top3", "rc5_avg_st",
    "rv5_logstarts", "rv5_win", "rv5_top3",
    "vc5_win",
]
COMMON_FEATURES = (
    [f"boat_{i}" for i in range(1, 7)]
    + [f"venue_{v}" for v in VENUE_CODES]
    + COMMON_NUMERIC_FEATURES
)
WEATHER_EXTRA_FEATURES = (
    [f"wind_{w}" for w in WIND_DIRS]
    + [f"windband_{b}" for b in WIND_BANDS]
    + ["wind_speed_scaled", "cond_logstarts", "cond_win", "cond_top2", "cond_top3", "cond_avg_st"]
)
BASE_FEATURES = COMMON_FEATURES
WEATHER_FEATURES = COMMON_FEATURES + WEATHER_EXTRA_FEATURES


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="history_state/boatrace_history.sqlite")
    ap.add_argument("--out", default="model_v4")
    ap.add_argument("--holdout-days", type=int, default=180)
    ap.add_argument("--burnin-days", type=int, default=365)
    ap.add_argument("--batch-size", type=int, default=6000)
    ap.add_argument("--upload-url", default=os.environ.get("HISTORY_ENDPOINT_URL", ""))
    ap.add_argument("--upload-token", default=os.environ.get("HISTORY_TOKEN", ""))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def softmax_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Race-level softmax over six boat logits. Common score shifts cancel out."""
    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        return x
    t = float(temperature) if temperature and math.isfinite(float(temperature)) else 1.0
    t = max(0.10, min(t, 50.0))
    x = x / t
    x = x - np.max(x)
    e = np.exp(np.clip(x, -80.0, 0.0))
    s = float(e.sum())
    if not math.isfinite(s) or s <= 0:
        return np.ones(len(x), dtype=float) / len(x)
    return e / s


def fit_temperature(samples: List[Tuple[np.ndarray, int]]) -> float:
    """Fit one scalar temperature on an earlier calibration segment by race log loss."""
    if not samples:
        return 1.0

    def loss(t: float) -> float:
        total = 0.0
        n = 0
        for scores, winner_idx in samples:
            p = softmax_scores(scores, t)
            wp = max(float(p[winner_idx]), 1e-12)
            total += -math.log(wp)
            n += 1
        return total / max(n, 1)

    # Broad deterministic search; no extra scipy dependency is needed.
    grid = np.exp(np.linspace(math.log(0.25), math.log(20.0), 241))
    vals = np.asarray([loss(float(t)) for t in grid])
    best_i = int(np.argmin(vals))
    best = float(grid[best_i])

    # Local refinement around the best log-temperature.
    lo = max(0.10, best / 1.35)
    hi = min(50.0, best * 1.35)
    fine = np.exp(np.linspace(math.log(lo), math.log(hi), 161))
    fine_vals = np.asarray([loss(float(t)) for t in fine])
    return float(fine[int(np.argmin(fine_vals))])


def scale_starts(n: int) -> float:
    if not n or n <= 0:
        return 0.0
    return float(min(math.log1p(n) / math.log1p(1000.0), 1.5))


def scale_avg_st(v: Optional[float]) -> float:
    # 0.18 is a neutral missing value; divide by .30 to keep a roughly 0-1 scale.
    if v is None or not math.isfinite(v):
        v = 0.18
    return float(max(0.0, min(v / 0.30, 2.0)))


def scale_std_st(v: Optional[float]) -> float:
    if v is None or not math.isfinite(v):
        v = 0.05
    return float(max(0.0, min(v / 0.20, 2.0)))


@dataclass
class Snapshot:
    starts: int = 0
    wins: int = 0
    top2: int = 0
    top3: int = 0
    sum_st: float = 0.0
    sum_st2: float = 0.0
    st_n: int = 0

    @property
    def win(self) -> float:
        return self.wins / self.starts if self.starts else 0.0

    @property
    def top2_rate(self) -> float:
        return self.top2 / self.starts if self.starts else 0.0

    @property
    def top3_rate(self) -> float:
        return self.top3 / self.starts if self.starts else 0.0

    @property
    def avg_st(self) -> Optional[float]:
        return self.sum_st / self.st_n if self.st_n else None

    @property
    def std_st(self) -> Optional[float]:
        if not self.st_n:
            return None
        m = self.sum_st / self.st_n
        return math.sqrt(max(self.sum_st2 / self.st_n - m * m, 0.0))


class RunningStats:
    __slots__ = ("starts", "wins", "top2", "top3", "sum_st", "sum_st2", "st_n")

    def __init__(self):
        self.starts = self.wins = self.top2 = self.top3 = 0
        self.sum_st = self.sum_st2 = 0.0
        self.st_n = 0

    def add(self, rank: Optional[int], st: Optional[float]):
        self.starts += 1
        if rank == 1:
            self.wins += 1
        if rank is not None and 1 <= rank <= 2:
            self.top2 += 1
        if rank is not None and 1 <= rank <= 3:
            self.top3 += 1
        if st is not None and math.isfinite(st):
            self.sum_st += st
            self.sum_st2 += st * st
            self.st_n += 1

    def snapshot(self) -> Snapshot:
        return Snapshot(self.starts, self.wins, self.top2, self.top3,
                        self.sum_st, self.sum_st2, self.st_n)


class WindowStats:
    __slots__ = ("days", "q", "starts", "wins", "top2", "top3", "sum_st", "sum_st2", "st_n")

    def __init__(self, days: int):
        self.days = days
        self.q = deque()
        self.starts = self.wins = self.top2 = self.top3 = 0
        self.sum_st = self.sum_st2 = 0.0
        self.st_n = 0

    def _remove(self, item):
        _, rank, st = item
        self.starts -= 1
        if rank == 1:
            self.wins -= 1
        if rank is not None and 1 <= rank <= 2:
            self.top2 -= 1
        if rank is not None and 1 <= rank <= 3:
            self.top3 -= 1
        if st is not None and math.isfinite(st):
            self.sum_st -= st
            self.sum_st2 -= st * st
            self.st_n -= 1

    def purge(self, current_ord: int):
        cutoff = current_ord - self.days
        while self.q and self.q[0][0] < cutoff:
            self._remove(self.q.popleft())

    def add(self, day_ord: int, rank: Optional[int], st: Optional[float]):
        item = (day_ord, rank, st)
        self.q.append(item)
        self.starts += 1
        if rank == 1:
            self.wins += 1
        if rank is not None and 1 <= rank <= 2:
            self.top2 += 1
        if rank is not None and 1 <= rank <= 3:
            self.top3 += 1
        if st is not None and math.isfinite(st):
            self.sum_st += st
            self.sum_st2 += st * st
            self.st_n += 1

    def snapshot(self, current_ord: int) -> Snapshot:
        self.purge(current_ord)
        return Snapshot(self.starts, self.wins, self.top2, self.top3,
                        self.sum_st, self.sum_st2, self.st_n)


class RacerWindows:
    __slots__ = ("w30", "w90", "w365")

    def __init__(self):
        self.w30 = WindowStats(30)
        self.w90 = WindowStats(90)
        self.w365 = WindowStats(365)

    def snapshots(self, day_ord: int):
        return (
            self.w30.snapshot(day_ord),
            self.w90.snapshot(day_ord),
            self.w365.snapshot(day_ord),
        )

    def add(self, day_ord: int, rank: Optional[int], st: Optional[float]):
        self.w30.add(day_ord, rank, st)
        self.w90.add(day_ord, rank, st)
        self.w365.add(day_ord, rank, st)


class HistoryState:
    def __init__(self):
        self.racer_windows: Dict[str, RacerWindows] = {}
        self.racer_all: Dict[str, RunningStats] = defaultdict(RunningStats)
        self.racer_course: Dict[Tuple[str, int], RunningStats] = defaultdict(RunningStats)
        self.racer_venue: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
        self.venue_course: Dict[Tuple[str, int], RunningStats] = defaultdict(RunningStats)
        self.venue_cond: Dict[Tuple[str, int, str, str], RunningStats] = defaultdict(RunningStats)

    def windows_for(self, reg: str, day_ord: int):
        rw = self.racer_windows.get(reg)
        if rw is None:
            return Snapshot(), Snapshot(), Snapshot()
        return rw.snapshots(day_ord)

    def add_day_row(self, row, day_ord: int):
        rank = row[4]
        st = row[8]
        reg = row[6]
        venue = row[1]
        course = row[7]
        wind_dir = row[9] or ""
        wind_band = row[11] or ""

        rw = self.racer_windows.get(reg)
        if rw is None:
            rw = self.racer_windows[reg] = RacerWindows()
        rw.add(day_ord, rank, st)
        self.racer_all[reg].add(rank, st)
        if course and 1 <= course <= 6:
            self.racer_course[(reg, course)].add(rank, st)
            self.venue_course[(venue, course)].add(rank, st)
            if wind_dir and wind_band:
                self.venue_cond[(venue, course, wind_dir, wind_band)].add(rank, st)
        self.racer_venue[(reg, venue)].add(rank, st)


def stat_features(s: Snapshot):
    return {
        "logstarts": scale_starts(s.starts),
        "win": s.win,
        "top2": s.top2_rate,
        "top3": s.top3_rate,
        "avg_st": scale_avg_st(s.avg_st),
        "std_st": scale_std_st(s.std_st),
    }


def feature_dict(row, state: HistoryState, day_ord: int, weather: bool) -> Dict[str, float]:
    # Row tuple: date, venue, race_no, boat, rank, rank_text, reg, course, st, wind_dir, wind_speed, wind_band
    venue = row[1]
    boat = int(row[3])
    reg = row[6]
    actual_course = int(row[7]) if row[7] is not None and 1 <= int(row[7]) <= 6 else boat

    d = {f: 0.0 for f in (WEATHER_FEATURES if weather else BASE_FEATURES)}
    d[f"boat_{boat}"] = 1.0
    if f"venue_{venue}" in d:
        d[f"venue_{venue}"] = 1.0

    r30, r90, r365 = state.windows_for(reg, day_ord)
    r5 = state.racer_all.get(reg, RunningStats()).snapshot()
    # BASE uses boat as course proxy, matching live production before exhibition.
    rc = state.racer_course.get((reg, boat), RunningStats()).snapshot()
    rv = state.racer_venue.get((reg, venue), RunningStats()).snapshot()
    vc = state.venue_course.get((venue, boat), RunningStats()).snapshot()

    a30, a90, a365, a5, ac, av, avc = map(stat_features, [r30, r90, r365, r5, rc, rv, vc])
    d.update({
        "r30_logstarts": a30["logstarts"], "r30_win": a30["win"],
        "r90_logstarts": a90["logstarts"], "r90_win": a90["win"],
        "r365_logstarts": a365["logstarts"], "r365_win": a365["win"],
        "r5_logstarts": a5["logstarts"], "r5_win": a5["win"], "r5_top2": a5["top2"],
        "r5_top3": a5["top3"], "r5_avg_st": a5["avg_st"], "r5_std_st": a5["std_st"],
        "rc5_logstarts": ac["logstarts"], "rc5_win": ac["win"], "rc5_top3": ac["top3"], "rc5_avg_st": ac["avg_st"],
        "rv5_logstarts": av["logstarts"], "rv5_win": av["win"], "rv5_top3": av["top3"],
        "vc5_win": avc["win"],
    })

    if weather:
        wind_dir = row[9] or ""
        wind_speed = row[10]
        wind_band = row[11] or ""
        if wind_dir in WIND_DIRS:
            d[f"wind_{wind_dir}"] = 1.0
        if wind_band in WIND_BANDS:
            d[f"windband_{wind_band}"] = 1.0
        if wind_speed is not None:
            d["wind_speed_scaled"] = max(0.0, min(float(wind_speed) / 10.0, 2.0))
        cond = state.venue_cond.get((venue, actual_course, wind_dir, wind_band), RunningStats()).snapshot()
        c = stat_features(cond)
        d.update({
            "cond_logstarts": c["logstarts"], "cond_win": c["win"], "cond_top2": c["top2"],
            "cond_top3": c["top3"], "cond_avg_st": c["avg_st"],
        })
    return d


def vectorize(d: Dict[str, float], features: List[str]) -> np.ndarray:
    return np.asarray([float(d.get(f, 0.0)) for f in features], dtype=np.float32)


def iter_rows_by_date(conn: sqlite3.Connection):
    sql = """
    SELECT date, venue_code, race_no, boat, rank_num, rank_text, reg, course, st,
           COALESCE(wind_dir,''), wind_speed, COALESCE(wind_band,'')
    FROM results
    ORDER BY date, venue_code, race_no, boat
    """
    cur = conn.execute(sql)
    current = None
    buf = []
    for row in cur:
        if current is None:
            current = row[0]
        if row[0] != current:
            yield current, buf
            current, buf = row[0], []
        buf.append(row)
    if buf:
        yield current, buf


def valid_races(day_rows):
    races = defaultdict(list)
    for r in day_rows:
        races[(r[1], r[2])].append(r)
    for key in sorted(races):
        rs = sorted(races[key], key=lambda x: x[3])
        if len(rs) != 6:
            continue
        if [r[3] for r in rs] != [1, 2, 3, 4, 5, 6]:
            continue
        winners = [r for r in rs if r[4] == 1]
        if len(winners) != 1:
            continue
        yield key, rs


def weather_race_usable(rs) -> bool:
    for r in rs:
        if not r[9] or not r[11] or r[10] is None:
            return False
        if r[7] is None or not (1 <= int(r[7]) <= 6):
            return False
    return True


class OnlineModel:
    def __init__(self, name: str, features: List[str], batch_size: int):
        self.name = name
        self.features = features
        self.batch_size = batch_size
        self.clf = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=2e-5,
            learning_rate="optimal", average=True, random_state=42,
        )
        self.xbuf: List[np.ndarray] = []
        self.ybuf: List[int] = []
        self.fitted = False
        self.train_rows = 0
        self.train_races = 0

    def add_race(self, xs: List[np.ndarray], ys: List[int]):
        self.xbuf.extend(xs)
        self.ybuf.extend(ys)
        self.train_races += 1
        if len(self.xbuf) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.xbuf:
            return
        X = np.vstack(self.xbuf)
        y = np.asarray(self.ybuf, dtype=np.int8)
        if not self.fitted:
            self.clf.partial_fit(X, y, classes=np.array([0, 1], dtype=np.int8))
            self.fitted = True
        else:
            self.clf.partial_fit(X, y)
        self.train_rows += len(y)
        self.xbuf.clear()
        self.ybuf.clear()

    def race_scores(self, xs: List[np.ndarray]) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(xs), dtype=float)
        X = np.vstack(xs)
        return np.asarray(self.clf.decision_function(X), dtype=float).reshape(-1)

    def race_probs(self, xs: List[np.ndarray], temperature: float = 1.0) -> np.ndarray:
        return softmax_scores(self.race_scores(xs), temperature)


class MetricAccumulator:
    def __init__(self):
        self.races = 0
        self.top1 = 0
        self.brier = 0.0
        self.logloss = 0.0
        self.winner_p = 0.0

    def add(self, p: np.ndarray, winner_idx: int):
        y = np.zeros(6, dtype=float)
        y[winner_idx] = 1.0
        self.races += 1
        self.top1 += int(int(np.argmax(p)) == winner_idx)
        self.brier += float(np.sum((p - y) ** 2))
        wp = max(float(p[winner_idx]), 1e-12)
        self.logloss += -math.log(wp)
        self.winner_p += wp

    def summary(self):
        if not self.races:
            return {"races": 0, "top1": None, "brier": None, "logloss": None, "winner_p": None}
        return {
            "races": self.races,
            "top1": self.top1 / self.races,
            "brier": self.brier / self.races,
            "logloss": self.logloss / self.races,
            "winner_p": self.winner_p / self.races,
        }


def train_models(db_path: Path, holdout_days: int, burnin_days: int, batch_size: int):
    conn = sqlite3.connect(str(db_path))
    minmax = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM results").fetchone()
    if not minmax or not minmax[0] or not minmax[1]:
        raise RuntimeError("results table is empty")
    start = parse_iso_date(minmax[0])
    end = parse_iso_date(minmax[1])
    if (end - start).days < burnin_days + 60:
        raise RuntimeError(f"Not enough history: {start}..{end}")

    holdout_days = min(max(60, holdout_days), max(60, (end - start).days // 3))
    test_start = end - timedelta(days=holdout_days - 1)
    calibration_days = max(30, holdout_days // 2)
    validation_start = min(end, test_start + timedelta(days=calibration_days))
    calibration_end = validation_start - timedelta(days=1)
    burnin_end = start + timedelta(days=burnin_days)

    print(f"v4 trainer {VERSION}")
    print(f"history={start}..{end} rows={minmax[2]:,}")
    print(f"burn-in through {burnin_end - timedelta(days=1)}")
    print(f"temperature calibration={test_start}..{calibration_end}")
    print(f"validation={validation_start}..{end}")
    print("holdout is day-prequential: score all races using data/model through the previous day, then learn that day")

    state = HistoryState()
    base = OnlineModel("BASE", BASE_FEATURES, batch_size)
    weather = OnlineModel("WEATHER", WEATHER_FEATURES, batch_size)
    base_metric = MetricAccumulator()
    weather_metric = MetricAccumulator()
    base_cal_samples: List[Tuple[np.ndarray, int]] = []
    weather_cal_samples: List[Tuple[np.ndarray, int]] = []
    base_temp: Optional[float] = None
    weather_temp: Optional[float] = None
    seen_days = 0

    for day_s, day_rows in iter_rows_by_date(conn):
        day = parse_iso_date(day_s)
        day_ord = day.toordinal()
        seen_days += 1
        can_train = day >= burnin_end
        is_cal = test_start <= day < validation_start
        is_validation = day >= validation_start
        in_holdout = day >= test_start

        # At holdout start and then each day, pending prior-day training must be committed before scoring today.
        if in_holdout:
            base.flush(); weather.flush()

        # Freeze temperatures once, before the later validation segment starts.
        if is_validation and base_temp is None:
            base_temp = fit_temperature(base_cal_samples)
            weather_temp = fit_temperature(weather_cal_samples)
            print(f"  calibrated BASE temperature={base_temp:.6f} from {len(base_cal_samples):,} races")
            print(f"  calibrated WEATHER temperature={weather_temp:.6f} from {len(weather_cal_samples):,} races")

        day_base_train = []
        day_weather_train = []

        for _, rs in valid_races(day_rows):
            winner_idx = next(i for i, r in enumerate(rs) if r[4] == 1)
            base_x = [vectorize(feature_dict(r, state, day_ord, False), BASE_FEATURES) for r in rs]
            y = [1 if i == winner_idx else 0 for i in range(6)]

            if can_train:
                if is_cal:
                    base_cal_samples.append((base.race_scores(base_x).copy(), winner_idx))
                elif is_validation:
                    base_metric.add(base.race_probs(base_x, base_temp or 1.0), winner_idx)
                day_base_train.append((base_x, y))

            if weather_race_usable(rs):
                wx = [vectorize(feature_dict(r, state, day_ord, True), WEATHER_FEATURES) for r in rs]
                if can_train:
                    if is_cal:
                        weather_cal_samples.append((weather.race_scores(wx).copy(), winner_idx))
                    elif is_validation:
                        weather_metric.add(weather.race_probs(wx, weather_temp or 1.0), winner_idx)
                    day_weather_train.append((wx, y))

        # Learn today's labels only after every race on this date has been scored.
        if can_train:
            for xs, y in day_base_train:
                base.add_race(xs, y)
            for xs, y in day_weather_train:
                weather.add_race(xs, y)
            # During holdout, flush once per day so tomorrow uses all information through today.
            if in_holdout:
                base.flush(); weather.flush()

        # IMPORTANT: history features are also updated only after every race on this calendar date was featurized.
        for r in day_rows:
            if r[4] is not None:
                state.add_day_row(r, day_ord)

        if seen_days % 180 == 0:
            print(f"  processed {seen_days} days through {day_s}")

    base.flush()
    weather.flush()
    if base_temp is None:
        base_temp = fit_temperature(base_cal_samples)
    if weather_temp is None:
        weather_temp = fit_temperature(weather_cal_samples)
    conn.close()

    if not base.fitted or not weather.fitted:
        raise RuntimeError("Model training did not receive enough usable races")
    return {
        "start": start, "end": end, "test_start": test_start,
        "calibration_end": calibration_end, "validation_start": validation_start,
        "base": base, "weather": weather,
        "base_temperature": float(base_temp), "weather_temperature": float(weather_temp),
        "base_calibration_races": len(base_cal_samples),
        "weather_calibration_races": len(weather_cal_samples),
        "base_metric": base_metric.summary(), "weather_metric": weather_metric.summary(),
    }


def write_coefficients(out_dir: Path, trained):
    path = out_dir / "BR_model_v4_coefficients.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "feature", "coefficient"])
        for name in ("BASE", "WEATHER"):
            model: OnlineModel = trained[name.lower()]
            w.writerow([name, "__INTERCEPT__", repr(float(model.clf.intercept_[0]))])
            coef = model.clf.coef_[0]
            for feature, value in zip(model.features, coef):
                w.writerow([name, feature, repr(float(value))])
    return path


def write_meta(out_dir: Path, trained):
    b = trained["base_metric"]
    w = trained["weather_metric"]
    rows = [
        ["モデルバージョン", VERSION],
        ["学習開始履歴日", trained["start"].isoformat()],
        ["学習終了履歴日", trained["end"].isoformat()],
        ["Temperature校正開始日", trained["test_start"].isoformat()],
        ["Temperature校正終了日", trained["calibration_end"].isoformat()],
        ["検証開始日", trained["validation_start"].isoformat()],
        ["検証終了日", trained["end"].isoformat()],
        ["BASE_Temperature", round(trained["base_temperature"], 8)],
        ["WEATHER_Temperature", round(trained["weather_temperature"], 8)],
        ["BASE_校正レース数", trained["base_calibration_races"]],
        ["WEATHER_校正レース数", trained["weather_calibration_races"]],
        ["BASE_学習レース数", trained["base"].train_races],
        ["WEATHER_学習レース数", trained["weather"].train_races],
        ["BASE_検証レース数", b["races"]],
        ["BASE_本命1着率", round((b["top1"] or 0) * 100, 3) if b["races"] else ""],
        ["BASE_Brier", round(b["brier"], 6) if b["races"] else ""],
        ["BASE_LogLoss", round(b["logloss"], 6) if b["races"] else ""],
        ["BASE_実勝艇平均予測確率", round((b["winner_p"] or 0) * 100, 3) if b["races"] else ""],
        ["WEATHER_検証レース数", w["races"]],
        ["WEATHER_本命1着率", round((w["top1"] or 0) * 100, 3) if w["races"] else ""],
        ["WEATHER_Brier", round(w["brier"], 6) if w["races"] else ""],
        ["WEATHER_LogLoss", round(w["logloss"], 6) if w["races"] else ""],
        ["WEATHER_実勝艇平均予測確率", round((w["winner_p"] or 0) * 100, 3) if w["races"] else ""],
        ["確率方式", "6艇logitのレース単位Softmax + モデル別Temperature校正"],
        ["検証方式", "前半holdoutでTemperatureを校正し、後半holdoutで検証。各レース予測後にのみオンライン学習。"],
        ["特徴量方針", "当日より前の結果だけで履歴特徴量を再構築。BASEは艇番をコース近似、WEATHERは風条件を追加。"],
        ["用途", "v4.0.1影運用。NotebookLMは説明担当とし、確率計算はこのモデルを優先。"],
        ["作成日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    path = out_dir / "BR_model_v4_meta.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        cw = csv.writer(f)
        cw.writerow(["項目", "値"])
        cw.writerows(rows)
    summary = out_dir / "V4_MODEL_SUMMARY.txt"
    summary.write_text("\n".join(f"{k}: {v}" for k, v in rows), encoding="utf-8")
    return path, summary


def upload_model_csvs(out_dir: Path, url: str, token: str):
    if not url or not token:
        raise RuntimeError("HISTORY_ENDPOINT_URL / HISTORY_TOKEN が未設定です")
    names = ["BR_model_v4_coefficients.csv", "BR_model_v4_meta.csv"]
    sess = requests.Session()
    for i, name in enumerate(names):
        raw = (out_dir / name).read_bytes()
        payload = {
            "action": "upload_model_csv",
            "token": token,
            "filename": name,
            "model_version": VERSION,
            "final": i == len(names) - 1,
            "data_gzip_base64": base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii"),
        }
        for attempt in range(4):
            try:
                r = sess.post(url, json=payload, timeout=180)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                if not data.get("ok"):
                    raise RuntimeError("receiver error: " + str(data))
                print(f"[UPLOAD] {name}: OK rows={data.get('rows','-')}")
                break
            except Exception as e:
                if attempt >= 3:
                    raise
                print(f"[UPLOAD] {name}: retry {attempt+1}/3: {e}")
                time.sleep(3 * (attempt + 1))


def make_synthetic_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute("""
    CREATE TABLE results(
      date TEXT, venue_code TEXT, venue_name TEXT, race_no INTEGER,
      rank_text TEXT, rank_num INTEGER, boat INTEGER, reg TEXT, racer_name TEXT,
      course INTEGER, st REAL, wind_dir TEXT, wind_speed INTEGER, wind_band TEXT
    )""")
    start = date(2022, 1, 1)
    regs = [f"{4000+i:04d}" for i in range(24)]
    for dd in range(650):
        dt = start + timedelta(days=dd)
        for race_no in range(1, 5):
            winner = ((dd + race_no) % 6) + 1
            venue = f"{((dd // 10) % 24) + 1:02d}"
            for boat in range(1, 7):
                rank = 1 if boat == winner else (2 + ((boat + dd) % 5))
                reg = regs[(race_no * 6 + boat + dd) % len(regs)]
                conn.execute("INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    dt.isoformat(), venue, "X", race_no, str(rank), rank, boat, reg, "R",
                    boat, 0.10 + boat * 0.01, "東", 3, "3-4m"
                ))
    conn.commit(); conn.close()


def self_test(tmp: Path):
    db = tmp / "synthetic.sqlite"
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    make_synthetic_db(db)
    trained = train_models(db, holdout_days=90, burnin_days=365, batch_size=1000)
    write_coefficients(out, trained)
    write_meta(out, trained)
    assert (out / "BR_model_v4_coefficients.csv").exists()
    assert trained["base_metric"]["races"] > 0
    print("SELF-TEST OK")


def main():
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(out_dir)
        return 0
    db = Path(args.db).resolve()
    if not db.exists():
        print(f"ERROR: SQLite cache not found: {db}", file=sys.stderr)
        return 2
    trained = train_models(db, args.holdout_days, args.burnin_days, args.batch_size)
    write_coefficients(out_dir, trained)
    write_meta(out_dir, trained)
    print((out_dir / "V4_MODEL_SUMMARY.txt").read_text(encoding="utf-8"))
    if not args.no_upload and args.upload_url and args.upload_token:
        upload_model_csvs(out_dir, args.upload_url, args.upload_token)
    elif not args.no_upload:
        print("[UPLOAD] URL/token missing; model upload skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
