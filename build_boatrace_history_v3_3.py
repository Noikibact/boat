# -*- coding: utf-8 -*-
"""
BOAT RACE 過去5年履歴ビルダー v3.3.0
- BOAT RACE公式「競走成績」日次LZH(Kデータ)を取得
- SQLiteへ保存
- Google Apps Script v3.3 が読める6つのCSVを生成
- --rolling で「昨日までの直近5年」を自動設定
- Apps Script Webアプリへ6CSVをgzip+base64で自動送信可能

標準の本番期間:
  2021-09-20 ～ 2026-09-19
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import io
import math
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

try:
    import lhafile
except ImportError:
    lhafile = None


VERSION = "3.3.0"
BASE_URL = "https://www1.mbrace.or.jp/od2/K/{yyyymm}/k{yymmdd}.lzh"
REFERER = "https://www1.mbrace.or.jp/od2/K/dindex.html"

VENUES = {
    "01": ("桐生", ["桐生"]),
    "02": ("戸田", ["戸田"]),
    "03": ("江戸川", ["江戸川"]),
    "04": ("平和島", ["平和島"]),
    "05": ("多摩川", ["多摩川"]),
    "06": ("浜名湖", ["浜名湖"]),
    "07": ("蒲郡", ["蒲郡"]),
    "08": ("常滑", ["常滑"]),
    "09": ("津", [" 津 ", "津"]),
    "10": ("三国", ["三国"]),
    "11": ("びわこ", ["びわこ", "琵琶湖"]),
    "12": ("住之江", ["住之江"]),
    "13": ("尼崎", ["尼崎"]),
    "14": ("鳴門", ["鳴門"]),
    "15": ("丸亀", ["丸亀"]),
    "16": ("児島", ["児島"]),
    "17": ("宮島", ["宮島"]),
    "18": ("徳山", ["徳山"]),
    "19": ("下関", ["下関"]),
    "20": ("若松", ["若松"]),
    "21": ("芦屋", ["芦屋"]),
    "22": ("福岡", ["福岡"]),
    "23": ("唐津", ["唐津", "からつ"]),
    "24": ("大村", ["大村"]),
}

RACER_HEADERS = ["登録番号", "最新出走日"]
for p in ["30日", "90日", "365日", "3年", "5年"]:
    RACER_HEADERS += [
        f"{p}_出走", f"{p}_1着率", f"{p}_2連率", f"{p}_3連率",
        f"{p}_平均ST", f"{p}_ST偏差"
    ]

RACER_COURSE_HEADERS = [
    "登録番号", "コース",
    "365日_出走", "365日_1着率", "365日_2連率", "365日_3連率", "365日_平均ST",
    "5年_出走", "5年_1着率", "5年_2連率", "5年_3連率", "5年_平均ST",
]

RACER_VENUE_HEADERS = [
    "登録番号", "場コード", "会場",
    "365日_出走", "365日_1着率", "365日_2連率", "365日_3連率", "365日_平均ST",
    "5年_出走", "5年_1着率", "5年_2連率", "5年_3連率", "5年_平均ST",
]

VENUE_COURSE_HEADERS = [
    "場コード", "会場", "コース",
    "365日_出走", "365日_1着率", "365日_2連率", "365日_3連率", "365日_平均ST",
    "5年_出走", "5年_1着率", "5年_2連率", "5年_3連率", "5年_平均ST",
]

VENUE_COND_HEADERS = [
    "場コード", "会場", "コース", "風向コード", "風速帯",
    "5年_出走", "5年_1着率", "5年_2連率", "5年_3連率", "5年_平均ST",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-09-20")
    ap.add_argument("--end", default="2026-09-19")
    ap.add_argument("--out", default="output_5y")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rolling", action="store_true", help="昨日までの直近5年を自動設定")
    ap.add_argument("--upload-url", default=os.environ.get("HISTORY_ENDPOINT_URL", ""))
    ap.add_argument("--upload-token", default=os.environ.get("HISTORY_TOKEN", ""))
    ap.add_argument("--no-upload", action="store_true")
    return ap.parse_args()


def d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def safe_year_shift(dt: date, years: int) -> date:
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        return dt.replace(month=2, day=28, year=dt.year + years)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def detect_venue(text: str):
    # 長い名称から先に判定
    for code, (name, aliases) in VENUES.items():
        for a in sorted(aliases, key=len, reverse=True):
            if a and a in text:
                return code, name
    return None, None


def parse_st(raw: str):
    s = raw.strip().replace(" ", "")
    if not s or "F" in s.upper() or "L" in s.upper():
        return None
    m = re.search(r"(?<!\d)(?:0)?\.(\d{1,2})(?!\d)", s)
    if not m:
        return None
    try:
        return float("0." + m.group(1).zfill(2))
    except Exception:
        return None


def parse_rank(raw: str):
    # Kデータでは環境・日付により "1" / "01" / 全角数字相当の揺れを吸収する。
    s = raw.strip().translate(str.maketrans("１２３４５６０", "1234560"))
    m = re.fullmatch(r"0?([1-6])", s)
    return int(m.group(1)) if m else None


def parse_wind(header: str):
    # 競走成績の固定長レイアウトを優先。取得できない場合は空欄。
    try:
        if "進入固定" in header:
            wind = header[46:49].strip()
            speed_raw = header[49:53]
        else:
            wind = header[50:53].strip()
            speed_raw = header[53:57]
        m = re.search(r"(\d+)", speed_raw)
        speed = int(m.group(1)) if m else None
    except Exception:
        wind, speed = "", None

    # 固定位置が取れない時の簡易フォールバック
    if speed is None:
        ms = re.search(r"風.{0,8}?(\d+)\s*[mｍ]", header)
        if ms:
            speed = int(ms.group(1))

    if not wind:
        # よくある風向語
        for w in ["北東", "南東", "南西", "北西", "北", "南", "東", "西"]:
            if w in header:
                wind = w
                break

    if speed is None:
        band = ""
    elif speed == 0:
        band = "0m"
    elif speed <= 2:
        band = "1-2m"
    elif speed <= 4:
        band = "3-4m"
    else:
        band = "5m以上"
    return wind, speed, band


def parse_result_row(line: str):
    # BOAT RACE公式Kデータの固定長行
    if len(line) < 47:
        return None
    try:
        boat = line[6:7].strip()
        reg = line[8:12].strip()
        if boat not in "123456" or not re.fullmatch(r"\d{4}", reg):
            return None

        rank_text = line[2:4].strip()
        rank_num = parse_rank(rank_text)
        # 固定位置で取れないレイアウトは艇番位置(6)より前だけを探索し、艇番誤認を避ける。
        if rank_num is None:
            lead = line[:6].translate(str.maketrans("１２３４５６０", "1234560"))
            mm = re.search(r"(?<!\d)0?([1-6])(?!\d)", lead)
            if mm:
                rank_num = int(mm.group(1))
        racer_name = line[13:20].strip()
        course_raw = line[38:39].strip()
        course = int(course_raw) if course_raw in "123456" else None
        st_raw = line[43:47]
        return {
            "rank_text": rank_text,
            "rank_num": rank_num,
            "boat": int(boat),
            "reg": reg,
            "racer_name": racer_name,
            "course": course,
            "st": parse_st(st_raw),
        }
    except Exception:
        return None


def parse_k_text(text: str, race_date: date):
    lines = text.splitlines()
    out = []
    current_code = None
    current_name = None

    for i, line in enumerate(lines):
        if "競走成績" in line:
            block = "\n".join(lines[i:i+9])
            code, name = detect_venue(block)
            if code:
                current_code, current_name = code, name
            continue

        m = re.match(r"^\s*(\d{1,2})R(?:\s|$)", line)
        if not m or not current_code:
            continue

        race_no = int(m.group(1))
        wind, wind_speed, wind_band = parse_wind(line)

        # レース見出しの後ろから選手行を最大24行探索
        seen = False
        for j in range(i + 1, min(i + 25, len(lines))):
            nxt = lines[j]
            # 次のレース見出しに到達
            if j > i + 1 and re.match(r"^\s*(\d{1,2})R(?:\s|$)", nxt):
                break
            row = parse_result_row(nxt)
            if row:
                seen = True
                row.update({
                    "date": race_date.isoformat(),
                    "venue_code": current_code,
                    "venue_name": current_name,
                    "race_no": race_no,
                    "wind_dir": wind,
                    "wind_speed": wind_speed,
                    "wind_band": wind_band,
                })
                out.append(row)
            elif seen and not nxt.strip():
                break
    return out


def decode_bytes(b: bytes):
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("cp932", errors="replace")


def extract_lzh(path: Path):
    if lhafile is None:
        raise RuntimeError("lhafile が未インストールです。requirements.txt をインストールしてください。")
    arc = lhafile.Lhafile(str(path))
    infos = arc.infolist()
    if not infos:
        raise RuntimeError("LZH内にファイルがありません")
    # 通常は KYYMMDD.TXT 1ファイル
    info = infos[0]
    raw = arc.read(info.filename)
    return decode_bytes(raw)


def looks_like_lzh(data: bytes):
    head = data[:64].lower()
    return (b"-lh" in head) and len(data) > 100


def ensure_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
      date TEXT NOT NULL,
      venue_code TEXT NOT NULL,
      venue_name TEXT NOT NULL,
      race_no INTEGER NOT NULL,
      rank_text TEXT,
      rank_num INTEGER,
      boat INTEGER NOT NULL,
      reg TEXT NOT NULL,
      racer_name TEXT,
      course INTEGER,
      st REAL,
      wind_dir TEXT,
      wind_speed INTEGER,
      wind_band TEXT,
      PRIMARY KEY(date, venue_code, race_no, boat, reg)
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS download_log (
      date TEXT PRIMARY KEY,
      status TEXT NOT NULL,
      rows INTEGER NOT NULL DEFAULT 0,
      message TEXT,
      updated_at TEXT NOT NULL
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_reg_date ON results(reg, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_reg_course_date ON results(reg, course, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_reg_venue_date ON results(reg, venue_code, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_venue_course_date ON results(venue_code, course, date)")
    conn.commit()


def log_download(conn, dt, status, rows, msg=""):
    conn.execute("""
      INSERT INTO download_log(date,status,rows,message,updated_at)
      VALUES(?,?,?,?,?)
      ON CONFLICT(date) DO UPDATE SET
        status=excluded.status, rows=excluded.rows, message=excluded.message, updated_at=excluded.updated_at
    """, (dt.isoformat(), status, rows, msg[:500], datetime.now().isoformat(timespec="seconds")))
    conn.commit()


def already_done(conn, dt):
    r = conn.execute("SELECT status FROM download_log WHERE date=?", (dt.isoformat(),)).fetchone()
    return bool(r and r[0] in ("OK", "MISSING"))


def download_day(session, dt: date, cache_dir: Path, force=False):
    yyyymm = dt.strftime("%Y%m")
    yymmdd = dt.strftime("%y%m%d")
    url = BASE_URL.format(yyyymm=yyyymm, yymmdd=yymmdd)
    path = cache_dir / f"k{yymmdd}.lzh"

    if path.exists() and path.stat().st_size > 100 and not force:
        return path, url, "CACHE"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": REFERER,
        "Accept": "*/*",
    }
    last_err = None
    for n in range(3):
        try:
            r = session.get(url, headers=headers, timeout=40)
            if r.status_code == 404:
                return None, url, "MISSING"
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                time.sleep(1.0 + n)
                continue
            if not looks_like_lzh(r.content):
                # 非開催日等でHTMLが返るケースは欠損扱い
                if b"<html" in r.content[:500].lower() or len(r.content) < 100:
                    return None, url, "MISSING"
                last_err = "LZH形式として認識できません"
                time.sleep(1.0 + n)
                continue
            path.write_bytes(r.content)
            return path, url, "DOWNLOADED"
        except Exception as e:
            last_err = repr(e)
            time.sleep(1.0 + n)
    raise RuntimeError(last_err or "download failed")


def insert_rows(conn, rows):
    if not rows:
        return 0
    vals = []
    for r in rows:
        vals.append((
            r["date"], r["venue_code"], r["venue_name"], r["race_no"],
            r["rank_text"], r["rank_num"], r["boat"], r["reg"], r["racer_name"],
            r["course"], r["st"], r["wind_dir"], r["wind_speed"], r["wind_band"]
        ))
    conn.executemany("""
      INSERT OR REPLACE INTO results(
        date,venue_code,venue_name,race_no,rank_text,rank_num,boat,reg,racer_name,
        course,st,wind_dir,wind_speed,wind_band
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, vals)
    conn.commit()
    return len(vals)


def aggregate_map(conn, start_s, end_s, group_cols, extra_where=""):
    cols = ", ".join(group_cols)
    sql = f"""
    SELECT {cols},
           COUNT(*) AS starts,
           SUM(CASE WHEN rank_num=1 THEN 1 ELSE 0 END) AS wins,
           SUM(CASE WHEN rank_num BETWEEN 1 AND 2 THEN 1 ELSE 0 END) AS top2,
           SUM(CASE WHEN rank_num BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS top3,
           AVG(st) AS avg_st,
           AVG(st*st) AS avg_st2
    FROM results
    WHERE date BETWEEN ? AND ?
      {extra_where}
    GROUP BY {cols}
    """
    out = {}
    for row in conn.execute(sql, (start_s, end_s)):
        key_vals = row[:len(group_cols)]
        key = key_vals[0] if len(key_vals) == 1 else tuple(key_vals)
        starts, wins, top2, top3, avg_st, avg_st2 = row[len(group_cols):]
        if starts:
            out[key] = {
                "starts": int(starts),
                "win": wins * 100.0 / starts,
                "top2": top2 * 100.0 / starts,
                "top3": top3 * 100.0 / starts,
                "avg_st": avg_st,
                "std_st": (math.sqrt(max(avg_st2 - avg_st * avg_st, 0.0))
                           if avg_st is not None and avg_st2 is not None else None),
            }
    return out


def fmt_rate(v):
    return "" if v is None else round(v, 2)


def fmt_st(v):
    return "" if v is None else round(v, 3)


def stat5(s):
    if not s:
        return ["", "", "", "", ""]
    return [s["starts"], fmt_rate(s["win"]), fmt_rate(s["top2"]), fmt_rate(s["top3"]), fmt_st(s["avg_st"])]


def stat6(s):
    if not s:
        return ["", "", "", "", "", ""]
    return [s["starts"], fmt_rate(s["win"]), fmt_rate(s["top2"]), fmt_rate(s["top3"]),
            fmt_st(s["avg_st"]), fmt_st(s["std_st"])]


def write_csv(path: Path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def build_outputs(conn, out_dir: Path, start: date, end: date):
    print("\n[集計] CSV生成を開始します。")
    periods = {
        "30日": max(start, end - timedelta(days=29)),
        "90日": max(start, end - timedelta(days=89)),
        "365日": max(start, end - timedelta(days=364)),
        "3年": max(start, safe_year_shift(end, -3) + timedelta(days=1)),
        "5年": start,
    }

    # 選手
    latest = {r[0]: r[1] for r in conn.execute(
        "SELECT reg, MAX(date) FROM results WHERE date BETWEEN ? AND ? GROUP BY reg",
        (start.isoformat(), end.isoformat())
    )}
    racer_stats = {}
    for label, st in periods.items():
        print(f"  - 選手 {label}")
        racer_stats[label] = aggregate_map(conn, st.isoformat(), end.isoformat(), ["reg"])

    regs = sorted(set(latest) | set(racer_stats["5年"]))
    racer_rows = []
    for reg in regs:
        row = [reg, latest.get(reg, "")]
        for label in ["30日", "90日", "365日", "3年", "5年"]:
            row += stat6(racer_stats[label].get(reg))
        racer_rows.append(row)
    write_csv(out_dir / "BR_history_racer.csv", RACER_HEADERS, racer_rows)

    # 選手×コース
    rc365 = aggregate_map(conn, periods["365日"].isoformat(), end.isoformat(),
                          ["reg", "course"], "AND course BETWEEN 1 AND 6")
    rc5 = aggregate_map(conn, start.isoformat(), end.isoformat(),
                        ["reg", "course"], "AND course BETWEEN 1 AND 6")
    rc_keys = sorted(set(rc365) | set(rc5), key=lambda x: (x[0], int(x[1])))
    rc_rows = [[k[0], k[1]] + stat5(rc365.get(k)) + stat5(rc5.get(k)) for k in rc_keys]
    write_csv(out_dir / "BR_history_racer_course.csv", RACER_COURSE_HEADERS, rc_rows)

    # 選手×会場
    rv365 = aggregate_map(conn, periods["365日"].isoformat(), end.isoformat(), ["reg", "venue_code"])
    rv5 = aggregate_map(conn, start.isoformat(), end.isoformat(), ["reg", "venue_code"])
    rv_keys = sorted(set(rv365) | set(rv5), key=lambda x: (x[0], x[1]))
    rv_rows = []
    for k in rv_keys:
        name = VENUES.get(k[1], ("", []))[0]
        rv_rows.append([k[0], k[1], name] + stat5(rv365.get(k)) + stat5(rv5.get(k)))
    write_csv(out_dir / "BR_history_racer_venue.csv", RACER_VENUE_HEADERS, rv_rows)

    # 会場×コース
    vc365 = aggregate_map(conn, periods["365日"].isoformat(), end.isoformat(),
                          ["venue_code", "course"], "AND course BETWEEN 1 AND 6")
    vc5 = aggregate_map(conn, start.isoformat(), end.isoformat(),
                        ["venue_code", "course"], "AND course BETWEEN 1 AND 6")
    vc_keys = sorted(set(vc365) | set(vc5), key=lambda x: (x[0], int(x[1])))
    vc_rows = []
    for k in vc_keys:
        name = VENUES.get(k[0], ("", []))[0]
        vc_rows.append([k[0], name, k[1]] + stat5(vc365.get(k)) + stat5(vc5.get(k)))
    write_csv(out_dir / "BR_history_venue_course.csv", VENUE_COURSE_HEADERS, vc_rows)

    # 会場×コース×風条件（現行v3.1.1では予備情報。将来拡張用）
    cond = aggregate_map(
        conn, start.isoformat(), end.isoformat(),
        ["venue_code", "course", "wind_dir", "wind_band"],
        "AND course BETWEEN 1 AND 6 AND wind_dir<>'' AND wind_band<>''"
    )
    cond_keys = sorted(cond, key=lambda x: (x[0], int(x[1]), x[2], x[3]))
    cond_rows = []
    for k in cond_keys:
        name = VENUES.get(k[0], ("", []))[0]
        cond_rows.append([k[0], name, k[1], k[2], k[3]] + stat5(cond[k]))
    write_csv(out_dir / "BR_history_venue_condition.csv", VENUE_COND_HEADERS, cond_rows)

    total_rows = conn.execute(
        "SELECT COUNT(*) FROM results WHERE date BETWEEN ? AND ?",
        (start.isoformat(), end.isoformat())
    ).fetchone()[0]
    race_count = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT date, venue_code, race_no
             FROM results
             WHERE date BETWEEN ? AND ?
             GROUP BY date, venue_code, race_no
           )""",
        (start.isoformat(), end.isoformat())
    ).fetchone()[0]
    days_ok = conn.execute(
        "SELECT COUNT(*) FROM download_log WHERE date BETWEEN ? AND ? AND status='OK'",
        (start.isoformat(), end.isoformat())
    ).fetchone()[0]

    meta_rows = [
        ["履歴開始日", start.isoformat()],
        ["履歴終了日", end.isoformat()],
        ["作成日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["ビルダーバージョン", VERSION],
        ["ソース", "BOAT RACE公式 競走成績ダウンロード(Kデータ)"],
        ["取得成功日数", days_ok],
        ["レース数", race_count],
        ["艇結果行数", total_rows],
        ["選手数", len(racer_rows)],
        ["選手コース組数", len(rc_rows)],
        ["選手会場組数", len(rv_rows)],
        ["会場コース組数", len(vc_rows)],
        ["会場条件組数", len(cond_rows)],
    ]
    write_csv(out_dir / "BR_history_meta.csv", ["項目", "値"], meta_rows)

    summary = (
        f"BOAT RACE 過去5年履歴ビルダー v{VERSION}\n"
        f"期間: {start.isoformat()} ～ {end.isoformat()}\n"
        f"取得成功日数: {days_ok}\n"
        f"レース数: {race_count}\n"
        f"艇結果行数: {total_rows}\n"
        f"選手数: {len(racer_rows)}\n"
        f"選手×コース: {len(rc_rows)}\n"
        f"選手×会場: {len(rv_rows)}\n"
        f"会場×コース: {len(vc_rows)}\n"
        f"会場条件: {len(cond_rows)}\n"
    )
    (out_dir / "BUILD_SUMMARY.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print("[完了] 次の6ファイルをGoogle Driveへアップロードしてください。")
    for name in [
        "BR_history_racer.csv",
        "BR_history_racer_course.csv",
        "BR_history_racer_venue.csv",
        "BR_history_venue_course.csv",
        "BR_history_venue_condition.csv",
        "BR_history_meta.csv",
    ]:
        print("  ", out_dir / name)



def rolling_period(today=None):
    today = today or date.today()
    end = today - timedelta(days=1)
    start = safe_year_shift(end, -5) + timedelta(days=1)
    return start, end


def purge_old_cache_rows(conn, start: date):
    # SQLiteキャッシュはローリング開始日より前を削除して肥大化を抑える。
    conn.execute("DELETE FROM results WHERE date < ?", (start.isoformat(),))
    conn.execute("DELETE FROM download_log WHERE date < ?", (start.isoformat(),))
    conn.commit()


def upload_history_csvs(out_dir: Path, url: str, token: str, start: date, end: date):
    if not url or not token:
        raise RuntimeError("HISTORY_ENDPOINT_URL / HISTORY_TOKEN が未設定です")
    names = [
        "BR_history_racer.csv",
        "BR_history_racer_course.csv",
        "BR_history_racer_venue.csv",
        "BR_history_venue_course.csv",
        "BR_history_venue_condition.csv",
        "BR_history_meta.csv",
    ]
    sess = requests.Session()
    for i, name in enumerate(names):
        path = out_dir / name
        raw = path.read_bytes()
        payload = {
            "action": "upload_history_csv",
            "token": token,
            "filename": name,
            "history_start": start.isoformat(),
            "history_end": end.isoformat(),
            "builder_version": VERSION,
            "final": i == len(names) - 1,
            "data_gzip_base64": base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii"),
        }
        last = None
        for attempt in range(4):
            try:
                r = sess.post(url, json=payload, timeout=180)
                last = r
                if r.status_code == 200:
                    data = r.json()
                    if data.get("ok"):
                        print(f"[UPLOAD] {name}: OK rows={data.get('rows','-')}")
                        break
                    raise RuntimeError("receiver error: " + str(data))
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            except Exception as e:
                if attempt >= 3:
                    raise
                print(f"[UPLOAD] {name}: retry {attempt+1}/3: {e}")
                time.sleep(3 * (attempt + 1))


def self_test():
    # 実際のKデータと同じ固定位置を使った最小行でパーサを検証
    chars = [" "] * 70
    chars[2:4] = list(" 1")
    chars[6] = "3"
    chars[8:12] = list("4895")
    name = "門間 雄大"
    for idx, ch in enumerate(name[:7]):
        chars[13 + idx] = ch
    chars[38] = "3"
    chars[43:47] = list(" .15")
    row = "".join(chars)
    parsed = parse_result_row(row)
    assert parsed and parsed["reg"] == "4895" and parsed["boat"] == 3 and parsed["course"] == 3
    assert parsed["rank_num"] == 1
    assert parse_rank("01") == 1 and parse_rank(" 6 ") == 6
    assert abs(parsed["st"] - 0.15) < 1e-9
    print("SELF-TEST OK")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    if lhafile is None:
        print("ERROR: lhafile がありません。先に requirements.txt をインストールしてください。")
        return 2

    if args.rolling:
        start, end = rolling_period()
    else:
        start, end = d(args.start), d(args.end)
    if end < start:
        raise SystemExit("end must be >= start")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache_lzh"
    cache_dir.mkdir(exist_ok=True)
    db_path = out_dir / "boatrace_history.sqlite"

    print(f"BOAT RACE 過去履歴 v{VERSION}")
    print(f"期間: {start} ～ {end}")
    print(f"保存先: {out_dir}")
    print("途中で止まっても、同じBATを再実行すれば続きから再開できます。\n")

    conn = sqlite3.connect(db_path)
    ensure_db(conn)
    purge_old_cache_rows(conn, start)
    session = requests.Session()

    total_days = (end - start).days + 1
    ok = missing = error = skipped = 0

    for idx, dt in enumerate(daterange(start, end), start=1):
        if already_done(conn, dt) and not args.force_download:
            skipped += 1
            if idx % 50 == 0 or idx == total_days:
                print(f"[{idx}/{total_days}] {dt} 既処理 / OK={ok} SKIP={skipped} MISSING={missing} ERROR={error}")
            continue

        try:
            path, url, status = download_day(session, dt, cache_dir, args.force_download)
            if status == "MISSING":
                missing += 1
                log_download(conn, dt, "MISSING", 0, url)
            else:
                text = extract_lzh(path)
                rows = parse_k_text(text, dt)
                if not rows:
                    # アーカイブがあるのに0行ならレイアウト変化の可能性があるのでERROR扱い
                    error += 1
                    log_download(conn, dt, "ERROR", 0, "解析0行: " + url)
                    print(f"[{idx}/{total_days}] {dt} WARNING: 解析0行")
                else:
                    n = insert_rows(conn, rows)
                    ok += 1
                    log_download(conn, dt, "OK", n, url)
        except Exception as e:
            error += 1
            log_download(conn, dt, "ERROR", 0, repr(e))
            print(f"[{idx}/{total_days}] {dt} ERROR: {e}")

        if idx % 10 == 0 or idx == total_days:
            print(f"[{idx}/{total_days}] {dt} OK={ok} SKIP={skipped} MISSING={missing} ERROR={error}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # ERRORの日が少数でも、取得できたデータからCSVは生成する。
    build_outputs(conn, out_dir, start, end)
    conn.close()

    if not args.no_upload and args.upload_url and args.upload_token:
        print("\n[クラウド] Apps Scriptへ6CSVを送信します。")
        upload_history_csvs(out_dir, args.upload_url, args.upload_token, start, end)
    elif not args.no_upload:
        print("\n[クラウド] URL/トークン未設定のため自動送信はスキップしました。")

    if error:
        print(f"\n注意: ERROR日が {error} 日あります。output内のSQLite download_logを確認してください。")
        print("同じBATをもう一度実行すると、ERROR日のみ再試行します。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
