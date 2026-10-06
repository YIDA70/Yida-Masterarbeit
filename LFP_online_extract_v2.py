"""LFP 在线筛选与 R0/R1/R2 提取；基于 LFP_time_diff_v4_BM_2s。
安装：python -m pip install pandas numpy pyarrow boto3
运行：python LFP_online_extract_v2.py
只读取远端对象；逐电芯下载到临时文件，分批读取，结束后删除临时原始数据。
"""
import gc
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from botocore.config import Config

# %% 参数：与原在线 notebook 使用同一 endpoint / credentials.json 格式
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "https://iseadocker.isea.rwth-aachen.de:9000")
CREDENTIALS_FILE = Path("credentials.json")  # {"accessKey": "...", "secretKey": "..."}
BUCKET_NAME = "projects"
BASE_PREFIX = "j8005-metabatt/Metabatt/A123/10_TRACY/GOLD/"
BATTERY_IDS = range(7, 10)  # 先验证 007–009；改为 None 遍历全部电芯
OUTPUT_ROOT = Path("LFP_online_results")
BATCH_SIZE = 65536
PROGRESS_EVERY = 10

CHEMISTRY = "LFP"
ACN1_A = 1.2
NOMINAL_CAPACITY_AH = 1.2
SOH_REFERENCE = "rated"  # rated：额定容量；bol：最小 BM 的首个有效容量
SOC_ORDER = ["90%", "50%", "10%"]
NOMINAL_CURRENT_LEVELS_A = np.array([2.4, 4.8, 7.2], dtype=float)
PULSE_TARGETS = ["PUL", "PUL*RES"]
PULSE_PROCEDURE = "jri_APR_CU_Pulse"
BLOCK_GAP_MIN = 120.0
ZERO_CURRENT_LIMIT = 1e-6

# LFP 协议筛选在 load-case 删除前进行：每 SOC 为 CHA/DCH × 2/4/6C。
REQUIRE_COMPLETE_LOAD_CASES = True
MIN_PULSES_PER_SOC = 1  # 关闭完整组合检查后使用的宽松数量下限
MAX_PULSES_PER_SOC = 6
MIN_CHECKUPS_EXCLUSIVE = 3
MIN_A_PASSED_SOH = 5
EXCLUDE_DCH_2C = True  # 对应 v4 DROP_FIRST_PULSE 的实际行为：所有 DCH-2C
APPLY_LOADCASE_REMOVAL = True
HIGH_CRATE_LEVELS = [4, 6]

# 冻结 v4 电阻定义：R0 外推到 PAU 末点后 0.5s，R1/R2 锚点为 10s/20s。
R0_TARGET_AFTER_PAUSE_SEC = 0.5
R1_ANCHOR_SEC, R2_ANCHOR_SEC = 10.0, 20.0
R0_FIT_POINT_START, R0_FIT_POINT_END = 2, 6
FIRST_TWO_VOLTAGE_EQUAL_ATOL = 1e-9
VOLTAGE_JUMP_LIMIT = 1e-3
MAX_PAUSE_TO_PULSE_GAP_SEC = 2.0
CC_CURRENT_DEVIATION_RATIO = 0.03
STD_LIMIT = {2.4: 0.1, 4.8: 0.1, 7.2: 0.1}
CC_VALID, CC_INSUF = "Valid", "Insufficient duration"
CC_DEV, CC_NE = "Current deviation > 3%", "Not evaluated"
FILE_PATTERN = re.compile(r"^METABatt_A123_APR18650M1B_(\d+)\.parquet$", re.I)
PULSE_COLS = ["Time", "Current", "Voltage", "Zustand", "target", "Prozedur", "BM_Programm", "ID"]
INV_COLS = ["pulse_uid", "BM_Programm", "ID", "Zustand", "I_peak", "t0",
            "Has_Nonzero_Current", "Nominal_Current_A", "C_rate", "block_in_bm",
            "SOC", "Capacity_Ah", "SOH", "chemistry", "Battery_ID", "load_case", "load_case_in_scope"]
BM_COLS = ["Battery_ID", "BM_Programm", "Capacity_Ah", "SOH", "SOH_Valid",
           "SOC_Block_Count", "Present_SOC", "Has_10_50_90_SOC", "Passed", "Reason"]
OUTPUT_COLUMNS = ["SOH", "SOC", "Battery_ID", "Time", "ID", "Zustand", "Zustand/Current",
    "Current", "Voltage", "Nominal_Current_A", "C_rate", "chemistry", "R0", "R1", "R2",
    "R0_mOhm", "R1_mOhm", "R2_mOhm", "CC_Valid_10s", "CC_Valid_20s",
    "CC_Max_Deviation_10s_pct", "CC_Max_Deviation_20s_pct", "Pause_to_Pulse_Time_Diff_s",
    "R0_Quality", "load_case", "load_case_in_scope", "BM_Programm", "Capacity_Ah",
    "pulse_uid", "Object_Key", "筛选分类", "用途"]


# %% 在线连接与分批读取
def make_s3_client():
    with CREDENTIALS_FILE.open(encoding="utf-8") as f:
        credentials = json.load(f)
    if not isinstance(credentials, dict):
        raise ValueError("credentials.json 必须是 JSON 对象。")
    keys = {k: credentials.get(k) for k in ["accessKey", "secretKey"]}
    if not all(isinstance(v, str) and v.strip() for v in keys.values()):
        raise ValueError("credentials.json 必须包含非空 accessKey / secretKey。")
    keys = {k: v.strip() for k, v in keys.items()}
    return boto3.client("s3", endpoint_url=MINIO_ENDPOINT, region_name="us-east-1",
        aws_access_key_id=keys["accessKey"], aws_secret_access_key=keys["secretKey"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      connect_timeout=30, read_timeout=120, retries={"max_attempts": 3}))


def list_battery_objects(client):
    selected = None if BATTERY_IDS is None else set(map(int, BATTERY_IDS))
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET_NAME, Prefix=BASE_PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            match = FILE_PATTERN.fullmatch(key[len(BASE_PREFIX):])
            if key.startswith(BASE_PREFIX) and match:
                number = int(match.group(1))
                if selected is None or number in selected:
                    objects.append((number, key))
    if not objects:
        raise FileNotFoundError("未找到 LFP 电芯；检查 BASE_PREFIX / BATTERY_IDS。")
    if len({n for n, _ in objects}) != len(objects):
        raise ValueError("同一电芯编号对应多个对象；请先消除重复文件。")
    return sorted(objects)


def iter_frames(pf, columns, targets=None, procedure=None):
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        table = pa.Table.from_batches([batch])
        if procedure is not None:
            table = table.filter(pc.equal(table["Prozedur"], procedure))
        if targets is not None:
            table = table.filter(pc.is_in(table["target"], value_set=pa.array(targets)))
        if table.num_rows:
            yield table.to_pandas()


def clean_bm(frame):
    frame = frame.copy()
    bm = pd.to_numeric(frame["BM_Programm"], errors="coerce")
    valid = bm.notna() & np.isfinite(bm) & bm.ge(0) & bm.mod(1).eq(0)
    frame = frame.loc[valid].copy()
    frame["BM_Programm"] = bm.loc[valid].astype("int64")
    return frame


def utc_time(values):
    return pd.to_datetime(values, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def build_capacity_map(pf):
    # 与 v4 一致：文件物理顺序中，每个 BM 首个非空 Capacity_py。
    capacity = {}
    for frame in iter_frames(pf, ["BM_Programm", "Capacity_py"]):
        frame = clean_bm(frame)
        frame["Capacity_py"] = pd.to_numeric(frame["Capacity_py"], errors="coerce")
        frame = frame.loc[np.isfinite(frame["Capacity_py"])]
        for bm, group in frame.groupby("BM_Programm", sort=False):
            capacity.setdefault(int(bm), float(group["Capacity_py"].iloc[0]))
    return capacity


def build_pulse_inventory(pf, capacity, reference, battery_id):
    parts, invalid = [], 0
    for frame in iter_frames(pf, PULSE_COLS, PULSE_TARGETS, PULSE_PROCEDURE):
        before = len(frame)
        frame = clean_bm(frame)
        frame["Time"] = utc_time(frame["Time"])
        for col in ["Current", "Voltage"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
            frame.loc[~np.isfinite(frame[col]), col] = np.nan
        frame = frame.dropna(subset=["Time", "Current", "ID"])
        invalid += before - len(frame)
        z = frame["Zustand"].astype(str)
        frame["Zustand"] = np.where(z.str.startswith("DCH"), "DCH",
                                   np.where(z.str.startswith("CHA"), "CHA", z))
        frame["ID"] = frame["ID"].astype(str)
        frame["pulse_uid"] = battery_id + "|" + frame["BM_Programm"].astype(str) + "|" + frame["ID"]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=PULSE_COLS + ["pulse_uid"]), pd.DataFrame(columns=INV_COLS), invalid
    points = pd.concat(parts, ignore_index=True).sort_values(["BM_Programm", "Time"], kind="stable")
    g = points.groupby("pulse_uid", sort=False)
    inv = g.agg(BM_Programm=("BM_Programm", "first"), ID=("ID", "first"),
                Zustand=("Zustand", "first"), t0=("Time", "min"))
    inv["I_peak"] = g["Current"].apply(lambda s: s.abs().max())
    # 一个 BM+ID 必须代表一个方向；混合方向不能作为合格脉冲。
    inv["Has_Nonzero_Current"] = (inv["I_peak"] > ZERO_CURRENT_LIMIT) & g["Zustand"].nunique().eq(1)
    inv = inv.reset_index().sort_values(["BM_Programm", "t0"], kind="stable").reset_index(drop=True)
    lv = NOMINAL_CURRENT_LEVELS_A
    inv["Nominal_Current_A"] = inv["I_peak"].map(lambda x: float(lv[np.argmin(abs(lv - x))]))
    inv["C_rate"] = (inv["Nominal_Current_A"] / ACN1_A).round().astype(int)
    gap = inv.groupby("BM_Programm")["t0"].diff().dt.total_seconds() / 60
    inv["block_in_bm"] = gap.gt(BLOCK_GAP_MIN).groupby(inv["BM_Programm"]).cumsum().astype(int)
    inv["SOC"] = inv["block_in_bm"].map(lambda b: SOC_ORDER[b] if b < len(SOC_ORDER) else "ERR")
    inv["Capacity_Ah"] = inv["BM_Programm"].map(capacity)
    inv["SOH"] = (inv["Capacity_Ah"] / reference * 100).round(2)
    inv["chemistry"], inv["Battery_ID"] = CHEMISTRY, battery_id
    inv["load_case"] = inv["SOC"].str.rstrip("%") + "_" + inv["Zustand"] + "_" + inv["C_rate"].astype(str) + "C"
    inv["load_case_in_scope"] = inv["SOC"].isin(SOC_ORDER)
    if EXCLUDE_DCH_2C:
        low = int(round(lv.min() / ACN1_A))
        inv.loc[inv["Zustand"].eq("DCH") & inv["C_rate"].eq(low), "load_case_in_scope"] = False
    if APPLY_LOADCASE_REMOVAL:
        extreme = ((inv["SOC"].eq("90%") & inv["Zustand"].eq("CHA")) |
                   (inv["SOC"].eq("10%") & inv["Zustand"].eq("DCH")))
        inv.loc[extreme & inv["C_rate"].isin(HIGH_CRATE_LEVELS), "load_case_in_scope"] = False
    return points, inv[INV_COLS], invalid


# %% 按独立 BM 筛选，再决定电芯是否进入参数提取
def screen_battery(number, key, capacity, inv, reference, invalid):
    battery_id = f"A123_APR18650M1B_{number:03d}"
    expected = {(z, int(round(i / ACN1_A))) for z in ["CHA", "DCH"] for i in NOMINAL_CURRENT_LEVELS_A}
    rows = []
    for bm in sorted(set(capacity) | set(inv["BM_Programm"])):
        cap = capacity.get(bm, np.nan)
        raw_soh = cap / reference * 100
        valid_soh = bool(np.isfinite(raw_soh) and 0 < raw_soh <= 100)
        group = inv.loc[inv["BM_Programm"].eq(bm)]
        present = set(group["SOC"])
        blocks = int(group["block_in_bm"].nunique())
        triple = blocks == len(SOC_ORDER) and present == set(SOC_ORDER)
        reasons = []
        if not valid_soh:
            reasons.append("SOH 缺失/非正值" if not np.isfinite(raw_soh) or raw_soh <= 0 else "SOH > 100%")
        if not triple:
            reasons.append("不是恰好三个 SOC 块")
        for soc in SOC_ORDER:
            sg = group.loc[group["SOC"].eq(soc)]
            counts = sg.groupby(["Zustand", "C_rate"]).size()
            if not MIN_PULSES_PER_SOC <= len(sg) <= MAX_PULSES_PER_SOC:
                reasons.append(f"{soc} 脉冲数量不合格")
            if REQUIRE_COMPLETE_LOAD_CASES and (set(counts.index) != expected or not counts.eq(1).all()):
                reasons.append(f"{soc} 六种脉冲组合缺失/重复")
            if not sg.empty and (not sg["Has_Nonzero_Current"].all() or not sg["Zustand"].isin(["CHA", "DCH"]).all()):
                reasons.append(f"{soc} 脉冲方向/有效电流异常")
        rows.append(dict(Battery_ID=battery_id, BM_Programm=int(bm), Capacity_Ah=cap,
            SOH=round(raw_soh, 2) if np.isfinite(raw_soh) else np.nan, SOH_Valid=valid_soh,
            SOC_Block_Count=blocks, Present_SOC=",".join(s for s in SOC_ORDER if s in present),
            Has_10_50_90_SOC=triple, Passed=not reasons, Reason="；".join(reasons) or "无"))
    checks = pd.DataFrame(rows, columns=BM_COLS)
    valid = checks.loc[checks["SOH_Valid"].eq(True)]
    passed = valid.loc[valid["Passed"].eq(True)]
    passed_sohs = int(passed["SOH"].nunique())
    reasons = []
    if len(valid) <= MIN_CHECKUPS_EXCLUSIVE:
        reasons.append(f"check-up ≤ {MIN_CHECKUPS_EXCLUSIVE}")
    if valid.empty or not valid["Has_10_50_90_SOC"].all():
        reasons.append("剩余 check-up 未全部满足三 SOC 协议")
    if passed.empty:
        reasons.append("没有通过脉冲协议筛选的 check-up")
    classification = "剔除" if reasons else ("A完整" if passed_sohs >= MIN_A_PASSED_SOH else "B受限")
    usage = {"剔除": "不用", "A完整": "train/val/test", "B受限": "仅train"}[classification]
    summary = {"电芯编号": number, "Battery_ID": battery_id, "Object_Key": key,
        "单独剔除后 check-up 数量": len(valid), "通过筛选的 check-up 数量": len(passed),
        "通过筛选的 SOH 数量": passed_sohs, "SOH 单独剔除数量": len(checks) - len(valid),
        "是否满足三SOC协议": bool(not valid.empty and valid["Has_10_50_90_SOC"].all()),
        "初始 SOH": valid["SOH"].iloc[0] if not valid.empty else np.nan,
        "最终 SOH": valid["SOH"].iloc[-1] if not valid.empty else np.nan,
        "SOH 最小值": valid["SOH"].min(), "SOH 最大值": valid["SOH"].max(),
        "筛选分类": classification, "用途": usage, "剔除原因": "；".join(reasons) or "无",
        "原始脉冲数量": len(inv), "范围内脉冲数量": int(inv["load_case_in_scope"].sum()),
        "无效脉冲点数量": invalid, "提取行数": 0, "范围内提取行数": 0, "运行状态": "完成"}
    return summary, checks, set(passed["BM_Programm"])


def collect_pause_for_pulses(pf, inv):
    st = inv[["pulse_uid", "BM_Programm", "t0"]].sort_values("t0").reset_index(drop=True)
    best = st.set_index("pulse_uid")[[]]
    best["pause_time"] = pd.Series(pd.NaT, index=best.index, dtype="datetime64[ns, UTC]")
    best["pause_voltage"] = np.nan
    cols = ["Time", "Voltage", "target", "Prozedur", "BM_Programm"]
    for pau in iter_frames(pf, cols, ["PAU"], PULSE_PROCEDURE):
        pau = clean_bm(pau)
        pau["pause_time"] = utc_time(pau["Time"])
        pau["pause_voltage"] = pd.to_numeric(pau["Voltage"], errors="coerce").astype(float)
        pau.loc[~np.isfinite(pau["pause_voltage"]), "pause_voltage"] = np.nan
        pau = pau.dropna(subset=["pause_time"])
        pau = pau.loc[pau["BM_Programm"].isin(st["BM_Programm"])]
        if pau.empty or st.empty:
            continue
        matched = pd.merge_asof(st, pau[["BM_Programm", "pause_time", "pause_voltage"]].sort_values("pause_time"),
            left_on="t0", right_on="pause_time", by="BM_Programm", direction="backward").set_index("pulse_uid")
        take = matched["pause_time"].notna() & (best["pause_time"].isna() | matched["pause_time"].gt(best["pause_time"]))
        best.loc[take, ["pause_time", "pause_voltage"]] = matched.loc[take, ["pause_time", "pause_voltage"]]
    return best.to_dict("index")


# %% R0/R1/R2：有效正常输入与 v4 数值定义一致
def _add_q(base, new):
    return new if base == "正常" else base + "；" + new


def _eff_start(current):
    nz = np.flatnonzero(np.abs(current) > ZERO_CURRENT_LIMIT)
    return int(nz[0]) if len(nz) else None


def calculate_r0_for_segment(group, esp, pause_time, pause_voltage):
    group = group.sort_values("Time", kind="stable")
    o = dict(Nominal_Current_A=np.nan, R0=np.nan, R1=np.nan, R2=np.nan,
        CC_Valid_10s=pd.NA, CC_Valid_20s=pd.NA, CC_10s_Status=CC_NE, CC_20s_Status=CC_NE,
        CC_Max_Deviation_10s_pct=np.nan, CC_Max_Deviation_20s_pct=np.nan,
        Pause_to_Pulse_Time_Diff_s=np.nan, R0_Quality="正常")
    if group.empty or esp is None:
        o["R0_Quality"] = "无法计算R0：无非零有效电流"
        return o
    if group["Time"].duplicated().any():
        o["R0_Quality"] = "无法计算R0：脉冲存在重复时间点"
        return o
    pst, first_i = group["Time"].iloc[0], group["Current"].iloc[0]
    lv = NOMINAL_CURRENT_LEVELS_A
    o["Nominal_Current_A"] = float(lv[np.argmin(abs(lv - group["Current"].abs().max()))])
    if pause_time is None or pd.isna(pause_time):
        o["R0_Quality"] = "无法计算R0：无前置pause点"
        return o
    gap = (pst - pause_time).total_seconds()
    o["Pause_to_Pulse_Time_Diff_s"] = gap
    tgt = pause_time + pd.Timedelta(seconds=R0_TARGET_AFTER_PAUSE_SEC)
    if gap < 0 or gap > MAX_PAUSE_TO_PULSE_GAP_SEC:
        o["R0_Quality"] = f"pause结束点到pulse首点时间差不在[0,{MAX_PAUSE_TO_PULSE_GAP_SEC:g}]s"
        return o
    if abs(first_i) <= ZERO_CURRENT_LIMIT:
        fv = group["Voltage"].iloc[0]
        if pd.notna(fv) and pd.notna(pause_voltage) and abs(fv - pause_voltage) > VOLTAGE_JUMP_LIMIT:
            o["R0_Quality"] = "首点0且电压跳变"
    fps = R0_FIT_POINT_START
    if esp + 1 < len(group):
        v1, v2 = group["Voltage"].iloc[esp], group["Voltage"].iloc[esp + 1]
        if pd.notna(v1) and pd.notna(v2) and np.isclose(float(v1), float(v2), rtol=0, atol=FIRST_TWO_VOLTAGE_EQUAL_ATOL):
            fps = 3
    fit = group.iloc[esp + fps - 1:esp + R0_FIT_POINT_END].dropna(subset=["Time", "Voltage", "Current"])
    if len(fit) < 2:
        o["R0_Quality"] = _add_q(o["R0_Quality"], "无法计算R0：拟合点不足")
        return o
    fca = fit["Current"]
    level = float(lv[np.argmin(abs(lv - abs(fca.iloc[0])))])
    if fca.std() > STD_LIMIT.get(level, np.inf):
        o["R0_Quality"] = _add_q(o["R0_Quality"], "无法计算R0：早期拟合窗口电流不稳定")
        return o
    eff_i = group["Current"].iloc[esp]
    if pd.isna(pause_voltage):
        o["R0_Quality"] = _add_q(o["R0_Quality"], "无法计算R0：pause电压缺失")
        return o
    x = (fit["Time"] - tgt) / pd.Timedelta(seconds=1)
    _, intercept = np.polyfit(x.to_numpy(), fit["Voltage"].to_numpy(), 1)
    o["R0"] = abs((intercept - pause_voltage) / eff_i)
    anc = group.iloc[esp:].dropna(subset=["Time", "Voltage", "Current"])
    rel = (anc["Time"] - group["Time"].iloc[esp]) / pd.Timedelta(seconds=1)
    eref = fca.median()
    fsr = (fit["Time"].min() - group["Time"].iloc[esp]) / pd.Timedelta(seconds=1)

    def evaluate_cc_until(anchor):
        if len(anc) < 2 or rel.max() < anchor:
            return CC_INSUF, np.nan
        if fsr >= anchor or pd.isna(eref) or abs(eref) <= ZERO_CURRENT_LIMIT:
            return CC_NE, np.nan
        wc = anc.loc[(rel >= fsr) & (rel <= anchor), "Current"].to_numpy()
        ac = np.interp(anchor, rel.to_numpy(), anc["Current"].to_numpy())
        cv = np.append(wc, ac)
        if len(cv) < 2:
            return CC_NE, np.nan
        md = np.max(abs(cv - eref)) / abs(eref) * 100
        limit = CC_CURRENT_DEVIATION_RATIO * 100
        within = md <= limit or np.isclose(md, limit, rtol=0, atol=1e-10)
        return (CC_VALID, md) if within else (CC_DEV, md)

    s10, m10 = evaluate_cc_until(R1_ANCHOR_SEC)
    s20, m20 = evaluate_cc_until(R2_ANCHOR_SEC)
    v10, v20 = s10 == CC_VALID, s20 == CC_VALID
    o.update(CC_Valid_10s=v10, CC_Valid_20s=v20, CC_10s_Status=s10, CC_20s_Status=s20,
             CC_Max_Deviation_10s_pct=m10, CC_Max_Deviation_20s_pct=m20)
    if v10:
        vv = np.interp(R1_ANCHOR_SEC, rel.to_numpy(), anc["Voltage"].to_numpy())
        o["R1"] = abs((vv - pause_voltage) / eff_i) - o["R0"]
    if v20:
        a10, a20 = np.interp([R1_ANCHOR_SEC, R2_ANCHOR_SEC], rel.to_numpy(), anc["Voltage"].to_numpy())
        o["R2"] = abs((a20 - pause_voltage) / eff_i) - abs((a10 - pause_voltage) / eff_i)
    return o


def extract_resistances(points, inv, pause_map, key, summary):
    rows = []
    for uid, group in points.groupby("pulse_uid", sort=False):
        group = group.sort_values("Time", kind="stable")
        esp = _eff_start(group["Current"].to_numpy())
        pm = pause_map.get(uid, {})
        result = calculate_r0_for_segment(group, esp, pm.get("pause_time"), pm.get("pause_voltage"))
        result.update(pulse_uid=uid, Current=float(group["Current"].iloc[esp]) if esp is not None else 0.0,
                      Voltage=group["Voltage"].iloc[0], Time=group["Time"].iloc[0])
        rows.append(result)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.DataFrame(rows).merge(inv.drop(columns=["Nominal_Current_A"]), on="pulse_uid", validate="one_to_one")
    out["Zustand/Current"] = out["Zustand"] + "/" + out["Current"].round(1).astype(str)
    for r in ["R0", "R1", "R2"]:
        out[r + "_mOhm"] = out[r] * 1000
    out["Object_Key"], out["筛选分类"], out["用途"] = key, summary["筛选分类"], summary["用途"]
    return out[OUTPUT_COLUMNS].sort_values(["BM_Programm", "Time"]).reset_index(drop=True)


def process_local_parquet(path, number, key):
    # path 是远端文件的临时副本；标识来自远端对象名，不依赖临时文件名。
    with pq.ParquetFile(path) as pf:
        missing = (set(PULSE_COLS) | {"Capacity_py"}) - set(pf.schema_arrow.names)
        if missing:
            raise ValueError("缺少必要列：" + ", ".join(sorted(missing)))
        capacity = build_capacity_map(pf)
        reference = NOMINAL_CAPACITY_AH
        if SOH_REFERENCE == "bol":
            if not capacity:
                raise ValueError("无容量记录，无法建立 BOL 参考容量。")
            reference = capacity[min(capacity)]
        if SOH_REFERENCE not in ["rated", "bol"] or not np.isfinite(reference) or reference <= 0:
            raise ValueError("SOH_REFERENCE / SOH 参考容量无效。")
        points, inv, invalid = build_pulse_inventory(pf, capacity, reference, f"A123_APR18650M1B_{number:03d}")
        summary, checks, passed_bms = screen_battery(number, key, capacity, inv, reference, invalid)
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if summary["筛选分类"] != "剔除":
            selected = inv.loc[inv["BM_Programm"].isin(passed_bms)].copy()
            points = points.loc[points["pulse_uid"].isin(selected["pulse_uid"])]
            pause_map = collect_pause_for_pulses(pf, selected)
            result = extract_resistances(points, selected, pause_map, key, summary)
        summary["提取行数"] = len(result)
        summary["范围内提取行数"] = int(result["load_case_in_scope"].sum())
        return summary, checks, inv, result


# %% 批量驱动：逐电芯筛选与导出，空结果也具有完整表头
def append_csv(frame, path):
    if not frame.empty:
        frame.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")


def run_online(client=None):
    if (ACN1_A <= 0 or BLOCK_GAP_MIN <= 0 or not np.isfinite(NOMINAL_CURRENT_LEVELS_A).all()
            or not (NOMINAL_CURRENT_LEVELS_A > 0).all()):
        raise ValueError("电流档位、ACN1_A、SOC 时间阈值必须有效。")
    if len(SOC_ORDER) != 3 or len(set(SOC_ORDER)) != 3 or MIN_PULSES_PER_SOC > MAX_PULSES_PER_SOC:
        raise ValueError("需要三个不同 SOC 标签及有效脉冲数量上下限。")
    client = client if client is not None else make_s3_client()
    objects = list_battery_objects(client)
    folder = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    exports = {"LFP_R012_all.csv": OUTPUT_COLUMNS, "LFP_R012_in_scope.csv": OUTPUT_COLUMNS,
               "bm_check.csv": BM_COLS + ["Object_Key"], "pulse_inventory.csv": INV_COLS + ["Object_Key"]}
    for name, cols in exports.items():
        pd.DataFrame(columns=cols).to_csv(folder / name, index=False, encoding="utf-8-sig")
    summaries = []
    print(f"发现 {len(objects)} 个 LFP 电芯；结果目录：{folder.resolve()}")
    for position, (number, key) in enumerate(objects, 1):
        try:
            with tempfile.TemporaryDirectory(prefix="lfp_") as temporary:
                local = Path(temporary) / "battery.parquet"
                with local.open("wb") as f:
                    client.download_fileobj(BUCKET_NAME, key, f)
                summary, checks, inv, result = process_local_parquet(local, number, key)
        except Exception as exc:
            summary = {"电芯编号": number, "Battery_ID": f"A123_APR18650M1B_{number:03d}",
                       "Object_Key": key, "筛选分类": "剔除", "用途": "不用",
                       "运行状态": "失败", "剔除原因": f"{type(exc).__name__}: {exc}"}
        else:
            # 输出写入失败应直接报错，避免把部分已导出的电芯误记为读取失败。
            checks["Object_Key"], inv["Object_Key"] = key, key
            for name, frame in [("bm_check.csv", checks), ("pulse_inventory.csv", inv),
                                ("LFP_R012_all.csv", result),
                                ("LFP_R012_in_scope.csv", result.loc[result["load_case_in_scope"].eq(True)])]:
                append_csv(frame, folder / name)
            del checks, inv, result
        summaries.append(summary)
        gc.collect()
        if position == 1 or position % PROGRESS_EVERY == 0 or position == len(objects):
            print(f"{position}/{len(objects)} | {number:03d} | {summary['筛选分类']} | {summary['运行状态']}")
    battery_summary = pd.DataFrame(summaries).sort_values("电芯编号")
    battery_summary.to_csv(folder / "battery_summary.csv", index=False, encoding="utf-8-sig")
    quality = battery_summary.groupby(["筛选分类", "用途"]).agg(
        电芯数量=("电芯编号", "size"), 电芯编号=("电芯编号", lambda s: ",".join(f"{n:03d}" for n in s)))
    quality.to_csv(folder / "quality_summary.csv", encoding="utf-8-sig")
    print(quality.to_string())
    print("完成。电阻单位：R0/R1/R2 为 Ω，*_mOhm 为 mΩ。")
    print("in_scope 表保留电阻缺失及质量标记，请按后续任务筛选有效参数。")
    return battery_summary, folder


if __name__ == "__main__":
    battery_summary, output_dir = run_online()
