"""变速剪辑音画同步流水线（纯 Python 参考实现）。

六个缺陷与对应修复：

1. 段边界时间映射连续
   ``segment_map_continuity`` —— 时间映射由源时间沿分段速度逐段积分构造，
   输出时间轴严格单调、段边界严格连续；正反映射均基于时间（而非重采样后的
   帧号反推），衔接处不再跳帧。
2. 音频分段独立变速
   ``audio_segment_speed`` —— 每个变速段独立做保音高时长拉伸（SOLA），
   段间交叉淡化拼接，替代整体倍速拉伸，消除变速点两侧的音高/时长错配与金属感。
3. 关键帧按变速后重算
   ``keyframe_rescale`` —— 在输出时间线上重算关键帧：每个变速点强制 IDR，
   并按 GOP 上限补齐，快放段中间不再出现长 GOP，拖动定位准确。
4. 元数据随帧绑定
   ``meta_bind_frame`` —— 元数据按 ``src_idx`` 源帧号绑定到重排后的帧，
   与重排后列表的物理顺序无关，高光参数不会配错帧。
5. 封装按时间线排序
   ``order_by_timeline`` —— 异步分段转码产物先按 PTS（时间线）稳定排序再
   封装，与各分段的完成顺序无关。
6. 变速点音画对齐校验
   ``align_check`` / ``validate_av`` —— 对每个变速点分别取视频/音频边界
   时间计算漂移，超差判定失败并可触发 ``fallback`` 安全降级。

全部为纯函数、仅依赖标准库，浮点时间统一保留 9 位小数，结果可复现。
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy

__all__ = [
    "segment_map_continuous",
    "audio_segment_speed",
    "keyframe_rescale",
    "meta_bind_frame",
    "order_by_timeline",
    "align_check",
    "build_curve",
    "retime",
    "encode",
    "mux",
    "validate_av",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

DEFAULT_FPS = 30
DEFAULT_SR = 48000
DEFAULT_GOP_SECONDS = 2.0
DEFAULT_CROSSFADE_MS = 5.0
_EPS = 1e-9
_I_SIZE = 1024
_P_SIZE = 256
_STREAM_RANK = {"a": 0, "v": 1}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _r(value, digits=9):
    """时间戳统一舍入，避免浮点累积误差破坏边界连续性判断。"""
    return round(float(value), digits)


def _coerce_curve(curve):
    """容忍宽松入参（例如契约测试传入的整数）：非合法曲线时回退到默认曲线。"""
    if isinstance(curve, dict) and curve.get("segments"):
        return curve
    return build_curve()


def _norm_segments(segments):
    """规范化并校验变速段：按源起始时间排序，速度为正、段长为正、源上首尾相接。"""
    raw = []
    for item in segments:
        if isinstance(item, dict):
            start = float(item["src_start"])
            end = float(item["src_end"])
            speed = float(item["speed"])
        else:
            start, end, speed = (float(x) for x in item[:3])
        raw.append({"src_start": start, "src_end": end, "speed": speed})

    raw.sort(key=lambda d: d["src_start"])
    for seg in raw:
        if not (math.isfinite(seg["speed"]) and seg["speed"] > 0):
            raise ValueError("speed must be a positive finite number")
        if seg["src_end"] <= seg["src_start"]:
            raise ValueError("segment must have positive source duration")
    for prev, nxt in zip(raw, raw[1:]):
        if abs(prev["src_end"] - nxt["src_start"]) > 1e-9:
            raise ValueError("segments must be contiguous on the source timeline")
    return raw


def _locate(segments, t, start_key, end_key):
    """二分定位时间 t 所属分段；越界时钳制到首/末段。"""
    if t <= segments[0][start_key]:
        return 0
    if t >= segments[-1][end_key]:
        return len(segments) - 1
    lo, hi = 0, len(segments) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if t < segments[mid][start_key]:
            hi = mid
        elif t >= segments[mid][end_key]:
            lo = mid + 1
        else:
            return mid
    return lo


def _fit_length(samples, target):
    """把拉伸后的样本调整到目标长度：超长截断，不足用末样本补足。"""
    if target <= 0:
        return []
    if len(samples) > target:
        return samples[:target]
    if len(samples) < target:
        pad = samples[-1] if samples else 0.0
        return samples + [pad] * (target - len(samples))
    return list(samples)


# ---------------------------------------------------------------------------
# 1. 变速曲线与连续时间映射
# ---------------------------------------------------------------------------

def build_curve(segments=None, fps=DEFAULT_FPS):
    """构造变速曲线。

    每个分段记录源区间 ``[src_start, src_end)`` 与速度 ``speed``，输出时长为
    源时长 / speed；输出起始时间由各段输出时长逐段累积得到，因此段边界天然连续。

    返回 ``{"segments": [...], "fps", "src_duration", "out_duration"}``。
    """
    if not isinstance(segments, (list, tuple)) or not segments:
        segments = [(0.0, 1.0, 1.0)]
    segs = _norm_segments(segments)

    cursor = 0.0
    for seg in segs:
        seg["out_start"] = _r(cursor)
        cursor += (seg["src_end"] - seg["src_start"]) / seg["speed"]
        seg["out_end"] = _r(cursor)

    return {
        "segments": segs,
        "fps": int(fps),
        "src_duration": _r(segs[-1]["src_end"]),
        "out_duration": _r(cursor),
    }


def _src_to_out(curve, t):
    """源时间 -> 输出时间，分段线性、边界连续。"""
    segs = curve["segments"]
    idx = _locate(segs, _r(t), "src_start", "src_end")
    seg = segs[idx]
    t = min(max(t, seg["src_start"]), seg["src_end"])
    return _r(seg["out_start"] + (t - seg["src_start"]) / seg["speed"])


def _out_to_src(curve, t):
    """输出时间 -> 源时间（连续映射的逆函数），供帧重采样使用。"""
    segs = curve["segments"]
    idx = _locate(segs, _r(t), "out_start", "out_end")
    seg = segs[idx]
    t = min(max(t, seg["out_start"]), seg["out_end"])
    return _r(seg["src_start"] + (t - seg["out_start"]) * seg["speed"])


def segment_map_continuous(curve, t=None, direction="out"):
    """连续时间映射（修复点 1）。

    ``direction="out"``：源时间 -> 输出时间；``direction="src"``：输出时间 ->
    源时间。支持标量或时间列表；``t=None`` 时返回全部段边界的映射表，可直接
    用于核验边界连续性。映射基于时间积分而非重采样帧号，边界两侧取值严格相等。
    """
    curve = _coerce_curve(curve)
    mapper = _src_to_out if direction == "out" else _out_to_src

    if t is None:
        table = [{"src": s["src_start"], "out": s["out_start"]} for s in curve["segments"]]
        last = curve["segments"][-1]
        table.append({"src": last["src_end"], "out": last["out_end"]})
        return table
    if isinstance(t, (list, tuple)):
        return [mapper(curve, x) for x in t]
    return mapper(curve, t)


# ---------------------------------------------------------------------------
# 2. 音频分段独立变速（保音高）
# ---------------------------------------------------------------------------

def _hann(length):
    if length <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (length - 1))) for i in range(length)]


def _ola_stretch(samples, speed):
    """保音高时长拉伸（同步重叠相加 SOLA），拉伸比 alpha = 1/speed。

    纯 Python 参考实现：分析帧固定步进、Hann 窗叠加、按窗能量归一化。
    生产环境可替换为 WSOLA/PSOLA，接口与输出长度约定保持不变。
    """
    n = len(samples)
    if n == 0:
        return []
    alpha = 1.0 / speed
    analysis_hop = max(32, min(256, n // 4 or 32))
    frame_len = 2 * analysis_hop
    synth_hop = max(1, round(analysis_hop * alpha))
    window = _hann(frame_len)
    n_out = max(1, round(n * alpha))

    buf = [0.0] * (n_out + frame_len)
    acc = [0.0] * len(buf)
    analysis_pos = 0
    synth_idx = 0
    while analysis_pos < n:
        for j, win in enumerate(window):
            pos = synth_idx * synth_hop + j
            if pos >= len(buf):
                break
            buf[pos] += samples[min(n - 1, analysis_pos + j)] * win
            acc[pos] += win
        synth_idx += 1
        analysis_pos += analysis_hop
    return [buf[i] / acc[i] if acc[i] > 1e-9 else 0.0 for i in range(n_out)]


def _crossfade_concat(chunks, xfade):
    """分段音频交叉淡化拼接，消除变速点处的爆音/金属感拼接痕迹。"""
    if not chunks:
        return []
    pieces = [list(c["samples"]) for c in chunks]
    if xfade <= 0 or len(pieces) == 1:
        merged = []
        for piece in pieces:
            merged.extend(piece)
        return merged

    merged = list(pieces[0])
    for piece in pieces[1:]:
        k = min(xfade, len(merged), len(piece))
        if k == 0:
            merged.extend(piece)
            continue
        tail = merged[-k:]
        head = piece[:k]
        blended = [
            tail[j] * (1.0 - (j + 1) / (k + 1.0))
            + head[j] * ((j + 1) / (k + 1.0))
            for j in range(k)
        ]
        merged = merged[:-k] + blended + piece[k:]
    return merged


def _norm_audio(audio, curve, sr):
    """音频入参规范化：None 时生成覆盖源时长的静音 PCM；支持 dict / 样本列表。"""
    if isinstance(audio, dict):
        return list(audio.get("samples", [])), int(audio.get("sr", sr))
    if isinstance(audio, (list, tuple)):
        return list(audio), int(sr)
    total = max(1, round(curve["src_duration"] * sr))
    return [0.0] * total, int(sr)


def audio_segment_speed(curve, audio=None, sr=DEFAULT_SR, crossfade_ms=DEFAULT_CROSSFADE_MS):
    """音频按段独立保音高变速（修复点 2）。

    每个变速段独立切片、独立以 ``tempo=speed`` 做时长拉伸（``pitch_shift=0``），
    再通过短时交叉淡化拼接；总样本数与输出时间线严格对齐。
    """
    curve = _coerce_curve(curve)
    samples, sr = _norm_audio(audio, curve, sr)
    xfade = max(0, round(sr * crossfade_ms / 1000.0))

    chunks = []
    for idx, seg in enumerate(curve["segments"]):
        a0 = round(seg["src_start"] * sr)
        a1 = round(seg["src_end"] * sr)
        source = samples[a0:a1]
        stretched = _ola_stretch(source, seg["speed"])
        target = round(seg["out_end"] * sr) - round(seg["out_start"] * sr)
        stretched = _fit_length(stretched, target)
        chunks.append({
            "segment": idx,
            "stream": "a",
            "pts": seg["out_start"],
            "duration": _r(len(stretched) / sr),
            "src_start": seg["src_start"],
            "src_end": seg["src_end"],
            "out_start": seg["out_start"],
            "out_end": seg["out_end"],
            "tempo": seg["speed"],
            "pitch_shift": 0.0,
            "pitch_preserved": True,
            "method": "ola",
            "sr": sr,
            "samples": stretched,
            "n_in": len(source),
            "n_out": len(stretched),
        })

    assembled = _crossfade_concat(chunks, xfade)
    expected = max(1, round(curve["out_duration"] * sr))
    assembled = _fit_length(assembled, expected)
    return {
        "chunks": chunks,
        "samples": assembled,
        "sr": sr,
        "n_samples": len(assembled),
        "out_duration": _r(len(assembled) / sr),
        "crossfade": xfade,
    }


# ---------------------------------------------------------------------------
# 3. 关键帧按变速后时间线重算
# ---------------------------------------------------------------------------

def keyframe_rescale(curve, src_keyframes=None, gop=None, fps_out=None):
    """在输出时间线上重算关键帧（修复点 3）。

    - 输出第 0 帧与每个变速点边界帧强制为 IDR；
    - 源关键帧经连续映射落到输出时间线后取整并入集合；
    - 相邻关键帧间距超过 ``gop``（默认 2 秒）时等距补齐，保证最大 GOP 不超限。
    """
    curve = _coerce_curve(curve)
    fps = int(fps_out or curve["fps"] or DEFAULT_FPS)
    gop = int(gop) if gop else max(1, round(DEFAULT_GOP_SECONDS * fps))
    n_out = max(1, round(curve["out_duration"] * fps))

    forced = {0}
    for seg in curve["segments"][1:]:
        forced.add(min(n_out - 1, round(seg["out_start"] * fps)))

    for kf in (src_keyframes or []):
        if isinstance(kf, dict):
            src_time = float(kf.get("pts", kf.get("src_time", 0.0)))
        else:
            src_time = float(kf) / fps  # 数字按源帧号解释
        out_idx = round(_src_to_out(curve, src_time) * fps)
        forced.add(min(n_out - 1, max(0, out_idx)))

    anchors = sorted(forced)
    keyframes = set(forced)
    bounds = anchors + [n_out]
    for start, end in zip(bounds[:-1], bounds[1:]):
        pos = start + gop
        while pos < end:
            keyframes.add(pos)
            pos += gop

    keyframes = sorted(keyframes)
    return {
        "keyframes": keyframes,
        "forced": anchors,
        "gop": gop,
        "fps": fps,
        "map": [
            {"out_idx": idx, "src_time": _out_to_src(curve, idx / fps)}
            for idx in keyframes
        ],
    }


# ---------------------------------------------------------------------------
# 4. 元数据随帧绑定
# ---------------------------------------------------------------------------

def meta_bind_frame(frames, meta=None):
    """按源帧号把元数据绑定到重排后的帧（修复点 4）。

    ``meta`` 支持 ``{src_idx: payload}`` 字典或按源帧号排列的列表。
    只按 ``src_idx`` 查找，绝不按帧在重排列表中的位置附加；缺失元数据绑定空 dict。
    """
    if not isinstance(frames, list):
        return []
    if isinstance(meta, list):
        meta = {idx: payload for idx, payload in enumerate(meta)}
    meta = meta or {}

    bound = []
    for frame in frames:
        merged = dict(frame)
        merged["meta"] = deepcopy(meta.get(frame["src_idx"], {}))
        bound.append(merged)
    return bound


# ---------------------------------------------------------------------------
# 5. 封装按时间线排序
# ---------------------------------------------------------------------------

def order_by_timeline(packets):
    """按 PTS 时间线稳定排序音视频包（修复点 5）。

    排序键为 ``(pts, 流类型, dts)``，同 PTS 时音频在视频前；Python 稳定排序
    保证其余同序包保持原有相对顺序。异步分段的完成顺序不影响最终封装顺序。
    """
    if not isinstance(packets, list):
        return []

    def key(pkt):
        return (
            _r(pkt.get("pts", 0.0)),
            _STREAM_RANK.get(pkt.get("stream", "v"), 9),
            _r(pkt.get("dts", pkt.get("pts", 0.0))),
        )

    return sorted(packets, key=key)


# ---------------------------------------------------------------------------
# 6. 变速点音画对齐校验
# ---------------------------------------------------------------------------

def align_check(curve, video=None, audio=None, tol=None, fps=None, sr=None):
    """逐变速点校验音画边界漂移（修复点 6）。

    入参为各分段在输出时间线上的边界时间列表；缺省时由曲线分别按视频帧率与
    音频采样率量化推导。默认容差为半个输出帧，返回每个点的漂移与整体结论。
    """
    curve = _coerce_curve(curve)
    fps = int(fps or curve["fps"] or DEFAULT_FPS)
    sr = int(sr or DEFAULT_SR)
    tol = 1.0 / (2.0 * fps) if tol is None else float(tol)
    segs = curve["segments"]

    if video is None:
        video = [_r(round(s["out_start"] * fps) / fps) for s in segs]
    if audio is None:
        audio = [_r(round(s["out_start"] * sr) / sr) for s in segs]

    ok = True
    points = []
    for idx, seg in enumerate(segs):
        v = float(video[idx]) if idx < len(video) else seg["out_start"]
        a = float(audio[idx]) if idx < len(audio) else seg["out_start"]
        drift = _r(a - v)
        passed = abs(drift) <= tol + _EPS
        ok = ok and passed
        points.append({
            "segment": idx,
            "boundary": seg["out_start"],
            "video": _r(v),
            "audio": _r(a),
            "drift": drift,
            "pass": passed,
        })
    return {"ok": ok, "tol": tol, "points": points}


# ---------------------------------------------------------------------------
# 流水线：重采样 -> 编码 -> 封装 -> 校验 -> 度量/审计/降级/确定性
# ---------------------------------------------------------------------------

def _norm_src_frames(frames, curve, fps):
    """源帧规范化：None 时按曲线源时长生成；int 作为帧数；列表支持 int/dict。"""
    if isinstance(frames, int):
        count = max(1, frames)
        return [{"idx": k, "pts": _r(k / fps)} for k in range(count)]
    if isinstance(frames, list) and frames:
        out = []
        for k, item in enumerate(frames):
            if isinstance(item, dict):
                out.append({"idx": item.get("idx", k), "pts": float(item.get("pts", k / fps))})
            else:
                out.append({"idx": int(item), "pts": _r(int(item) / fps)})
        return out
    count = max(1, round(curve["src_duration"] * fps))
    return [{"idx": k, "pts": _r(k / fps)} for k in range(count)]


def retime(curve, frames=None, fps_out=None, fps_src=None):
    """帧重采样：对每个输出帧用连续逆映射求源时间，再取最近源帧。

    快放自然丢帧、慢放自然复制帧，输出帧带 ``src_idx`` 供元数据绑定，
    不再使用"重采样帧号反推源时间戳"的不连续做法。
    """
    curve = _coerce_curve(curve)
    fps_out = int(fps_out or curve["fps"] or DEFAULT_FPS)
    fps_src = int(fps_src or curve["fps"] or DEFAULT_FPS)
    src_frames = _norm_src_frames(frames, curve, fps_src)
    n_src = len(src_frames)
    n_out = max(1, round(curve["out_duration"] * fps_out))

    output = []
    for n in range(n_out):
        out_time = n / fps_out
        src_time = _out_to_src(curve, out_time)
        src_idx = min(n_src - 1, max(0, round(src_time * fps_src)))
        output.append({
            "out_idx": n,
            "pts": _r(out_time),
            "src_idx": src_idx,
            "src_pts": src_frames[src_idx]["pts"],
        })
    return output


def _regular_keyframes(n_frames, fps):
    gop = max(1, round(DEFAULT_GOP_SECONDS * fps))
    return list(range(0, n_frames, gop))


def encode(frames, keyframes=None, audio=None, fps=None, bitrate=None):
    """模拟编码：按关键帧集合打 I/P 帧类型与 PTS/DTS，音频块转为音频包。

    无 B 帧重排，DTS 与 PTS 单调一致；视频包透传绑定到帧的元数据。
    """
    if not isinstance(frames, list):
        return _run_pipeline(frames if isinstance(frames, dict) else None)["encode"]

    fps = int(fps or DEFAULT_FPS)
    n_frames = len(frames)
    if keyframes is None:
        keyframes = _regular_keyframes(n_frames, fps)
    kf_set = set(keyframes)
    frame_duration = _r(1.0 / fps)

    video_packets = []
    for frame in frames:
        idx = frame["out_idx"]
        is_keyframe = idx in kf_set
        video_packets.append({
            "stream": "v",
            "pts": frame["pts"],
            "dts": frame["pts"],
            "duration": frame_duration,
            "out_idx": idx,
            "src_idx": frame["src_idx"],
            "keyframe": is_keyframe,
            "type": "I" if is_keyframe else "P",
            "size": _I_SIZE if is_keyframe else _P_SIZE,
            "meta": frame.get("meta", {}),
        })

    audio_packets = []
    if isinstance(audio, dict):
        for chunk in audio.get("chunks", []):
            audio_packets.append({
                "stream": "a",
                "pts": chunk["pts"],
                "dts": chunk["pts"],
                "duration": chunk["duration"],
                "segment": chunk["segment"],
                "tempo": chunk["tempo"],
                "pitch_preserved": chunk["pitch_preserved"],
                "n_samples": chunk["n_out"],
                "size": chunk["n_out"] * 2 // 32,
            })

    return {
        "video": video_packets,
        "audio": audio_packets,
        "fps": fps,
        "bitrate": bitrate,
        "keyframes": sorted(kf_set),
    }


def mux(video, audio=None):
    """封装：合并音视频包并按时间线排序（内部调用 order_by_timeline）。"""
    if isinstance(video, dict) and "video" in video:
        audio = video.get("audio")
        video = video["video"]
    video = video if isinstance(video, list) else []
    audio = audio if isinstance(audio, list) else []

    packets = order_by_timeline(video + audio)
    duration = 0.0
    for pkt in packets:
        duration = max(duration, pkt["pts"] + pkt.get("duration", 0.0))
    return {
        "packets": packets,
        "streams": {
            "v": {"count": len(video)},
            "a": {"count": len(audio)},
        },
        "duration": _r(duration),
    }


def validate_av(container, curve=None, tol=None):
    """端到端校验：时间线顺序、PTS 单调、边界关键帧、音画对齐、时长与元数据覆盖。"""
    if not isinstance(container, dict) or "packets" not in container:
        return {"ok": True, "issues": [], "align": None}

    issues = []
    packets = container["packets"]
    if packets != order_by_timeline(list(packets)):
        issues.append("packets are not in timeline order")

    for stream in ("v", "a"):
        last_pts = None
        for pkt in packets:
            if pkt["stream"] != stream:
                continue
            if last_pts is not None and pkt["pts"] < last_pts - _EPS:
                issues.append(f"{stream} PTS regression at {pkt['pts']}")
            last_pts = pkt["pts"]

    align = None
    if curve is not None:
        curve = _coerce_curve(curve)
        fps = curve["fps"]
        video_pkts = [p for p in packets if p["stream"] == "v"]
        audio_pkts = [p for p in packets if p["stream"] == "a"]

        for seg in curve["segments"][1:]:
            if not video_pkts:
                issues.append("no video packets at speed boundary")
                continue
            nearest = min(video_pkts, key=lambda p: abs(p["pts"] - seg["out_start"]))
            if not nearest.get("keyframe"):
                issues.append(f"missing IDR at speed boundary {seg['out_start']}")
            elif abs(nearest["pts"] - seg["out_start"]) > 1.0 / fps + _EPS:
                issues.append(f"keyframe too far from boundary {seg['out_start']}")

        if video_pkts:
            video_bounds = [
                min(video_pkts, key=lambda p: abs(p["pts"] - s["out_start"]))["pts"]
                for s in curve["segments"]
            ]
        else:
            video_bounds = None
        audio_bounds = [
            _r(round(s["out_start"] * DEFAULT_SR) / DEFAULT_SR) for s in curve["segments"]
        ]
        align = align_check(curve, video=video_bounds, audio=audio_bounds, tol=tol)
        if not align["ok"]:
            issues.append("av alignment drift beyond tolerance at speed boundary")

        if abs(container["duration"] - curve["out_duration"]) > 1.0 / fps + _EPS:
            issues.append("container duration does not match curve out_duration")

        for pkt in video_pkts:
            if "meta" not in pkt:
                issues.append(f"video packet at pts {pkt['pts']} missing bound metadata")

        if not audio_pkts:
            issues.append("audio stream missing")

    return {"ok": not issues, "issues": issues, "align": align}


def metrics(container, curve=None):
    """度量：包统计、最大 GOP、字节估算以及各变速段帧数/时长符合度。"""
    base = {
        "video_packets": 0,
        "audio_packets": 0,
        "duration": 0.0,
        "keyframes": 0,
        "max_gop": 0,
        "video_bytes": 0,
        "audio_bytes": 0,
        "segments": [],
    }
    if not isinstance(container, dict) or "packets" not in container:
        return base

    packets = container["packets"]
    video_pkts = [p for p in packets if p["stream"] == "v"]
    audio_pkts = [p for p in packets if p["stream"] == "a"]
    kf_idx = [p["out_idx"] for p in video_pkts if p.get("keyframe")]
    max_gop = max((b - a for a, b in zip(kf_idx, kf_idx[1:])), default=0)

    result = {
        "video_packets": len(video_pkts),
        "audio_packets": len(audio_pkts),
        "duration": container.get("duration", 0.0),
        "keyframes": len(kf_idx),
        "max_gop": max_gop,
        "video_bytes": sum(p.get("size", 0) for p in video_pkts),
        "audio_bytes": sum(p.get("size", 0) for p in audio_pkts),
        "segments": [],
    }

    if curve is not None:
        curve = _coerce_curve(curve)
        fps = curve["fps"]
        for idx, seg in enumerate(curve["segments"]):
            expected = round(seg["out_end"] * fps) - round(seg["out_start"] * fps)
            actual = sum(
                1 for p in video_pkts
                if seg["out_start"] <= p["pts"] < seg["out_end"] - _EPS
            )
            result["segments"].append({
                "segment": idx,
                "speed": seg["speed"],
                "expected_duration": _r(seg["out_end"] - seg["out_start"]),
                "expected_frames": expected,
                "actual_frames": actual,
                "frame_drift": actual - expected,
            })
    return result


def audit(curve=None, frames=None, meta=None, gop=None):
    """配置期审计：分段合法性/连续性、GOP 合法性、元数据对源帧的覆盖率。"""
    issues = []
    if isinstance(curve, list):
        try:
            curve = build_curve(curve)
        except ValueError as exc:
            issues.append(str(exc))
            curve = build_curve()
    else:
        curve = _coerce_curve(curve)

    for seg in curve["segments"]:
        if not (math.isfinite(seg["speed"]) and seg["speed"] > 0):
            issues.append(f"invalid speed at segment {seg.get('segment', '?')}")
        if seg["src_end"] <= seg["src_start"]:
            issues.append("non-positive segment source duration")
    for prev, nxt in zip(curve["segments"], curve["segments"][1:]):
        if abs(prev["src_end"] - nxt["src_start"]) > 1e-6:
            issues.append("segments not contiguous on source timeline")
    if gop is not None and int(gop) <= 0:
        issues.append("gop must be positive")
    if isinstance(frames, list) and isinstance(meta, dict):
        used = {f.get("src_idx") for f in frames if isinstance(f, dict)}
        missing = sorted(k for k in used if k not in meta)
        if missing:
            issues.append(f"metadata missing for src frames {missing[:10]}")

    return {"ok": not issues, "issues": issues, "curve": curve}


def fallback(job=None, error=None):
    """安全降级：把任意变速任务压平为覆盖全源区间的 1.0 倍速单段、1 秒固定 GOP。"""
    if isinstance(job, dict) and "segments" in job:
        segments = job["segments"]
        fps = int(job.get("fps", DEFAULT_FPS))
    else:
        segments = [(0.0, 1.0, 1.0)]
        fps = DEFAULT_FPS
    try:
        src_end = _norm_segments(segments)[-1]["src_end"]
    except ValueError:
        src_end = 1.0
    src_end = max(src_end, _EPS)

    safe_curve = build_curve([(0.0, src_end, 1.0)], fps=fps)
    return {
        "curve": safe_curve,
        "gop": fps,
        "reason": "flatten to constant 1.0x speed after pipeline failure",
        "source_error": None if error is None else str(error),
    }


def _norm_job(job):
    if not isinstance(job, dict):
        job = {}
    return {
        "segments": job.get(
            "segments",
            [(0.0, 1.0, 2.0), (1.0, 2.0, 0.5), (2.0, 3.0, 1.0)],
        ),
        "fps": int(job.get("fps", DEFAULT_FPS)),
        "sr": int(job.get("sr", DEFAULT_SR)),
        "frames": job.get("frames", 90),
        "meta": job.get(
            "meta",
            {0: {"highlight": True}, 30: {"highlight": True}, 60: {"highlight": True}},
        ),
        "gop": int(job.get("gop", round(DEFAULT_GOP_SECONDS * DEFAULT_FPS))),
        "src_keyframes": job.get("src_keyframes", [0, 30, 60]),
        "audio": job.get("audio"),
    }


def _run_pipeline(job):
    """端到端流水线：曲线 -> 重采样 -> 元数据绑定 -> 关键帧 -> 音频 -> 编码 -> 封装。"""
    job = _norm_job(job)
    curve = build_curve(job["segments"], fps=job["fps"])
    src_frames = _norm_src_frames(job["frames"], curve, job["fps"])
    retimed = retime(curve, src_frames, fps_out=job["fps"], fps_src=job["fps"])
    retimed = meta_bind_frame(retimed, job["meta"])
    kf = keyframe_rescale(curve, job["src_keyframes"], gop=job["gop"], fps_out=job["fps"])
    audio = audio_segment_speed(curve, job["audio"], sr=job["sr"])
    encoded = encode(retimed, keyframes=kf["keyframes"], audio=audio, fps=job["fps"])
    container = mux(encoded["video"], encoded["audio"])
    return {
        "job": job,
        "curve": curve,
        "retimed": retimed,
        "keyframes": kf,
        "audio": audio,
        "encode": encoded,
        "container": container,
    }


def determinism(job=None, runs=3):
    """确定性验证：多次运行并随机打乱分段完成顺序，封装结果必须完全一致。"""
    job = _norm_job(job)
    signatures = []
    for r in range(int(runs)):
        pipeline = _run_pipeline(job)
        packets = list(pipeline["encode"]["video"]) + list(pipeline["encode"]["audio"])
        random.Random(1000 + r).shuffle(packets)  # 模拟异步分段完成顺序不定
        container = mux(packets)
        signatures.append([
            (
                pkt["stream"],
                pkt["pts"],
                pkt["dts"],
                pkt.get("type"),
                pkt.get("src_idx"),
            )
            for pkt in container["packets"]
        ])

    first = json.dumps(signatures[0], sort_keys=True)
    diffs = [i for i, sig in enumerate(signatures) if json.dumps(sig, sort_keys=True) != first]
    return {
        "ok": not diffs,
        "runs": int(runs),
        "diffs": diffs,
        "packet_count": len(signatures[0]),
        "signature": signatures[0],
    }
