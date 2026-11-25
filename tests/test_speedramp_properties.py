"""六个修复语义的属性验证（与题目契约测试相互独立）。"""

import random

import media.speedramp_av_sync as mod


SEGMENTS = [(0.0, 1.0, 2.0), (1.0, 2.0, 0.5), (2.0, 3.0, 1.0)]
FPS = 30
SR = 48000
GOP = 60


def _job_pipeline():
    curve = mod.build_curve(SEGMENTS, fps=FPS)
    src_frames = [{"idx": k, "pts": round(k / FPS, 9)} for k in range(3 * FPS)]
    retimed = mod.retime(curve, src_frames, fps_out=FPS, fps_src=FPS)
    meta = {i: {"highlight": True} for i in (0, 30, 60)}
    retimed = mod.meta_bind_frame(retimed, meta)
    kf = mod.keyframe_rescale(curve, src_keyframes=[0, 30, 60], gop=GOP, fps_out=FPS)
    audio = mod.audio_segment_speed(curve, sr=SR)
    encoded = mod.encode(retimed, keyframes=kf["keyframes"], audio=audio, fps=FPS)
    container = mod.mux(encoded["video"], encoded["audio"])
    return curve, retimed, kf, audio, encoded, container


def test_boundary_mapping_is_continuous():
    """修复 1：段边界两侧映射严格相等，无跳帧。"""
    curve = mod.build_curve(SEGMENTS, fps=FPS)
    for seg in curve["segments"][1:]:
        b = seg["src_start"]
        left = mod.segment_map_continuous(curve, b - 1e-12, direction="out")
        right = mod.segment_map_continuous(curve, b + 1e-12, direction="out")
        assert abs(left - seg["out_start"]) < 1e-6
        assert abs(right - seg["out_start"]) < 1e-6
        # 正反映射互逆
        src = mod.segment_map_continuous(curve, seg["out_start"], direction="src")
        assert abs(src - b) < 1e-6
    table = mod.segment_map_continuous(curve)
    assert table[1]["src"] == curve["segments"][1]["src_start"]
    assert table[1]["out"] == curve["segments"][1]["out_start"]
    # 输出时长：1/2 + 1/0.5 + 1/1 = 3.5
    assert abs(curve["out_duration"] - 3.5) < 1e-9


def test_audio_independent_segment_speed_with_pitch_preserved():
    """修复 2：分段独立拉伸、保音高、段数与曲线一致、总长与输出时间线对齐。"""
    curve = mod.build_curve(SEGMENTS, fps=FPS)
    result = mod.audio_segment_speed(curve, sr=SR)
    assert len(result["chunks"]) == len(curve["segments"])
    for chunk, seg in zip(result["chunks"], curve["segments"]):
        assert chunk["tempo"] == seg["speed"]
        assert chunk["pitch_preserved"] is True
        assert chunk["pitch_shift"] == 0.0
        assert abs(chunk["duration"] - (seg["out_end"] - seg["out_start"])) < 2.0 / SR
    assert abs(result["n_samples"] / SR - curve["out_duration"]) <= 1.0 / SR


def test_keyframes_recomputed_on_output_timeline():
    """修复 3：边界强制关键帧且最大 GOP 不超限。"""
    curve = mod.build_curve(SEGMENTS, fps=FPS)
    kf = mod.keyframe_rescale(curve, src_keyframes=[0, 30, 60], gop=GOP, fps_out=FPS)
    keyframes = kf["keyframes"]
    assert keyframes[0] == 0
    for seg in curve["segments"][1:]:
        assert round(seg["out_start"] * FPS) in keyframes
    gaps = [b - a for a, b in zip(keyframes, keyframes[1:])]
    assert max(gaps) <= GOP


def test_metadata_bound_to_source_frame_identity():
    """修复 4：元数据随 src_idx 绑定，与重排后位置无关。"""
    curve = mod.build_curve(SEGMENTS, fps=FPS)
    retimed = mod.retime(curve, fps_out=FPS, fps_src=FPS)
    meta = {i: {"highlight": True} for i in (0, 30, 60)}
    bound = mod.meta_bind_frame(retimed, meta)
    for frame in bound:
        expect = meta.get(frame["src_idx"], {})
        assert frame["meta"] == expect
    # 打乱顺序后绑定结果不变
    shuffled = list(reversed(retimed))
    bound_shuffled = mod.meta_bind_frame(shuffled, meta)
    for frame in bound_shuffled:
        assert frame["meta"] == meta.get(frame["src_idx"], {})


def test_timeline_ordering_independent_of_completion_order():
    """修复 5：任意打乱输入，封装顺序都等于 PTS 时间线顺序。"""
    _, _, _, _, encoded, container = _job_pipeline()
    packets = encoded["video"] + encoded["audio"]
    rng = random.Random(42)
    for _ in range(5):
        shuffled = list(packets)
        rng.shuffle(shuffled)
        ordered = mod.order_by_timeline(shuffled)
        pts = [p["pts"] for p in ordered]
        assert pts == sorted(pts)
        assert [(p["stream"], p["pts"], p["dts"]) for p in ordered] == [
            (p["stream"], p["pts"], p["dts"]) for p in container["packets"]
        ]


def test_av_alignment_check_at_each_speed_boundary():
    """修复 6：每个变速点独立出漂移结论，超差判定失败。"""
    curve = mod.build_curve(SEGMENTS, fps=FPS)
    result = mod.align_check(curve, tol=1.0 / FPS)
    assert result["ok"] is True
    assert len(result["points"]) == len(curve["segments"])
    bad = mod.align_check(curve, audio=[0.0, 1.0, 3.05], tol=1.0 / FPS)
    assert bad["ok"] is False
    assert bad["points"][2]["pass"] is False


def test_validate_metrics_determinism_and_fallback():
    curve, _, kf, _, _, container = _job_pipeline()
    report = mod.validate_av(container, curve)
    assert report["ok"] is True, report["issues"]
    assert report["align"]["ok"] is True

    stats = mod.metrics(container, curve)
    assert stats["max_gop"] <= GOP
    assert stats["video_packets"] == round(curve["out_duration"] * FPS)
    assert all(abs(seg["frame_drift"]) <= 1 for seg in stats["segments"])

    audit = mod.audit(curve)
    assert audit["ok"] is True
    bad_audit = mod.audit([(0.0, 1.0, 2.0), (1.5, 2.0, 1.0)])
    assert bad_audit["ok"] is False

    fb = mod.fallback({"segments": SEGMENTS, "fps": FPS}, error="boom")
    assert len(fb["curve"]["segments"]) == 1
    assert fb["curve"]["segments"][0]["speed"] == 1.0

    det = mod.determinism(runs=3)
    assert det["ok"] is True
    assert det["packet_count"] > 0
