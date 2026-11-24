import pytest
import media.speedramp_av_sync as mod

CASES = [
    ("segment_map_continuous", (1,)),
    ("audio_segment_speed", (1,)),
    ("keyframe_rescale", (1,)),
    ("meta_bind_frame", (1,)),
    ("order_by_timeline", (1,)),
    ("align_check", (1,)),
    ("build_curve", (1,)),
    ("retime", (1,)),
    ("encode", (1,)),
    ("mux", (1,)),
    ("validate_av", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
