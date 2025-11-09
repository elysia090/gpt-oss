from __future__ import annotations

import math

from tests.gpt_oss import test_sera as sera_test_module

np = sera_test_module.np
BridgeConfig = sera_test_module.BridgeConfig
BridgeState = sera_test_module.BridgeState
BridgeGuardRecord = sera_test_module._SERA_MODULE.BridgeGuardRecord


def test_guard_record_check_enforces_full_conditions() -> None:
    record = BridgeGuardRecord(
        margin=1.2,
        eps_row=0.1,
        leg_bound_in=0.2,
        leg_bound_out=0.25,
        competitor_bounds=(0.3, 0.4),
    )

    assert record.check()


def test_guard_record_check_rejects_inconsistent_values() -> None:
    invalid_records = [
        BridgeGuardRecord(0.0, 0.1, 0.1, 0.1, (0.05,)),
        BridgeGuardRecord(math.nan, 0.1, 0.1, 0.1, (0.05,)),
        BridgeGuardRecord(1.0, -0.1, 0.1, 0.1, (0.05,)),
        BridgeGuardRecord(1.0, 0.1, -0.1, 0.1, (0.05,)),
        BridgeGuardRecord(1.0, 0.1, 0.1, float("inf"), (0.05,)),
        BridgeGuardRecord(1.0, 0.1, 0.1, 0.1, (0.05, -0.1)),
        BridgeGuardRecord(1.0, 0.1, 0.1, 0.1, (float("inf"),)),
        BridgeGuardRecord(1.0, 0.1, 0.6, 0.5, (0.05,)),
        BridgeGuardRecord(1.0, 0.1, 0.1, 0.1, (0.6,)),
    ]

    for record in invalid_records:
        assert not record.check()


def test_bridge_read_requires_margin_consistency() -> None:
    config = BridgeConfig(hub_window=1, candidate_bound=4, projection=((1.0,),))
    bridge = BridgeState(config)

    ctx = 3
    token = 7
    value = np.array([5.0], dtype=float)

    for hub in (0, 1):
        bridge.set_hub(ctx, hub)
        bridge.set_hub(token, hub)

    bridge.set_qdout(ctx, 0, 0.4)
    bridge.set_qdin(0, token, 0.6)
    bridge.set_qdout(ctx, 1, 1.5)
    bridge.set_qdin(1, token, 3.0)

    bridge.promote(
        ctx,
        token,
        value,
        guard_margin=2.0,
        guard_eps_row=0.1,
        leg_bound_in=0.2,
        leg_bound_out=0.2,
        competitor_bounds=(0.3,),
    )

    bridge_value, guard_ok, _ = bridge.read(ctx, token)
    assert guard_ok
    np.testing.assert_allclose(bridge_value, value)

    bridge.promote(
        ctx,
        token,
        value,
        guard_margin=3.6,
        guard_eps_row=0.1,
        leg_bound_in=0.2,
        leg_bound_out=0.2,
        competitor_bounds=(0.3,),
    )

    bridge_value, guard_ok, _ = bridge.read(ctx, token)
    assert not guard_ok
    np.testing.assert_allclose(bridge_value, np.array([1.0], dtype=float))


def test_bridge_last_read_info_tracks_route_projection() -> None:
    config = BridgeConfig(hub_window=1, candidate_bound=4, projection=((1.0,),))
    bridge = BridgeState(config)

    ctx = 11
    token = 23

    bridge.set_hub(ctx, 0)
    bridge.set_hub(token, 0)
    for hub in (0, 1):
        bridge.set_hub(ctx, hub)
        bridge.set_hub(token, hub)

    bridge.set_qdout(ctx, 0, 0.2)
    bridge.set_qdin(0, token, 0.3)
    bridge.set_qdout(ctx, 1, 1.1)
    bridge.set_qdin(1, token, 1.4)

    bridge.promote(
        ctx,
        token,
        np.array([4.0], dtype=float),
        guard_margin=1.6,
        guard_eps_row=0.05,
        leg_bound_in=0.1,
        leg_bound_out=0.1,
        competitor_bounds=(0.2,),
    )

    value, guard_ok, _ = bridge.read(ctx, token)
    assert guard_ok
    np.testing.assert_allclose(value, np.array([4.0], dtype=float))
    info = bridge.last_read_info
    assert info is not None
    assert info.best_hub == 0
    assert info.guard_ok is True
    assert info.dictionary_hit is True
    assert info.projected == (0.5,)

    bridge.promote(
        ctx,
        token,
        np.array([4.0], dtype=float),
        guard_margin=4.0,
        guard_eps_row=0.05,
        leg_bound_in=0.1,
        leg_bound_out=0.1,
        competitor_bounds=(0.2,),
    )

    fallback, guard_ok, _ = bridge.read(ctx, token)
    assert not guard_ok
    np.testing.assert_allclose(fallback, np.array([0.5], dtype=float))
    info = bridge.last_read_info
    assert info is not None
    assert info.guard_ok is False
    assert info.dictionary_hit is True
    assert info.projected == (0.5,)
