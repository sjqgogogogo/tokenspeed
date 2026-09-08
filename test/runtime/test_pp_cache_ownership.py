# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU coverage of PP cache geometry, producer barriers and PD routes."""

from collections import Counter
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
    CacheLayerOwnership,
    pipeline_cache_ownership,
    target_stage_windows,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_field_layer_id,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
    build_cache_fields_by_producer_step,
)
from tokenspeed.runtime.pd.transfer_plan import build_pipeline_transfer_plan


def _k3_contract(num_draft_layers: int):
    # K3's topology at smaller per-layer widths: 24 MLA, 69 KDA, 5 draft
    # layers. KDA groups retain the actual shared-plane aliasing structure.
    full_layers = (*range(3, 92, 4), 92)
    state_layers = [layer for layer in range(93) if layer not in full_layers]
    full_spec = CacheGroupSpec(
        group_id="full_attention",
        retention="full_history",
        rows_per_page=128,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        family="history",
        transfer_policy="full_suffix",
        checkpoint_granularity=None,
    )
    full_fields = tuple(
        CacheFieldSpec(
            f"layer.{layer}.latent_kv",
            f"slot.{slot}",
            (128, 1, 8),
            "uint8",
            exact_page_stride=True,
            page_stride_alignment_bytes=1,
        )
        for slot, layer in enumerate((*full_layers, *range(93, 93 + num_draft_layers)))
    )
    groups = [(full_spec, full_fields)]
    for group_index in range(3):
        state_spec = CacheGroupSpec(
            group_id=f"linear_attention_{group_index}",
            retention="full_history",
            rows_per_page=None,
            entry_stride_tokens=None,
            sliding_window_tokens=None,
            family="state",
            transfer_policy="latest_snapshot",
            checkpoint_granularity=128,
        )
        fields = []
        for slot, layer in enumerate(
            state_layers[group_index * 23 : (group_index + 1) * 23]
        ):
            for suffix, shape, dtype in (
                ("conv_state", (24, 3), "bfloat16"),
                ("recurrent_state", (1, 8, 8), "float32"),
            ):
                fields.append(
                    CacheFieldSpec(
                        f"layer.{layer}.{suffix}",
                        f"slot.{slot}",
                        shape,
                        dtype,
                        exact_page_stride=False,
                        page_stride_alignment_bytes=1,
                    )
                )
        groups.append((state_spec, tuple(fields)))
    layout = pack(
        groups,
        prefix_granularity=128,
        cache_blocks_per_lcm_block={
            spec.group_id: 1 if spec.family == "history" else 2 for spec, _ in groups
        },
        alignment=1,
        max_padding_fraction=float("inf"),
    )
    plan = layout.bind(num_lcm_blocks=3)
    target = SimpleNamespace(
        num_attention_layers=93,
        hf_text_config=SimpleNamespace(
            linear_attn_config={"num_heads": 8, "head_dim": 8}
        ),
    )
    draft = (
        SimpleNamespace(
            num_attention_layers=num_draft_layers, hf_text_config=SimpleNamespace()
        )
        if num_draft_layers
        else None
    )
    return CacheTransferContract(
        plan=plan,
        group_specs=tuple(spec for spec, _ in groups),
        transfer_schema=build_cache_transfer_schema(
            plan,
            model_config=target,
            draft_model_config=draft,
        ),
    )


@pytest.mark.parametrize("partition", [None, (20, 26, 26, 21)])
def test_draft_layers_never_change_target_stage_cuts(partition):
    owners = pipeline_cache_ownership(93, 5, 4, partition)
    targets = target_stage_windows(93, 4, partition)
    assert [owner.target_window for owner in owners] == targets
    assert [owner.owns_draft for owner in owners] == [False, False, False, True]
    assert owners[-1].resident_window == (targets[-1][0], 98)
    layers = Counter(
        layer for owner in owners for layer in range(*owner.resident_window)
    )
    assert layers == Counter(range(98))


def test_physical_stage_plans_keep_one_logical_geometry_and_bounds():
    contract = _k3_contract(5)
    plan = contract.plan
    assert len(plan.planes) == 29
    owners = pipeline_cache_ownership(93, 5, 4, None)
    fields = Counter()
    for owner in owners:
        resident = plan.narrow_to_layers(*owner.resident_window)
        assert resident.groups == plan.groups
        assert resident.prefix_granularity == plan.prefix_granularity
        assert resident.num_lcm_blocks == plan.num_lcm_blocks
        assert resident.lcm_block_bytes == plan.lcm_block_bytes
        assert resident.arena_bytes < plan.arena_bytes
        draft_planes = {f"slot.{slot}" for slot in range(24, 29)}
        resident_planes = {plane.plane_id for plane in resident.planes}
        assert resident_planes & draft_planes == (
            draft_planes if owner.owns_draft else set()
        )
        for field in resident.fields:
            fields[field.field_id] += 1
            last_page = resident.group(field.group_id).page_count - 1
            for page in (0, last_page):
                offset = resident.field_page_byte_offset(field.field_id, page)
                assert 0 <= offset <= resident.arena_bytes - field.payload_bytes
    assert fields == Counter(field.field_id for field in plan.fields)


@pytest.mark.parametrize("partition", [None, (20, 26, 26, 21)])
def test_pp4_tp8_routes_cover_kda_mla_and_draft_on_all_decode_replicas(partition):
    contract = _k3_contract(5)
    owners = pipeline_cache_ownership(93, 5, 4, partition)
    for global_rank in range(32):
        tp_rank = global_rank % 8
        route, dummy = build_pipeline_transfer_plan(
            prefill_tp_size=8,
            decode_tp_size=8,
            decode_tp_rank=tp_rank,
            prefill_layout=contract,
            decode_layout=contract,
            num_target_layers=93,
            pp_size=4,
            pp_layer_partition=partition,
        )
        assert dummy == ()
        assert set(route.fragments_by_prefill_rank) == {
            tp_rank + 8 * stage for stage in range(4)
        }
        all_fields = Counter()
        for stage, owner in enumerate(owners):
            fragments = route.fragments_by_prefill_rank[tp_rank + 8 * stage]
            expected = {
                field.field_id
                for field in contract.plan.fields
                if owner.resident_window[0]
                <= cache_field_layer_id(field.field_id)
                < owner.resident_window[1]
            }
            assert {fragment.field_id for fragment in fragments} == expected
            all_fields.update({fragment.field_id for fragment in fragments})
            # KDA convolution has Q/K/V fragments. Their byte intervals must
            # cover each rank-local field once, despite sharing its field id.
            by_field = {}
            for fragment in fragments:
                spans = by_field.setdefault(fragment.field_id, [])
                spans.extend(
                    (
                        fragment.dst_byte_offset + row * fragment.dst_row_stride_bytes,
                        fragment.dst_byte_offset
                        + row * fragment.dst_row_stride_bytes
                        + fragment.bytes_per_row,
                    )
                    for row in range(fragment.rows_per_page)
                )
            for field_id, spans in by_field.items():
                spans.sort()
                assert spans[0][0] == 0
                assert spans[-1][1] == contract.plan.field(field_id).payload_bytes
                assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        assert all_fields == Counter(field.field_id for field in contract.plan.fields)


def test_last_stage_context_cache_waits_for_one_final_barrier():
    contract = _k3_contract(5)
    for owner in pipeline_cache_ownership(93, 5, 4, None):
        resident = contract.plan.narrow_to_layers(*owner.resident_window)
        schedule = build_cache_fields_by_producer_step(resident, ownership=owner)
        target_steps = owner.target_window[1] - owner.target_window[0]
        assert schedule.step_count == target_steps + int(owner.owns_draft)
        expected_target = {
            field.field_id
            for field in resident.fields
            if cache_field_layer_id(field.field_id) < 93
        }
        assert schedule.fields_in_range(0, target_steps) == expected_target
        if owner.owns_draft:
            assert schedule.fields_by_step[-1] == tuple(
                f"layer.{layer}.latent_kv" for layer in range(93, 98)
            )
        assert schedule.fields_in_range(0, schedule.step_count) == {
            field.field_id for field in resident.fields
        }


def test_non_pp_keeps_the_whole_arena_and_equal_tp_fast_route():
    contract = _k3_contract(5)
    (owner,) = pipeline_cache_ownership(93, 5, 1, None)
    assert owner.resident_window == (0, 98)
    schedule = build_cache_fields_by_producer_step(contract.plan, ownership=owner)
    assert schedule.step_count == 94
    route, dummy = build_pipeline_transfer_plan(
        prefill_tp_size=8,
        decode_tp_size=8,
        decode_tp_rank=3,
        prefill_layout=contract,
        decode_layout=contract,
        num_target_layers=93,
        pp_size=1,
        pp_layer_partition=None,
    )
    assert route.fragments_by_prefill_rank == {3: ()}
    assert dummy == ()


@pytest.mark.parametrize(
    "target,draft,window",
    [
        (0, 5, (0, 1)),
        (93, -1, (0, 24)),
        (93, 5, (70, 98)),
        (93, 5, (0, 0)),
    ],
)
def test_invalid_owner_is_rejected(target, draft, window):
    with pytest.raises(ValueError):
        CacheLayerOwnership(target, draft, window)


def test_partition_must_count_target_layers_only():
    with pytest.raises(ValueError, match="sums to"):
        pipeline_cache_ownership(93, 5, 4, (25, 25, 24, 24))


def test_context_free_draft_does_not_add_an_empty_producer_barrier():
    # Logical cache layers, not the draft model's network depth, define
    # producer readiness. A draft with no independent cache adds no barrier.
    contract = _k3_contract(0)
    for owner in pipeline_cache_ownership(93, 0, 4, None):
        resident = contract.plan.narrow_to_layers(*owner.resident_window)
        schedule = build_cache_fields_by_producer_step(resident, ownership=owner)
        assert schedule.step_count == owner.target_window[1] - owner.target_window[0]
        assert not owner.owns_draft
        assert all(schedule.fields_by_step)
