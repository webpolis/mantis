import pytest

from mantis.training.length_schedule import (
    parse_count, parse_schedule, phase_source, plan_phases, strip_flags,
)


def test_parse_schedule():
    assert parse_count('650M') == 650_000_000 and parse_count('1.5B') == 1_500_000_000 and parse_count('32768') == 32768
    assert parse_schedule('32768:130M,262144:650M') == [(32768, 130_000_000), (262144, 650_000_000)]
    for bad in ('32768', '262144:1M,32768:1M', '32K:x'):
        with pytest.raises(ValueError):
            parse_schedule(bad)


def test_plan_keeps_the_base_run_tokens_per_update_epoch_and_validation():
    # The medium run on the 4090: batch 8, accumulation 8, 1000 micro-steps per epoch at 2048, 25 val batches
    phases = plan_phases(parse_schedule('2048:100M,32768:130M,262144:650M'), base_length=2048, batch_size=8,
                         accumulation=8, steps_per_epoch=1000, learning_rate=3e-4, warmup_steps=500,
                         extension_lr=1e-5, val_max_batches=25)
    base, mid, long = phases
    assert (base.batch_size, base.accumulation, base.learning_rate, base.long_context) == (8, 8, 3e-4, False)
    assert (mid.batch_size, mid.accumulation, mid.learning_rate, mid.long_context) == (1, 4, 1e-5, True)
    assert (long.batch_size, long.accumulation) == (1, 1)
    for p in phases:
        assert p.steps_per_epoch % p.accumulation == 0
        assert p.batch_size * p.length * p.steps_per_epoch * p.epochs >= p.tokens      # the budget is covered
        assert p.batch_size * p.length * p.steps_per_epoch <= 1000 * 8 * 2048         # epochs no longer than the base
    assert mid.steps_per_epoch == 500 and long.steps_per_epoch == 62
    assert long.val_max_batches == 1 and mid.val_max_batches == 12
    assert long.warmup_steps == max(1, long.epochs * long.steps_per_epoch // 20)


def test_strip_flags_keeps_everything_else():
    argv = ['data.jsonl', '--init-from', 'a.pt', '--seq-len=4096', '--use-8bit-optimizer',
            '--length-schedule', '8192:1M', '--hf-config', 'x', '--profile', 'long-context']
    assert strip_flags(argv) == ['data.jsonl', '--use-8bit-optimizer', '--hf-config', 'x']


def test_phase_source_skips_finished_phases_and_resumes_the_latest_epoch(tmp_path):
    first, second = tmp_path / 'ctx-8192', tmp_path / 'ctx-32768'
    assert phase_source(str(first), 0, 'base.pt', None) == ['--init-from', 'base.pt']
    assert phase_source(str(first), 0, None, None) == []
    first.mkdir()
    for n in (2, 10, 9):
        (first / f'epoch_{n}.pt').touch()
    assert phase_source(str(first), 0, 'base.pt', None) == ['--resume', str(first / 'epoch_10.pt')]
    (first / 'final_model.pt').touch()
    assert phase_source(str(first), 0, 'base.pt', None) is None
    assert phase_source(str(second), 1, 'base.pt', str(first)) == ['--init-from', str(first / 'final_model.pt')]
