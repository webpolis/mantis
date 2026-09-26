"""
Context extension in one command.

`train.py --length-schedule 32768:130M,131072:200M,262144:650M` runs one
train.py process per phase, in order, each writing to `<output-dir>/ctx-<length>`:

- the first phase starts from `--init-from` (or from scratch), every later one
  from the previous phase's final_model.pt, with a fresh LR schedule;
- phases longer than the pretraining length get `--profile long-context`
  (local/global layout, YaRN factor = length / pretraining length, bounded
  memory), batch 1 and enough gradient accumulation to keep the tokens per
  optimizer update of the base run, `--extension-lr` and a 5% warmup;
- every phase saves each epoch, so rerunning the same command skips finished
  phases and resumes the unfinished one from its latest epoch checkpoint.

Phases run as separate processes: each gets a clean GPU, accelerator and
compile cache, and each is exactly the `--init-from`/`--resume` run a user
could have typed by hand.
"""

import glob
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

UNITS = {'': 1, 'K': 10**3, 'M': 10**6, 'B': 10**9}

# Flags the driver sets per phase; the rest of the command line passes through
PER_PHASE_FLAGS = {
    '--length-schedule', '--init-from', '--resume', '--seq-len', '--output-dir', '--steps-per-epoch',
    '--epochs', '--batch-size', '--gradient-accumulation-steps', '--learning-rate', '--warmup-steps',
    '--save-every', '--rope-factor', '--extension-lr', '--profile', '--val-max-batches',
}


@dataclass
class Phase:
    length: int
    tokens: int
    batch_size: int
    accumulation: int
    steps_per_epoch: int
    epochs: int
    learning_rate: float
    warmup_steps: int
    val_max_batches: Optional[int]
    long_context: bool


def parse_count(text: str) -> int:
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMB]?)', text.strip().upper())
    if not match:
        raise ValueError(f"not a token count: {text!r} (use e.g. 650M or 1.5B)")
    return int(float(match.group(1)) * UNITS[match.group(2)])


def format_count(count: int) -> str:
    for unit in ('B', 'M', 'K'):
        if count >= UNITS[unit]:
            return f"{count / UNITS[unit]:g}{unit}"
    return str(count)


def parse_schedule(text: str) -> List[Tuple[int, int]]:
    """'32768:130M,262144:650M' -> [(32768, 130000000), (262144, 650000000)], lengths increasing."""
    phases = []
    for item in text.split(','):
        length, _, tokens = item.partition(':')
        if not tokens:
            raise ValueError(f"phase {item!r} needs LENGTH:TOKENS, e.g. 32768:130M")
        phases.append((parse_count(length), parse_count(tokens)))
    lengths = [length for length, _ in phases]
    if lengths != sorted(set(lengths)):
        raise ValueError("phase lengths must be strictly increasing")
    return phases


def plan_phases(schedule, base_length, batch_size, accumulation, steps_per_epoch, learning_rate,
                warmup_steps, extension_lr, val_max_batches) -> List[Phase]:
    """
    Turn (length, tokens) pairs into train.py settings. `base_length`,
    `batch_size`, `accumulation` and `steps_per_epoch` describe the
    pretraining run: its tokens per optimizer update, per epoch and per
    validation are kept at every length.
    """
    step_tokens = batch_size * accumulation * base_length
    epoch_tokens = (steps_per_epoch or 1000) * batch_size * base_length
    val_tokens = val_max_batches * batch_size * base_length if val_max_batches else None
    phases = []
    for length, tokens in schedule:
        long_context = length > base_length
        if long_context:
            batch, accum, lr = 1, max(1, -(-step_tokens // length)), extension_lr
        else:
            batch, accum, lr = batch_size, accumulation, learning_rate
        per_micro = batch * length
        micro = max(accum, -(-tokens // per_micro))
        micro = -(-micro // accum) * accum
        epoch_micro = max(accum, epoch_tokens // per_micro // accum * accum)
        spe = min(micro, epoch_micro)
        epochs = -(-micro // spe)
        warmup = max(1, epochs * spe // accum // 20) if long_context else warmup_steps
        phases.append(Phase(length, tokens, batch, accum, spe, epochs, lr, warmup,
                            max(1, val_tokens // per_micro) if val_tokens else None, long_context))
    return phases


def strip_flags(argv: List[str], flags=PER_PHASE_FLAGS) -> List[str]:
    """Drop `flags` (each taking one value, as `--flag v` or `--flag=v`) from a command line."""
    out, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        name = token.split('=', 1)[0]
        if name in flags:
            skip = '=' not in token
            continue
        out.append(token)
    return out


def latest_epoch_checkpoint(phase_dir: str) -> Optional[str]:
    found = []
    for path in glob.glob(os.path.join(phase_dir, 'epoch_*.pt')):
        match = re.search(r'epoch_(\d+)\.pt$', path)
        if match:
            found.append((int(match.group(1)), path))
    return max(found)[1] if found else None


def phase_source(phase_dir: str, index: int, init_from: Optional[str], previous_dir: Optional[str]):
    """None when the phase is finished, else the flags that start or resume it."""
    if os.path.exists(os.path.join(phase_dir, 'final_model.pt')):
        return None
    latest = latest_epoch_checkpoint(phase_dir)
    if latest:
        return ['--resume', latest]
    if index > 0:
        return ['--init-from', os.path.join(previous_dir, 'final_model.pt')]
    return ['--init-from', init_from] if init_from else []


def trained_length(checkpoint_path: str) -> int:
    """The context a checkpoint was pretrained at (before any YaRN extension)."""
    from mantis.utils.checkpoints import compat_load
    config = compat_load(checkpoint_path, mmap=True)['config'].base_moe
    return config.rope_original_context if config.local_window else config.max_seq_len


def run_length_schedule(args, argv: List[str], script: str) -> int:
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        print("Error: --length-schedule starts one single-GPU run per phase; launch it without torchrun")
        return 1
    schedule = parse_schedule(args.length_schedule)
    base_length = trained_length(args.init_from) if args.init_from else schedule[0][0]
    phases = plan_phases(schedule, base_length, args.batch_size, args.gradient_accumulation_steps,
                         args.steps_per_epoch, args.learning_rate, args.warmup_steps, args.extension_lr,
                         args.val_max_batches)
    passthrough = strip_flags(argv)

    print(f"Length schedule from {base_length:,}-token pretraining"
          f"{' (' + args.init_from + ')' if args.init_from else ''}:")
    for p in phases:
        print(f"  {p.length:>9,} tokens, {format_count(p.tokens)} in total: batch {p.batch_size} x accumulation "
              f"{p.accumulation}, {p.epochs} epoch(s) of {p.steps_per_epoch} steps, lr {p.learning_rate:g}"
              f"{', long-context profile' if p.long_context else ''}")

    previous_dir = None
    for index, phase in enumerate(phases):
        phase_dir = os.path.join(args.output_dir, f'ctx-{phase.length}')
        source = phase_source(phase_dir, index, args.init_from, previous_dir)
        previous_dir = phase_dir
        if source is None:
            print(f"\n✓ Phase {phase.length:,} already finished ({phase_dir}/final_model.pt)")
            continue
        extra = source + [
            '--seq-len', str(phase.length), '--output-dir', phase_dir,
            '--batch-size', str(phase.batch_size), '--gradient-accumulation-steps', str(phase.accumulation),
            '--steps-per-epoch', str(phase.steps_per_epoch), '--epochs', str(phase.epochs),
            '--learning-rate', f'{phase.learning_rate:g}', '--warmup-steps', str(phase.warmup_steps),
            '--save-every', '1',
        ]
        if phase.val_max_batches:
            extra += ['--val-max-batches', str(phase.val_max_batches)]
        if phase.long_context:
            extra += ['--profile', 'long-context']
        if source and source[0] == '--resume':
            extra += ['--tokenizer-path', os.path.join(phase_dir, 'tokenizer')]  # the copy the phase saved
        command = [sys.executable, script] + passthrough + extra
        print(f"\n{'=' * 80}\nPhase {index + 1}/{len(phases)}: {phase.length:,} tokens\n"
              f"{shlex.join(command)}\n{'=' * 80}", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0 or not os.path.exists(os.path.join(phase_dir, 'final_model.pt')):
            print(f"\nPhase {phase.length:,} stopped (exit code {result.returncode}); "
                  "rerun the same command to resume it")
            return result.returncode or 1
    print(f"\n✓ Length schedule complete: {os.path.join(previous_dir, 'final_model.pt')}")
    return 0
