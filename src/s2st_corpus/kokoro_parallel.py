"""Spawned Kokoro workers own WAVs; the parent alone owns checkpoints."""
from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def terminate_as_interrupt():
    """Let Colab subprocess.terminate() unwind child pools on Linux."""
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)
        def interrupt(signum, frame):
            raise KeyboardInterrupt("termination requested")
        signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def memory_peaks(torch):
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    return {"allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}


def _worker_main(worker_id, config, shard_name, asset_root, tasks, results, backend_factory=None):
    # This top-level target is pickleable; no CUDA state is inherited by fork.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        from .kokoro_backend import KokoroBackend
        from .runner import _generate_attempt, _paths
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        factory = backend_factory or KokoroBackend
        backend = factory(config.tts, asset_root=asset_root, local_only=True, preload_voices=True)
        load_seconds = time.perf_counter() - started
        results.put({"event": "ready", "worker_id": worker_id,
                     "load_seconds": load_seconds, "memory": memory_peaks(torch)})
        while True:
            task = tasks.get()
            if task is None:
                results.put({"event": "done", "worker_id": worker_id, "memory": memory_peaks(torch)})
                return
            pair_id, language, attempt, text = task
            record = {"pair_id": pair_id, f"{language}_tts_text": text}
            paths = [p for p in _paths(config, shard_name, pair_id, language, attempt) if p is not None]
            before = {p: (p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else (0, None) for p in paths}
            result = _generate_attempt(backend, torch, config, shard_name, record, language, attempt)
            files = []
            for path in paths:
                after = (path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else (0, None)
                files.append({"path": str(path), "previous_bytes": before[path][0], "new_size": after[0],
                              "written_bytes": after[0] if after != before[path] else 0})
            result.update(worker_id=worker_id, worker_count=config.tts.worker_count)
            results.put({"event": "result", "key": (pair_id, language, attempt),
                         "attempt_result": result, "files": files, "worker_id": worker_id,
                         "model_load_seconds": load_seconds,
                         "memory": memory_peaks(torch)})
    except BaseException as error:
        results.put({"event": "fatal", "worker_id": worker_id,
                     "error": f"{type(error).__name__}: {error}"})


def iter_worker_results(config, shard_name, tasks, asset_root, *, backend_factory=None, context=None, timeout=1200, worker_stats=None):
    """Dynamic bounded queue, no restart after a worker or CUDA failure.

    Payloads contain text on input and metadata on output, never waveforms.
    """
    if len({t[:3] for t in tasks}) != len(tasks):
        raise ValueError("duplicate Kokoro task key")
    if not tasks:
        return
    ctx = context or multiprocessing.get_context("spawn")
    if ctx.get_start_method() != "spawn":
        raise ValueError("Kokoro multiprocessing requires spawn; CUDA fork is forbidden")
    count = config.tts.worker_count
    incoming, outgoing = ctx.Queue(maxsize=2 * count), ctx.Queue()
    processes = []
    completed = set()
    in_flight = set()
    iterator = iter(tasks)
    last_activity = time.monotonic()
    done = set()
    def submit():
        task = next(iterator, None)
        if task is None:
            return False
        incoming.put(task)
        in_flight.add(task[:3])
        return True
    try:
        with terminate_as_interrupt():
            for worker_id in range(count):
                process = ctx.Process(target=_worker_main, args=(worker_id, config, shard_name,
                    str(asset_root), incoming, outgoing, backend_factory), name=f"kokoro-{worker_id}")
                process.start()
                processes.append(process)
            for _ in range(min(len(tasks), 2 * count)):
                submit()
            while len(completed) < len(tasks) or len(done) < count:
                try:
                    event = outgoing.get(timeout=.5)
                except queue.Empty:
                    failed = [p for p in processes if p.exitcode is not None and (p.exitcode != 0 or int(p.name.split('-')[-1]) not in done)]
                    if failed:
                        raise RuntimeError(f"Kokoro worker process crashed: {[(p.name, p.exitcode) for p in failed]}")
                    if time.monotonic() - last_activity > timeout:
                        raise RuntimeError("Kokoro worker timed out; refusing automatic restart")
                    continue
                last_activity = time.monotonic()
                if worker_stats is not None and "memory" in event:
                    stats = worker_stats.setdefault(str(event["worker_id"]), {})
                    stats["memory"] = event["memory"]
                    if "load_seconds" in event:
                        stats["load_seconds"] = event["load_seconds"]
                if event["event"] == "fatal":
                    raise RuntimeError(f"Kokoro worker {event['worker_id']} fatal: {event['error']}")
                if event["event"] == "ready":
                    continue
                if event["event"] == "done":
                    done.add(event["worker_id"])
                    continue
                key = tuple(event["key"])
                if key not in in_flight or key in completed:
                    raise RuntimeError(f"unexpected/duplicate Kokoro worker result: {key}")
                in_flight.remove(key)
                completed.add(key)
                yield event
                submit()
                if len(completed) == len(tasks):
                    for _ in processes:
                        incoming.put(None)
    finally:
        # No Whisper model may load until all TTS processes have exited.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for channel in (incoming, outgoing):
            channel.cancel_join_thread()
            channel.close()


def generate_missing_parallel(config, shard_name, records, attempt, missing, checkpoint_path):
    from . import metrics
    from .io import account_worker_write
    from .kokoro_backend import materialize_assets
    from .runner import _IntervalCheckpoint, _paths
    saver = _IntervalCheckpoint(config, checkpoint_path, records)
    by_id = {r["pair_id"]: r for r in records}
    tasks = [(pid, lang, attempt, by_id[pid][f"{lang}_tts_text"]) for pid, lang in sorted(missing)]
    worker_stats = {}
    events = None
    def record_stats():
        active = metrics.ACTIVE.get()
        if active is not None and worker_stats:
            peaks = {k: v["memory"] for k, v in worker_stats.items()}
            active["tts_worker_per_process_peak"] = peaks
            active["tts_worker_load_seconds"] = {k: v.get("load_seconds") for k, v in worker_stats.items()}
            active["tts_worker_peak_reserved_sum_gib_reference"] = sum(p["reserved_gib"] for p in peaks.values())
            active["tts_worker_peak_allocated_sum_gib_reference"] = sum(p["allocated_gib"] for p in peaks.values())
            active["tts_max_worker_peak_reserved_gib"] = max(p["reserved_gib"] for p in peaks.values())
            active["tts_max_worker_peak_allocated_gib"] = max(p["allocated_gib"] for p in peaks.values())
    try:
        started = time.perf_counter()
        root = materialize_assets(config.tts)  # parent completes all downloads first
        metrics.add_time("tts_asset_seconds", time.perf_counter() - started)
        events = iter_worker_results(config, shard_name, tasks, root, worker_stats=worker_stats)
        for i, event in enumerate(events, 1):
            pid, language, returned_attempt = event["key"]
            result = event["attempt_result"]
            expected_path = _paths(config, shard_name, pid, language, attempt)[1]
            if returned_attempt != attempt or Path(result["wav_16k"]) != expected_path:
                raise RuntimeError("Kokoro result path/attempt mismatch")
            by_id[pid]["attempts"][language].append(result)
            if result.get("generation_status") == "ok":
                metrics.add_time("tts_generated_utterances", 1)
                metrics.add_time("tts_generated_audio_seconds", result["duration"])
            for file in event["files"]:
                account_worker_write(**file)
            record_stats()
            saver.advance()
            print(f"[tts-worker] {i}/{len(tasks)} {pid} {language} worker={event['worker_id']} status={result['generation_status']}", flush=True)
        record_stats()
        saver.flush()
    except BaseException:
        record_stats()
        saver.flush()
        raise
    finally:
        if events is not None:
            events.close()
