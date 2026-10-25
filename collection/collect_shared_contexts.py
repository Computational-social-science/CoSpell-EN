"""One CPU-only C4 pass for all target words; no sampling or model inference."""
import argparse
from collections import defaultdict
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import time
import traceback

POLICY = "shared_full_c4_collect_then_filter_20261021"
LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
SPLIT = re.compile(r'(?<=[.!?])\s+')
AUTOMATON = None
WORD_IDS = None
ROOT = None
CONFIG = None


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def atomic(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + "." + str(os.getpid()) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def identity(pid=None):
    pid = pid or os.getpid()
    p = Path("/proc") / str(pid)
    try:
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, uid=p.stat().st_uid, state=stat[0],
                    pgid=int(stat[2]), start_ticks=int(stat[19]),
                    argv=[x.decode() for x in (p / "cmdline").read_bytes().split(b"\0") if x])
    except FileNotFoundError:
        return None


def init_matcher(words):
    global AUTOMATON, WORD_IDS
    import ahocorasick
    grouped = defaultdict(list)
    for i, word in enumerate(words):
        assert word
        grouped[word.translate(LOWER)].append(i)
    AUTOMATON = ahocorasick.Automaton()
    for key, ids in grouped.items():
        AUTOMATON.add_word(key, (len(key), tuple(ids)))
    AUTOMATON.make_automaton()
    WORD_IDS = words


def word_char(char):
    # Recall deliberately includes quoted/hyphenated forms. Downstream quality
    # checks retain their stricter original boundaries and case requirements.
    return char.isalnum() or char == "_"


def sentence_spans(text):
    left = 0
    for boundary in SPLIT.finditer(text):
        right = boundary.start()
        a = left
        while a < right and text[a].isspace():
            a += 1
        while right > a and text[right - 1].isspace():
            right -= 1
        if a < right:
            yield a, right
        left = boundary.end()
    right = len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        yield left, right


def extract(text):
    """One automaton pass per document. Preserve every matching source span."""
    matches = []
    for end, (length, ids) in AUTOMATON.iter(text.translate(LOWER)):
        start = end - length + 1
        if start and word_char(text[start - 1]):
            continue
        if end + 1 < len(text) and word_char(text[end + 1]):
            continue
        matches.append((start, end + 1, ids))
    if not matches:
        return
    matches.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    for start, end in sentence_spans(text):
        found = set()
        while cursor < len(matches) and matches[cursor][0] < end:
            a, b, ids = matches[cursor]
            if a >= start and b <= end:
                found.update(ids)
            cursor += 1
        if found:
            yield start, end, sorted(found)


def disk_room():
    while shutil.disk_usage(ROOT).free < CONFIG["minimum_free_disk_gib"] * 2**30:
        atomic(ROOT / "workers" / (str(os.getpid()) + ".json"),
               dict(identity=identity(), phase="waiting_disk", time=time.time()))
        time.sleep(30)


def source_valid(source):
    stat = Path(source["path"]).stat()
    assert (stat.st_size, stat.st_mtime_ns) == (source["bytes"], source["mtime_ns"]), "source changed"


def scan_shard(source):
    import orjson
    source_valid(source)
    shard = ROOT / "shards" / source["name"]
    shard.mkdir(parents=True, exist_ok=True)
    with (shard / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = shard / "checkpoint.json"
        state = read(manifest_path) if manifest_path.exists() else {
            "policy": POLICY, "source": source, "plan_sha256": CONFIG["plan_sha256"],
            "committed_line": 0, "records": 0, "segments": [], "eof": False,
        }
        assert state["source"] == source and state["plan_sha256"] == CONFIG["plan_sha256"]
        if state["eof"]:
            return dict(name=source["name"], lines=state["committed_line"], records=state["records"])
        resume = state["committed_line"]
        rows = []
        start_line = resume + 1
        last_commit = time.monotonic()
        progress_at = 0
        line_no = 0

        def commit(line, eof=False):
            nonlocal rows, last_commit, start_line
            disk_room()
            index = len(state["segments"])
            dest = shard / ("part-%07d.jsonl.gz" % index)
            tmp = dest.with_name(dest.name + ".writing")
            # At most one uncommitted tail file is replaced after a crash.
            # A segment referenced by checkpoint.json is never overwritten.
            with tmp.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as out:
                    for row in rows:
                        out.write(orjson.dumps(row) + b"\n")
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(tmp, dest)
            part = dict(file=dest.name, first_line=start_line, last_line=line,
                        records=len(rows), bytes=dest.stat().st_size, sha256=sha(dest))
            state["segments"].append(part)
            state.update(committed_line=line, records=state["records"] + len(rows),
                         eof=eof, updated_at_unix=time.time())
            atomic(manifest_path, state)
            rows = []
            start_line = line + 1
            last_commit = time.monotonic()

        with gzip.open(source["path"], "rb") as stream:
            for line_no, raw in enumerate(stream, 1):
                if line_no <= resume:
                    continue
                # Malformed input fails visibly; it is never counted as complete.
                document = orjson.loads(raw)
                text = document.get("text")
                if text is None:
                    text = ""
                assert isinstance(text, str), "non-text C4 document"
                document_hash = None
                for start, end, ids in extract(text):
                    if document_hash is None:
                        document_hash = text_sha(text)
                    sentence = text[start:end]
                    rows.append(dict(sentence=sentence, sentence_hash=text_sha(sentence),
                                     word_ids=ids, line=line_no, start=start, end=end,
                                     document_hash=document_hash, url=document.get("url")))
                if len(rows) >= CONFIG["segment_rows"] or line_no - state["committed_line"] >= CONFIG["checkpoint_lines"] or time.monotonic() - last_commit >= 20:
                    commit(line_no)
                if time.monotonic() - progress_at >= 10:
                    atomic(ROOT / "workers" / (str(os.getpid()) + ".json"),
                           dict(identity=identity(), phase="scanning", source=source["name"],
                                read_line=line_no, committed_line=state["committed_line"],
                                saved_records=state["records"], time=time.time()))
                    progress_at = time.monotonic()
        assert line_no >= resume, "source became shorter"
        source_valid(source)
        commit(line_no, eof=True)
        atomic(ROOT / "workers" / (str(os.getpid()) + ".json"),
               dict(identity=identity(), phase="shard_complete", source=source["name"],
                    committed_line=line_no, saved_records=state["records"], time=time.time()))
        return dict(name=source["name"], lines=line_no, records=state["records"])


def complete_proof(root):
    root = Path(root)
    plan = read(root / "scan_plan.json")
    entries = []
    for source in plan["files"]:
        p = root / "shards" / source["name"] / "checkpoint.json"
        state = read(p)
        assert state["eof"] and state["source"] == source
        assert state["plan_sha256"] == sha(root / "scan_plan.json")
        assert state["committed_line"] > 0
        source_valid(source)
        line = 0
        records = 0
        for part in state["segments"]:
            assert part["first_line"] == line + 1
            assert part["last_line"] >= line
            dest = p.parent / part["file"]
            assert dest.stat().st_size == part["bytes"]
            # Full content hashes are verified as segments enter the shared index.
            line = part["last_line"]
            records += part["records"]
        assert line == state["committed_line"] and records == state["records"]
        entries.append(dict(**source, final_line=line, scanned_to_eof=True, complete_file=True,
                            records=records, checkpoint_sha256=sha(p)))
    return dict(policy=POLICY, plan_sha256=sha(root / "scan_plan.json"),
                targets_sha256=sha(root / "targets.json"), files=entries,
                scope="all C4 English train shards; validation excluded",
                no_per_word_or_per_relation_cap=True, completed_at_unix=time.time())


def main():
    global ROOT, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    ROOT = Path(args.directory).resolve()
    CONFIG = read(ROOT / "config.json")
    CONFIG["plan_sha256"] = sha(ROOT / "scan_plan.json")
    assert CONFIG["policy"] == POLICY
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(ROOT / "worker.json", identity())
        plan = read(ROOT / "scan_plan.json")
        assert plan["targets_sha256"] == sha(ROOT / "targets.json")
        words = read(ROOT / "targets.json")
        atomic(ROOT / "status.json", dict(phase="building_matcher", time=time.time(),
                                         target_words=len(words), worker=identity()))
        init_matcher(words)
        # Fork once after constructing a read-only matcher. Avoid one large
        # matcher copy per task and avoid shipping candidate sentences via IPC.
        ctx = mp.get_context("fork")
        done = 0
        with ctx.Pool(CONFIG["workers"]) as pool:
            results = pool.imap_unordered(scan_shard, plan["files"], chunksize=1)
            while done < len(plan["files"]):
                try:
                    result = results.next(timeout=20)
                    done += 1
                except mp.TimeoutError:
                    result = None
                atomic(ROOT / "status.json", dict(phase="collecting", completed_shards=done,
                                                 total_shards=len(plan["files"]), last_completed=result,
                                                 workers=CONFIG["workers"], model_inference_enabled=False,
                                                 time=time.time(), worker=identity()))
        proof = complete_proof(ROOT)
        atomic(ROOT / "collection_complete.json", proof)
        atomic(ROOT / "status.json", dict(phase="collection_complete", time=time.time(),
                                         total_shards=len(plan["files"]), model_inference_enabled=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if ROOT:
            atomic(ROOT / "error.json", dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
