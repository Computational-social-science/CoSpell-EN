"""Bounded access to saved sentence blocks, without a global sentence index."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from collections import defaultdict, OrderedDict, Counter

try:
    import orjson
except ImportError:
    orjson = None


def decode_row(line):
    # Keep the standard decoder's accepted input domain and exact model inputs.
    if orjson is not None:
        try:
            return orjson.loads(line)
        except orjson.JSONDecodeError:
            pass
    return json.loads(line)
from shared_collect import atomic, read, sha, complete_proof

MODE = "sequential_blocks_first_success_20261021"


def prepare_stream(root):
    root = Path(root)
    if (root / "deployment_manifest.json").exists():
        deployment = read(root / "deployment_manifest.json")
        # The original supervisor is still running its pre-upgrade code. Its
        # in-memory manifest may predate this upgrade; keep a separate receipt.
        if (root / "stream_deployment.json").exists():
            deployment["streaming_update"] = read(root / "stream_deployment.json")
        deployment.update(phase="preparing_saved_stream", updated_at_unix=time.time())
        atomic(root / "deployment_manifest.json", deployment)
    collection = read(root / "collection_complete.json")
    checked = complete_proof(root)
    assert checked["files"] == collection["files"]
    result = dict(mode=MODE, collection_certificate_sha256=sha(root / "collection_complete.json"),
                  targets_sha256=sha(root / "targets.json"),
                  total_records=sum(s["records"] for s in collection["files"]),
                  shards=len(collection["files"]), original_sentence_files_shared=True,
                  global_sentence_index=False, completed_at_unix=time.time())
    path = root / "stream_ready.json"
    if path.exists():
        previous = read(path)
        assert all(previous[k] == result[k] for k in result if k != "completed_at_unix")
        return previous
    atomic(path, result)
    return result


def ready(root):
    root = Path(root)
    result = read(root / "stream_ready.json")
    assert result["mode"] == MODE
    assert result["collection_certificate_sha256"] == sha(root / "collection_complete.json")
    assert result["targets_sha256"] == sha(root / "targets.json")
    return result


class Blocks:
    """A small block catalog; sentence contents stay in original saved segments."""
    def __init__(self, root):
        self.root = Path(root)
        self.proof = ready(self.root)
        self.sources = read(self.root / "collection_complete.json")["files"]
        self.cached_source = None
        self.parts = None
        self.source_parts = OrderedDict()

    def next(self, position):
        source_index, part_index = position
        while source_index < len(self.sources):
            source = self.sources[source_index]
            if self.cached_source != source_index:
                if source_index not in self.source_parts:
                    path = self.root / "shards" / source["name"] / "checkpoint.json"
                    assert sha(path) == source["checkpoint_sha256"]
                    self.source_parts[source_index] = read(path)["segments"]
                    while len(self.source_parts) > 16:
                        self.source_parts.popitem(last=False)
                self.parts = self.source_parts[source_index]
                self.source_parts.move_to_end(source_index)
                self.cached_source = source_index
            if part_index < len(self.parts):
                part = self.parts[part_index]
                return dict(key=f"{source_index}:{part_index}", source=source, part=part), [
                    source_index, part_index + 1]
            source_index += 1
            part_index = 0
        return None, [len(self.sources), 0]

    def get(self, key):
        source_index, part_index = map(int, key.split(":"))
        block, _ = self.next([source_index, part_index])
        assert block and block["key"] == key
        return block


class LegacyBlockCache:
    """Share decoded blocks between POS workers. Eviction only removes cache files."""
    def __init__(self, root, max_blocks=256, max_bytes=8 * 2**30):
        self.root = Path(root)
        self.directory = self.root / "decoded_block_cache"
        self.directory.mkdir(exist_ok=True)
        self.max_blocks, self.max_bytes = max_blocks, max_bytes
        self.stats = Counter()

    @staticmethod
    def key(block):
        return hashlib.sha256((block["source"]["path"] + block["part"]["sha256"]).encode()).hexdigest()

    def count(self, key):
        self.stats[key] += 1

    def load(self, block):
        return self._load(block)

    def _load(self, block):
        part = block["part"]
        key = self.key(block)
        bucket = self.directory / key[:2]
        bucket.mkdir(exist_ok=True)
        path = bucket / (key + ".pickle")
        # Cache files are immutable after atomic rename. Readers may open them
        # concurrently; eviction after open cannot invalidate the open file.
        # Only cache misses/rebuilds need the exclusive producer lock.
        try:
            with path.open("rb") as stream:
                value = pickle.load(stream)
                modified = os.fstat(stream.fileno()).st_mtime
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
            value = None
        if value is not None:
            assert value["sha256"] == part["sha256"]
            if time.time() - modified > 30:
                try:
                    os.utime(path, None)
                except FileNotFoundError:
                    pass
            self.count("disk_hits")
            self.count("lockfree_cache_hits")
            return value
        # A fixed set of lock files avoids one permanent lock per data block.
        with (self.directory / ("lock-%02d" % (int(key[:4], 16) % 64))).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                try:
                    with path.open("rb") as stream:
                        value = pickle.load(stream)
                except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
                    # Cache loss is recoverable from checksummed originals.
                    value = None
                if value is not None:
                    assert value["sha256"] == part["sha256"]
                    try:
                        os.utime(path, None)
                    except FileNotFoundError:
                        pass
                    self.count("disk_hits")
                    return value
            source = self.root / "shards" / block["source"]["name"] / part["file"]
            assert sha(source) == part["sha256"], "saved sentence block changed"
            rows, word_rows = [], defaultdict(list)
            with gzip.open(source, "rb") as stream:
                for index, line in enumerate(stream):
                    row = decode_row(line)
                    assert row["word_ids"] == sorted(set(row["word_ids"]))
                    assert row["end"] - row["start"] == len(row["sentence"])
                    rows.append(row)
                    for word in row["word_ids"]:
                        word_rows[word].append(index)
            assert len(rows) == part["records"]
            value = dict(sha256=part["sha256"], rows=rows, word_rows=dict(word_rows))
            temporary = path.with_suffix("." + str(os.getpid()) + ".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                # This cache can be rebuilt. Durable candidate, accepted and
                # checkpoint commits are unchanged; cache uses atomic rename.
            os.replace(temporary, path)
            self.count("decoded_blocks")
        self.trim()
        return value

    def trim(self):
        with (self.directory / "trim.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            files = []
            for path in self.directory.glob("*/*.pickle"):
                try:
                    st = path.stat()
                    files.append((st.st_mtime_ns, st.st_size, path))
                except FileNotFoundError:
                    pass
            total = sum(v[1] for v in files)
            remaining = len(files)
            for _, size, path in sorted(files):
                if remaining <= self.max_blocks and total <= self.max_bytes:
                    break
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                total -= size
                remaining -= 1



class BlockCache(LegacyBlockCache):
    """Shared bounded cache with two decode workers and persistent word directories."""
    def __init__(self, root, max_blocks=8192, max_bytes=32 * 2**30):
        super().__init__(root, max_blocks, max_bytes)
        self.directory = self.root / 'decoded_block_cache_v2'
        self.directory.mkdir(exist_ok=True)
        self.catalog_memory = OrderedDict()
        self.last_ensure = 0
        self.last_trim = 0
        self.writes_since_trim = 0
        self.service_enabled = (self.root/'decode_pipeline_config.json').exists()
        try:
            self.minimum_free_gib = read(self.root/'config.json').get('minimum_free_disk_gib', 100)
        except FileNotFoundError:
            self.minimum_free_gib = 100

    def cache_path(self, block):
        key = self.key(block)
        return self.directory / key[:2] / (key+'.pickle')

    def catalog_path(self, block):
        key = self.key(block)
        return self.directory / 'word_catalog' / key[:2] / (key+'.json')

    def _load(self, block):
        if not self.cache_path(block).exists():
            import shutil
            if shutil.disk_usage(self.root).free < self.minimum_free_gib * 2**30:
                raise RuntimeError('Saved-block cache stopped at minimum free disk threshold')
        value = super()._load(block)
        catalog = self.catalog_path(block)
        if not catalog.exists():
            catalog.parent.mkdir(parents=True, exist_ok=True)
            from decode_service import atomic as cache_atomic
            cache_atomic(catalog, dict(sha256=block['part']['sha256'], word_ids=sorted(value['word_rows'])))
        return value

    def relevant(self, block, words):
        """Unknown directories are never grounds to skip a block."""
        key = self.key(block)
        if key not in self.catalog_memory:
            try:
                value = read(self.catalog_path(block))
            except (FileNotFoundError, ValueError):
                return True
            if value.get('sha256') != block['part']['sha256']:
                return True
            self.catalog_memory[key] = frozenset(value['word_ids'])
            while len(self.catalog_memory) > 8192:
                self.catalog_memory.popitem(last=False)
        self.catalog_memory.move_to_end(key)
        relevant = not self.catalog_memory[key].isdisjoint(words)
        if not relevant:
            self.count('directory_skips')
        return relevant

    def ensure_service(self):
        from decode_service import ensure
        if time.monotonic()-self.last_ensure > 5:
            if not ensure(self.root):
                self.service_enabled = False
                return
            self.last_ensure = time.monotonic()

    def prefetch(self, blocks):
        if not self.service_enabled:
            return
        from decode_service import submit
        self.ensure_service()
        for block in blocks[:4]:
            if not self.cache_path(block).exists():
                if submit(self.root, block, priority=1):
                    self.count('prefetch_submitted')

    def load(self, block):
        path = self.cache_path(block)
        if self.service_enabled and not path.exists():
            from decode_service import submit, read as service_read
            start = time.monotonic()
            error_path = self.directory/'runtime'/'errors'/(self.key(block)+'.json')
            while not path.exists():
                self.ensure_service()
                if not self.service_enabled:
                    break
                error = service_read(error_path)
                if error:
                    raise RuntimeError('Shared CPU decoder failed: '+str(error))
                submit(self.root, block, priority=0)
                if time.monotonic()-start > 120:
                    raise TimeoutError('Shared CPU decoder timeout: '+block['key'])
                time.sleep(.02)
            self.stats['decode_wait_seconds'] += time.monotonic()-start
        return self._load(block)

    def trim(self):
        self.writes_since_trim += 1
        # Rebuildable cache; small explicit test bounds are enforced immediately.
        if self.max_blocks <= 32:
            return super().trim()
        if self.writes_since_trim < 32 and time.monotonic()-self.last_trim < 30:
            return
        self.last_trim = time.monotonic()
        self.writes_since_trim = 0
        return super().trim()


def provenance(block, row):
    return dict(corpus="c4_en_train", source_file=block["source"]["path"],
                source_line=row["line"], document_hash=row["document_hash"], url=row["url"],
                sentence_start=row["start"], sentence_end=row["end"],
                source_sentence_hash=row["sentence_hash"],
                extraction="shared full scan; original untruncated sentence span",
                shared_scan_policy=MODE)


class SequentialStage:
    """Persist bounded look-ahead and per-relation offsets with pending candidates.

    An owner never advances past an untested candidate. Different owners may
    continue through look-ahead blocks, sharing the same decoded raw sentences.
    """
    def __init__(self, queue, root, stage, lookahead=4):
        self.queue, self.stage = queue, stage
        self.blocks = Blocks(root)
        self.cache = BlockCache(root)
        self.lookahead = lookahead
        queue.executescript("""
            CREATE TABLE IF NOT EXISTS stream_meta(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS stream_positions(
                stage TEXT,block TEXT,owner TEXT,position INTEGER,
                PRIMARY KEY(stage,block,owner));
        """)
        row = queue.execute("SELECT value FROM stream_meta WHERE key=?", (stage,)).fetchone()
        self.state = json.loads(row[0]) if row else dict(next=[0, 0], loaded=[], eof=False)
        self.loaded = {}
        self.scanned_candidates = 0

    @property
    def complete(self):
        return self.state["eof"] and not self.state["loaded"]

    def persist(self):
        self.queue.execute("INSERT OR REPLACE INTO stream_meta VALUES (?,?)",
                           (self.stage, json.dumps(self.state)))

    def fill(self, owners_for_word, enabled, candidate, enqueue, pending,
             capacity, seconds=5):
        started, added = time.monotonic(), 0
        blocked = set()
        while time.monotonic() - started < seconds and added < capacity:
            while len(self.state["loaded"]) < self.lookahead and not self.state["eof"]:
                block, position = self.blocks.next(self.state["next"])
                self.state["next"] = position
                if block is None:
                    self.state["eof"] = True
                else:
                    self.state["loaded"].append(block["key"])
                self.persist()
            if not self.state["loaded"]:
                break
            changed = False
            for key in list(self.state["loaded"]):
                if time.monotonic() - started >= seconds or added >= capacity:
                    break
                if key not in self.loaded:
                    block = self.blocks.get(key)
                    data = self.cache.load(block)
                    positions = dict(self.queue.execute(
                        "SELECT owner,position FROM stream_positions WHERE stage=? AND block=?",
                        (self.stage, key)))
                    self.loaded[key] = block, data, positions
                block, data, positions = self.loaded[key]
                unfinished = False
                for wid, indices in data["word_rows"].items():
                    # Resolve active owners each time, so acceptance removes all
                    # later candidates, including those already in look-ahead.
                    for owner in owners_for_word(wid):
                        position = positions.get(owner, 0)
                        if position >= len(indices):
                            continue
                        if owner in blocked or pending(owner) or not enabled(owner):
                            blocked.add(owner)
                            unfinished = True
                            continue
                        while position < len(indices):
                            if time.monotonic() - started >= seconds or added >= capacity:
                                unfinished = True
                                break
                            row = data["rows"][indices[position]]
                            position += 1
                            self.scanned_candidates += 1
                            record = candidate(owner, block, row)
                            changed = True
                            positions[owner] = position
                            self.queue.execute(
                                "INSERT OR REPLACE INTO stream_positions VALUES (?,?,?,?)",
                                (self.stage, key, owner, position))
                            if record and enqueue(record, owner):
                                added += 1
                                blocked.add(owner)
                                break
                        if position < len(indices):
                            unfinished = True
                            blocked.add(owner)
                if not unfinished:
                    self.state["loaded"].remove(key)
                    self.loaded.pop(key)
                    self.queue.execute("DELETE FROM stream_positions WHERE stage=? AND block=?",
                                       (self.stage, key))
                    self.persist()
                    changed = True
            if not changed:
                break
        self.persist()
        return added


def validate_exhaustion(record, directory):
    directory = Path(directory)
    configuration = read(directory / "shared_stream_source.json")
    expected = ready(configuration["directory"])
    path = directory / "shared_stream_natural_complete.json"
    result = read(path)
    assert result["mode"] == MODE and result["all_saved_blocks_consumed"] is True
    assert result["collection_certificate_sha256"] == expected["collection_certificate_sha256"]
    proof = record["scan_gate"]["shared_stream"]
    assert proof["natural_complete_sha256"] == sha(path)
    assert proof["collection_certificate_sha256"] == expected["collection_certificate_sha256"]
    assert record["scan_gate"]["natural_cursor"] >= configuration["virtual_high_rowid"]
    return True
