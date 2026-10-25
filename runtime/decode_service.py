"""Two shared CPU decoders; no model, source cursor, or experiment DB access."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

POLICY = 'shared_decode_pipeline_20261023'


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
    os.replace(tmp, path)


def process(pid):
    try:
        p = Path('/proc') / str(pid)
        st = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if st[0] == 'Z':
            return None
        return dict(pid=int(pid), identity=[p.stat().st_uid, int(st[19])],
                    pgid=int(st[2]), argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
    except (OSError, ValueError, IndexError):
        return None


def layout(root):
    folder = Path(root) / 'decoded_block_cache_v2' / 'runtime'
    for child in ['queue', 'processing', 'errors']:
        (folder / child).mkdir(parents=True, exist_ok=True)
    return folder


def config(root):
    value = read(Path(root) / 'decode_pipeline_config.json')
    if not value or not value.get('enabled'):
        return None
    assert value['policy'] == POLICY and value['workers'] == 2
    for name, digest in value['code_sha256'].items():
        assert hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() == digest, name
    return value


def ensure(root):
    cfg = config(root)
    if not cfg:
        return False
    folder = layout(root)
    with (folder/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for slot in range(2):
            meta = read(folder/f'worker_{slot}.json')
            actual = process(meta['pid']) if meta else None
            argv = [sys.executable, '-u', str(Path(root)/'decode_service.py'), '--root', str(root), '--slot', str(slot)]
            if actual and actual['identity'] == meta['identity']:
                assert actual['argv'] == argv and actual['identity'][0] == os.getuid()
                continue
            # The worker's flock is the authoritative lease, including startup.
            with (folder/f'worker_{slot}.lock').open('a') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                fcntl.flock(lease, fcntl.LOCK_UN)
            with (folder/f'worker_{slot}.log').open('a') as log:
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         close_fds=True, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            meta = process(child.pid)
            assert meta and meta['identity'][0] == os.getuid()
            # The worker replaces this startup identity with its exact argv.
            atomic(folder/f'worker_{slot}.json', dict(meta, expected_argv=argv, starting=True))
    return True


def submit(root, block, priority=1):
    folder = layout(root)
    key = hashlib.sha256((block['source']['path'] + block['part']['sha256']).encode()).hexdigest()
    path = folder/'queue'/f'{key}.json'
    with (folder/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'processing'/path.name).exists():
            return True
        old = read(path)
        if old:
            if priority < old['priority']:
                old['priority'] = priority
                atomic(path, old)
            return True
        queued = list((folder/'queue').glob('*.json'))
        if len(queued) >= 64:
            return False
        atomic(path, dict(key=block['key'], sha256=block['part']['sha256'],
                          priority=priority, created=time.time(), caller=process(os.getpid())))
    return True


def claim(folder):
    with (folder/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in (folder/'processing').glob('*.json'):
            job = read(path)
            owner = job.get('worker')
            live = process(owner['pid']) if owner else None
            if not live or live['identity'] != owner['identity']:
                os.replace(path, folder/'queue'/path.name)
        jobs = [(read(path), path) for path in (folder/'queue').glob('*.json')]
        if not jobs:
            return None
        job, path = min(jobs, key=lambda item: (item[0]['priority'], item[0]['created']))
        job['worker'] = process(os.getpid())
        atomic(path, job)
        target = folder/'processing'/path.name
        os.replace(path, target)
        return target, job


def worker(root, slot):
    assert os.getuid() == 1005
    cfg = config(root)
    assert cfg
    folder = layout(root)
    from stream_blocks import Blocks, BlockCache
    with (folder/f'worker_{slot}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        identity = process(os.getpid())
        atomic(folder/f'worker_{slot}.json', identity)
        blocks, cache = Blocks(root), BlockCache(root)
        done, idle = 0, time.monotonic()
        def status(phase, **values):
            atomic(folder/f'status_{slot}.json', dict(identity, phase=phase, at_unix=time.time(),
                                                    completed_blocks=done, stats=dict(cache.stats), **values))
        status('ready')
        while not (folder/'stop_when_idle').exists():
            task = claim(folder)
            if task is None:
                if time.monotonic() - idle > 300:
                    break
                time.sleep(.05)
                continue
            path, job = task
            started = time.monotonic()
            status('decoding', block=job['key'])
            try:
                block = blocks.get(job['key'])
                assert block['part']['sha256'] == job['sha256']
                cache._load(block)
                done += 1
                status('ready', last_block=job['key'], last_completed_at_unix=time.time(),
                       last_decode_seconds=time.monotonic()-started)
            except Exception as error:
                atomic(folder/'errors'/path.name, dict(job, error=repr(error), at_unix=time.time()))
                status('error', error=repr(error), block=job['key'])
                raise
            finally:
                path.unlink(missing_ok=True)
            idle = time.monotonic()
        status('stopped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--slot', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    worker(args.root, args.slot)
