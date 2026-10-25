import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

EXPERIMENT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('ADJ_RUN_DIR', str(EXPERIMENT)))
CORPUS_ID = 'short-adjective-streaming-20261015'


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def read_jsonl(name):
    with (BASE / name).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(name, records):
    target = BASE / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
    os.replace(temporary, target)


def write_json(name, data):
    target = BASE / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, target)


def progress(stage, **items):
    data = {'stage': stage, 'updated_at_unix': time.time(), **items}
    write_json('status_' + stage.split('_')[0] + os.environ.get('EXPERIMENT_STATUS_SUFFIX', '') + '.json', data)
    print(json.dumps(data), flush=True)


class Cache:
    def __init__(self, name, config):
        assert config.get('corpus_id', CORPUS_ID) == CORPUS_ID, 'Wrong corpus cache identity'
        self.db = sqlite3.connect(EXPERIMENT / (name + '.sqlite'), timeout=120)
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS outputs (key TEXT PRIMARY KEY,source TEXT,output TEXT)')
        # Even an accidentally copied cache cannot supply another POS experiment's decisions.
        expected = json.dumps({'corpus_id': CORPUS_ID, **config}, sort_keys=True)
        row = self.db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if row:
            if row[0] != expected:
                self.db.close()
                raise AssertionError('Cache configuration mismatch')
        else:
            self.db.execute('INSERT INTO metadata VALUES (?,?)', ('config', expected))
            self.db.commit()

    def get(self, text):
        key = hashlib.sha256(text.encode()).hexdigest()
        row = self.db.execute('SELECT source,output FROM outputs WHERE key=?', (key,)).fetchone()
        if row:
            assert row[0] == text
            return json.loads(row[1])
        return None

    def put(self, text, value):
        key = hashlib.sha256(text.encode()).hexdigest()
        self.db.execute('INSERT OR REPLACE INTO outputs VALUES (?,?,?)',
                        (key, text, json.dumps(value, ensure_ascii=False)))

    def flush(self):
        self.db.commit()


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
