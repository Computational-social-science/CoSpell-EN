import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

EXP_DIR = Path(__file__).resolve().parent
DATA = EXP_DIR.parent
OLD = EXP_DIR / 'empty_legacy'
# Only immutable model/runtime assets are reused, never ADV samples or caches.
PREVIOUS = Path('/path/to/cospell/Vocabulary/LongWord/ADV/parallel_corpus_expansion_20261013')
TARGET = DATA / 'organized_nouns_wordnet_20261018/final_adversarial_nouns_organized.json'
_manifest = EXP_DIR / 'baseline_manifest.json'
BASELINE_SHA = json.loads(_manifest.read_text())['baseline_sha256'] if _manifest.exists() else None
ASCII_LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def sid(*parts):
    return hashlib.sha256('\0'.join(map(str, parts)).encode()).hexdigest()

def read_json(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'); os.replace(temp, path)

def write_jsonl(path, rows):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    os.replace(temp, path)

def progress(stage, **values):
    obj = dict(stage=stage, updated_at_unix=time.time(), **values)
    write_json(EXP_DIR/f'status_{stage}.json',obj); print(json.dumps(obj),flush=True)

def pattern(word):
    return re.compile(r"(?<![\w'\u2019-])"+re.escape(word)+r"(?![\w'\u2019-])",re.I)

def surface_like(word, original):
    if original == original.lower(): return word.lower()
    if original.isupper(): return word.upper()
    if original == original.capitalize(): return word.capitalize()
    return None

def structural(sentence, word):
    matches=list(pattern(word).finditer(sentence))
    if len(matches)!=1: return None
    if not 4 <= len(sentence.split()) <= 60 or len(sentence)>1600: return None
    if re.search(r'https?://|www\.|\b\S+@\S+\.\S+|<\/?[A-Za-z][^>]*>|[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\ufeff]',sentence,re.I): return None
    if not re.search(r'[.!?]["\u201d\u2019\x27)]*\s*$',sentence): return None
    if sum(c.isalpha() and not c.isascii() for c in sentence)>max(2,sum(c.isalpha() for c in sentence)*.03):return None
    hit=matches[0]
    if surface_like(word,hit.group()) is None:return None
    return hit.span()

def pool(path):
    db=sqlite3.connect(path,timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, word TEXT, corpus TEXT, sentence_hash TEXT, sentence TEXT, provenance TEXT, UNIQUE(word,corpus,sentence_hash))')
    db.execute('CREATE INDEX IF NOT EXISTS sample_word ON samples(word,corpus)')
    db.execute('CREATE TABLE IF NOT EXISTS completed_files (path TEXT PRIMARY KEY, data TEXT)')
    return db

def config():return read_json(EXP_DIR/'expansion_config.json')

def attempt_limit(name):
    value=config().get(name)
    return float('inf') if value is None else value

def uncovered_words():
    with sqlite3.connect('file:'+str(EXP_DIR/'expansion_state.sqlite')+'?mode=ro',uri=True,timeout=120) as db:
        rows=db.execute('SELECT root,variant FROM pairs p WHERE p.existing=0 AND NOT EXISTS (SELECT 1 FROM accepted a WHERE a.record_id=p.id)').fetchall()
    return {r[0] for r in rows},{r[1] for r in rows}

def add_sample(db, word, sentence, provenance):
    sh=sid(sentence); key=sid(word,provenance['corpus'],sh)
    cur=db.execute('INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?)',
        (key,word,provenance['corpus'],sh,sentence,json.dumps(provenance,ensure_ascii=False)))
    return cur.rowcount
