"""Re-iterable one-root-per-line JSON arrays; bounded frequency-data memory."""
import json, os
from pathlib import Path

class RootRecords:
    def __init__(self,path):self.path=Path(path);self.length=None
    def __iter__(self):
        # Small fixture arrays may use regular JSON formatting.
        if self.path.stat().st_size<5*1024*1024:
            yield from json.loads(self.path.read_text(encoding='utf-8'));return
        with self.path.open(encoding='utf-8') as f:
            assert next(f).strip()=='['
            closed=False
            for line in f:
                raw=line.strip()
                if raw==']':closed=True;break
                assert raw.startswith('{') and raw.rstrip(',').endswith('}'),'Expected compact one-root-per-line JSON'
                yield json.loads(raw.rstrip(','))
            assert closed and not f.read().strip()
    def __len__(self):
        if self.length is None:self.length=sum(1 for _ in self)
        return self.length

def write_roots(path,roots):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        f.write('[\n');first=True
        for root in roots:
            if not first:f.write(',\n')
            f.write(json.dumps(root,ensure_ascii=False,separators=(',',':')));first=False
        f.write('\n]\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
