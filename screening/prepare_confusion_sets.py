"""Per-experiment immutable CPU preparation; never shares acceptance caches."""
import fcntl,hashlib,json,os,pickle,time
from pathlib import Path

def signature(path):
    p=Path(path).resolve();s=p.stat()
    return [str(p),s.st_size,s.st_mtime_ns,s.st_ino]

def prepared(experiment,source,tokenizer,build):
    exp=Path(experiment).resolve();folder=exp/'rwse_cpu_preparation';folder.mkdir(exist_ok=True)
    model=exp/'models/bert-base-cased'
    inputs=[source,exp/'paper_confusion_sets_modified.csv',Path(build.__code__.co_filename)]
    inputs += sorted(p for p in model.iterdir() if p.suffix in ['.json','.txt'] and p.is_file())
    key={'experiment':str(exp),'inputs':[signature(p) for p in inputs],
         'tokenizer':type(tokenizer).__name__,'special_ids':tokenizer.all_special_ids}
    digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    target=folder/(digest+'.pickle');meta=folder/(digest+'.json')
    with (folder/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists() and meta.exists():
            proof=json.loads(meta.read_text());data=target.read_bytes()
            assert proof['key']==key and hashlib.sha256(data).hexdigest()==proof['sha256']
            return pickle.loads(data)
        started=time.monotonic();value=build();data=pickle.dumps(value,protocol=5)
        tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
        os.replace(tmp,target)
        proof={'key':key,'sha256':hashlib.sha256(data).hexdigest(),'seconds':time.monotonic()-started,'created_at_unix':time.time()}
        tmp=meta.with_suffix('.tmp');tmp.write_text(json.dumps(proof,indent=2));os.replace(tmp,meta)
        return value
