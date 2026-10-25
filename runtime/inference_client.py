"""Prepare locally, then submit bounded FIFO inference jobs to warm model workers.

Only model weights are shared. Inputs, responses, quality decisions and caches
remain under the calling experiment. A cache hit never invokes this module.
"""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time,uuid
from pool_support import live_requests, review_demand, slot_folder

RUNTIME=Path('/path/to/cospell/Vocabulary/.shared_model_runtime_20261019')

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return None if s[0]=='Z' else [p.stat().st_uid,int(s[19])]
    except (OSError,IndexError):return None

def configuration(experiment):
    exp=Path(experiment).resolve();proof=read(exp/'scheduling_validation.json')['model_runtime_update']
    assert proof['policy']=='prepared_fifo_warm_models_20261019'
    assert digest(RUNTIME/'config.json')==proof['config_sha256']
    cfg=read(RUNTIME/'config.json')
    assert digest(cfg['worker'])==proof['worker_sha256']==cfg['worker_sha256']
    assert digest(Path(cfg['worker']).with_name('shared_gpu.py'))==cfg['scheduler_sha256']
    assert digest(RUNTIME/'pool_support.py')==digest(Path(__file__).with_name('pool_support.py'))==cfg['pool_support_sha256']
    assert str(exp) in cfg['experiments'] and os.getuid()==cfg['uid']==1005
    return exp,cfg,proof

def registered_worker_alive(old,proof):
    # A worker can exit between reading stat and cmdline; never accept a reused PID.
    for retry in range(3):
        if identity(old['pid']) != old['identity']:return False
        try:argv=[x.decode() for x in (Path('/proc')/str(old['pid'])/'cmdline').read_bytes().split(b'\0') if x]
        except FileNotFoundError:
            if identity(old['pid']) != old['identity']:return False
            raise
        if identity(old['pid']) != old['identity']:return False
        if not argv and retry < 2:
            time.sleep(.01);continue
        assert argv==old['argv'],('Shared worker argv mismatch',old['pid'],argv,old['argv'])
        assert old['worker_sha256']==proof['worker_sha256'],('Shared worker hash mismatch',old['pid'])
        return True
    return False

def ensure_slot(kind,cfg,proof,slot):
    folder=slot_folder(RUNTIME,kind,slot);folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (folder/'launch.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        p=folder/'worker.json';old=read(p) if p.exists() else None
        if old and registered_worker_alive(old,proof):
            return
        argv=[sys.executable,'-u',cfg['worker'],'--kind',kind,'--config',str(RUNTIME/'config.json'),'--config-sha',proof['config_sha256'],'--slot',str(slot)]
        with (folder/'worker.log').open('a') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        atomic(p,{'pid':child.pid,'identity':identity(child.pid),'argv':argv,'worker_sha256':proof['worker_sha256'],'started_at_unix':time.time()})

def ensure_worker(kind,cfg,proof):
    target=1
    if kind=='small' and not review_demand(RUNTIME):
        target=min(cfg['small_pool_size'],max(1,len(live_requests(RUNTIME/'small'))))
    for slot in range(target):ensure_slot(kind,cfg,proof,slot)

def submit(operation,payload,experiment=None,round_dir=None):
    exp=Path(experiment or Path(__file__).resolve().parent)
    exp,cfg,proof=configuration(exp);kind='review' if operation=='review' else 'small'
    queue=RUNTIME/kind/'queue';queue.mkdir(mode=0o700,parents=True,exist_ok=True)
    started=time.time();request_id=str(time.time_ns())+'_'+str(os.getpid())+'_'+uuid.uuid4().hex
    folder=exp/'model_requests'/request_id;folder.mkdir(mode=0o700,parents=True)
    input_file=folder/'input.json';response=folder/'response.json'
    atomic(input_file,{'operation':operation,'payload':payload})
    request={'request_id':request_id,'experiment':str(exp),'round':str(round_dir or ''),
        'pid':os.getpid(),'identity':identity(os.getpid()),'input':str(input_file),'input_sha256':digest(input_file),
        'response':str(response),'submitted_at_unix':started}
    deadline=time.monotonic()+cfg['request_timeout_seconds']
    while True:
        with (queue.parent/'enqueue.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            if len(list(queue.glob('*.json')))<cfg['queue_limit']:
                request['enqueued_at_ns']=time.time_ns()
                atomic(queue/(str(request['enqueued_at_ns'])+'_'+request_id+'.json'),request)
                break
        assert time.monotonic()<deadline,'Shared model queue timed out'
        ensure_worker(kind,cfg,proof);time.sleep(.2)
    ensure_worker(kind,cfg,proof)
    last=0
    while not response.exists():
        assert time.monotonic()<deadline,('Shared inference timed out',request_id)
        if time.monotonic()-last>5:
            ensure_worker(kind,cfg,proof)
            atomic(folder/'waiting.json',{'at_unix':time.time(),'kind':kind,'operation':operation,
                'wait_seconds':time.time()-started,'request_id':request_id})
            last=time.monotonic()
        time.sleep(.05)
    result=read(response)
    assert result['request_id']==request_id and result['input_sha256']==request['input_sha256']
    if result.get('error'):
        if result.get('error_type')=='OutOfMemoryError':
            import torch
            raise torch.cuda.OutOfMemoryError(result['error'])
        raise RuntimeError('Shared model worker failed: '+result['error'])
    return result['result']

def generate(inputs,round_dir):
    return submit('gec',{'inputs':{k:v.tolist() for k,v in inputs.items()}},round_dir=round_dir)

def masked_scores(inputs,tasks,ids,round_dir):
    columns=[list(dict.fromkeys([ids[t['word']]]+[ids[w] for w in t['alternatives']])) for t in tasks]
    result=submit('rwse',{'inputs':{k:v.tolist() for k,v in inputs.items()},'positions':[t['mask_position'] for t in tasks],'columns':columns},round_dir=round_dir)
    # Keep the original float32 subtraction, including its rounding at thresholds.
    import torch
    return [{i:v for i,v in zip(cols,torch.tensor(values,dtype=torch.float32))} for cols,values in zip(columns,result)]

def review_many(bodies,workers,round_dir):
    if not bodies:return []
    assert workers==64
    return submit('review',{'bodies':bodies,'workers':workers},round_dir=round_dir)
