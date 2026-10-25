"""UID-owned FIFO workers share weights, never experiment caches or decisions."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from shared_gpu import GpuLease,LeaseUnavailable,atomic,identity
from pool_support import live_requests, review_demand, slot_folder, claim, return_claim, YieldForReview

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def oldest_queued_small_wait(runtime,now=None):
    """Count only prepared requests still waiting, never in-flight GPU work."""
    now=time.time() if now is None else now
    ages=[]
    for path in (Path(runtime)/'small'/'queue').glob('*.json'):
        try:
            r=read(path)
            if identity(r['pid'])!=r['identity'] or Path(r['response']).exists():continue
            ages.append(max(0,now-r['enqueued_at_ns']/1e9))
        except FileNotFoundError:pass
    return max(ages) if ages else None

def idle_exit_reason(kind,cfg,idle,has_lease,stop_requested,small_wait=None):
    if stop_requested:return 'explicit_stop_when_idle'
    if kind=='review' and has_lease:
        # A busy small worker or a momentary gap alone is not a reason to
        # destroy a warm TP2 engine. Both conditions must remain satisfied.
        if idle>=cfg['review_warm_idle_seconds'] and small_wait is not None and small_wait>=cfg['review_small_queue_wait_seconds']:
            return 'review_idle_with_aged_small_queue'
        return None
    if idle>cfg['idle_seconds']:return 'idle_timeout'
    return None

def same_small_pool_request(request,runtime,cfg):
    if request.get('stage')!='resident_small':return False
    folder=Path(request.get('experiment',''))
    if folder not in [slot_folder(runtime,'small',slot) for slot in range(cfg['small_pool_size'])]:return False
    try:
        meta=read(folder/'worker.json')
        if request['pid']!=meta['pid'] or request['identity']!=meta['identity'] or identity(meta['pid'])!=meta['identity']:return False
        argv=[v.decode() for v in (Path('/proc')/str(meta['pid'])/'cmdline').read_bytes().split(b'\0') if v]
        return argv==meta['argv'] and meta['worker_sha256']==cfg['worker_sha256']
    except (FileNotFoundError,KeyError):return False

def validate_request(r,cfg):
    assert r['experiment'] in cfg['experiments']
    parent=Path(r['experiment']).resolve()/'model_requests'/r['request_id']
    assert Path(r['input']).resolve()==parent/'input.json'
    assert Path(r['response']).resolve()==parent/'response.json'
    assert sha(r['input'])==r['input_sha256']
    value=read(r['input']);op=value['operation'];p=value['payload']
    assert op in ['gec','rwse','review']
    if op in ['gec','rwse']:
        ids=p['inputs']['input_ids'];assert 0<len(ids)<=(128 if op=='gec' else 64)
        width=len(ids[0]);assert 0<width<=(256 if op=='gec' else 512)
        assert all(len(row)==width for rows in p['inputs'].values() for row in rows)
        assert all(len(rows)==len(ids) for rows in p['inputs'].values())
        if op=='rwse':assert len(p['positions'])==len(p['columns'])==len(ids)
    else:
        assert p['workers']==64 and 0<len(p['bodies'])<=8192
        for body in p['bodies']:
            assert body['model'] in cfg['reviewer_aliases']
            assert body['temperature']==0 and body['top_p']==1 and body['max_tokens']==512
            assert body['seed']==20261011 and body['response_format']=={'type':'json_object'}
            assert body['chat_template_kwargs']=={'enable_thinking':False}
    return value

class Worker:
    def __init__(self,kind,cfg,folder):
        self.kind=kind;self.cfg=cfg;self.folder=folder;self.lease=None;self.server=None
        self.models={};self.loads={};self.completed=0;self.last_work=time.monotonic();self.current=None;self.port=None;self.last_status=None;self.last_status_at=0
        self.release_reason=None;self.queued_small_wait=None
    def status(self,phase,**extra):
        if phase==self.last_status and not extra and time.monotonic()-self.last_status_at<10:return
        self.last_status=phase;self.last_status_at=time.monotonic()
        atomic(self.folder/'status.json',dict(phase=phase,pid=os.getpid(),identity=identity(os.getpid()),
            kind=self.kind,at_unix=time.time(),completed_requests=self.completed,model_loads=self.loads,
            gpus=self.lease.gpus if self.lease else [],current_request=self.current,
            idle_seconds=max(0,time.monotonic()-self.last_work),release_reason=self.release_reason,
            queued_small_wait_seconds=self.queued_small_wait,reuse_policy=self.cfg.get('review_reuse_policy'),**extra))
    def prepare_small(self):
        if self.models:return
        self.status('loading_models_on_cpu')
        import torch
        from transformers import AutoModelForSeq2SeqLM,AutoModelForMaskedLM
        torch.set_num_threads(8);torch.manual_seed(20261011);self.torch=torch
        self.models['gec']=AutoModelForSeq2SeqLM.from_pretrained(self.cfg['models']['gec'],torch_dtype=torch.bfloat16,local_files_only=True).eval()
        self.models['rwse']=AutoModelForMaskedLM.from_pretrained(self.cfg['models']['rwse'],torch_dtype=torch.float32,local_files_only=True).eval()
        self.loads={'gec':1,'rwse':1}
    def acquire(self,nonblocking=False):
        if self.lease:return
        if self.kind=='small':
            self.prepare_small()
            if getattr(self,'slot',0) and review_demand(self.runtime):raise YieldForReview()
        self.status('waiting_shared_gpu')
        def waiting(*args,**details):
            if getattr(self,'slot',0) and review_demand(self.runtime):raise YieldForReview()
            self.status('waiting_shared_gpu',queue_detail=details)
        self.lease=GpuLease(self.folder,self.folder,'resident_'+self.kind,self.cfg['gpus'],
            minimum=2 if self.kind=='review' else 1,maximum=2 if self.kind=='review' else 1,
            minimum_mib=0 if self.kind=='review' else 8192,fraction=.85 if self.kind=='review' else 0,
            scheduler=self.cfg['scheduler'],progress=waiting,wait=not nonblocking)
        try:self.lease.__enter__()
        except BaseException:
            self.lease=None
            raise
        if self.kind=='small':
            self.device='cuda:'+str(self.cfg['gpus'].index(self.lease.gpus[0]))
            self.status('moving_models_to_gpu')
            for model in self.models.values():model.to(self.device)
        else:self.start_reviewer()
    def start_reviewer(self):
        import requests
        with socket.socket() as s:s.bind(('127.0.0.1',0));self.port=s.getsockname()[1]
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',self.cfg['models']['review'],
            '--served-model-name',*self.cfg['reviewer_aliases'],'--host','127.0.0.1','--port',str(self.port),
            '--tensor-parallel-size','2','--max-model-len','2048','--max-num-seqs','64',
            '--gpu-memory-utilization','.85','--enforce-eager','--seed','20261011','--disable-log-requests']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,self.lease.gpus)),VLLM_WORKER_MULTIPROC_METHOD='spawn',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',OMP_NUM_THREADS='8')
        with (self.folder/'qwen_server.log').open('a') as log:
            self.server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=self.lease.fds)
        atomic(self.folder/'server.json',{'pid':self.server.pid,'identity':identity(self.server.pid),'argv':cmd,'port':self.port,'gpus':self.lease.gpus})
        started=time.monotonic();client=requests.Session();client.trust_env=False
        while time.monotonic()-started<1200:
            assert self.server.poll() is None,'Shared reviewer exited during loading'
            try:
                if client.get(f'http://127.0.0.1:{self.port}/health',timeout=2).status_code==200:break
            except requests.RequestException:pass
            self.status('reviewer_loading',startup_seconds=time.monotonic()-started);time.sleep(2)
        else:raise RuntimeError('Shared reviewer startup timed out')
        self.loads['review']=self.loads.get('review',0)+1
        self.status('reviewer_ready',startup_seconds=time.monotonic()-started,port=self.port)
    def infer(self,value):
        op=value['operation'];p=value['payload'];self.acquire();self.status('inference',operation=op)
        if op=='review':
            import requests
            def one(body):
                for attempt in range(3):
                    try:
                        with requests.Session() as s:
                            s.trust_env=False
                            r=s.post(f'http://127.0.0.1:{self.port}/v1/chat/completions',json=body,timeout=(10,240));r.raise_for_status();raw=r.json()
                            raw['choices'][0]['message']['content'];raw['choices'][0]['finish_reason']
                            return raw
                    except (requests.RequestException,ValueError,KeyError):
                        if attempt==2:raise
                        time.sleep(2*(attempt+1))
            with concurrent.futures.ThreadPoolExecutor(max_workers=p['workers']) as pool:return list(pool.map(one,p['bodies']))
        torch=self.torch;inputs={k:torch.tensor(v,dtype=torch.long,device=self.device) for k,v in p['inputs'].items()}
        with torch.inference_mode():
            if op=='gec':
                generated=self.models['gec'].generate(**inputs,max_new_tokens=320,do_sample=False,num_beams=1)
                return generated.tolist()
            predictions=self.models['rwse'](**inputs).logits
            scores=predictions[torch.arange(len(p['positions']),device=self.device),torch.tensor(p['positions'],device=self.device)].float().cpu()
            return [[float(row[i]) for i in cols] for row,cols in zip(scores,p['columns'])]
    def close(self):
        self.status('stopping')
        if self.server and self.server.poll() is None:
            config=read(self.folder/'server.json')
            assert identity(self.server.pid)==config['identity'] and os.getpgid(self.server.pid)==self.server.pid
            assert [v.decode() for v in (Path('/proc')/str(self.server.pid)/'cmdline').read_bytes().split(b'\0') if v]==config['argv']
            os.killpg(self.server.pid,signal.SIGTERM)
            self.server.wait(timeout=180)
            # Do not release the parent lease while an inherited child remains.
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                live=[]
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit():continue
                    try:
                        st=(p/'stat').read_text().rsplit(')',1)[1].split()
                        if st[0]!='Z' and int(st[2])==self.server.pid:live.append(p.name)
                    except OSError:pass
                if not live:break
                time.sleep(1)
            assert not live,'Reviewer child still owns inherited lease'
        if self.models:
            self.models.clear();self.torch.cuda.empty_cache()
        if self.lease and self.lease.record:self.lease.__exit__(None,None,None)
        self.lease=None
        self.status('stopped')

def reply(request,**result):
    atomic(Path(request['response']),dict(request_id=request['request_id'],input_sha256=request['input_sha256'],**result))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['small','review'],required=True);parser.add_argument('--config',required=True);parser.add_argument('--config-sha',required=True);parser.add_argument('--slot',type=int,default=0);a=parser.parse_args()
    assert sha(a.config)==a.config_sha;cfg=read(a.config)
    assert os.getuid()==cfg['uid']==1005 and sha(__file__)==cfg['worker_sha256']
    assert sha(Path(__file__).with_name('shared_gpu.py'))==cfg['scheduler_sha256']
    assert sha(Path(__file__).with_name('pool_support.py'))==cfg['pool_support_sha256']
    assert 0<=a.slot<(cfg['small_pool_size'] if a.kind=='small' else 1)
    assert cfg['review_warm_idle_seconds']>=120 and cfg['review_small_queue_wait_seconds']>=15
    runtime=Path(a.config).parent;shared=runtime/a.kind
    folder=slot_folder(runtime,a.kind,a.slot);folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    queue=shared/'queue';working=shared/'processing'
    queue.mkdir(mode=0o700,parents=True,exist_ok=True);working.mkdir(mode=0o700,exist_ok=True)
    with (folder/'instance.lock').open('a') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        worker=Worker(a.kind,cfg,folder);worker.slot=a.slot;worker.runtime=runtime;worker.status('ready_on_cpu')
        try:
            while True:
                # Extra small workers yield at request boundaries, even with a
                # continuous GEC backlog. The primary can keep the spare card.
                if a.slot and review_demand(runtime):
                    worker.status('yielding_for_review');break
                # Do not strand FIFO work behind a third worker waiting for a GPU.
                # CPU preparation remains before the lease; inference stays unchanged.
                if a.kind=='small' and worker.lease is None and live_requests(shared):
                    try:worker.acquire(nonblocking=True)
                    except LeaseUnavailable:
                        worker.status('waiting_pool_capacity');time.sleep(.2);continue
                selected=claim(shared,a.slot,reply)
                if selected is None:
                    worker.current=None
                    worker.queued_small_wait=oldest_queued_small_wait(runtime) if a.kind=='review' else None
                    worker.release_reason=idle_exit_reason(a.kind,cfg,time.monotonic()-worker.last_work,
                        worker.lease is not None,(shared/'stop_when_idle').exists(),worker.queued_small_wait)
                    if worker.release_reason:break
                    # Yield unused warm capacity when another live stage needs GPUs.
                    # Review has its own conjunction above; this old fast-yield
                    # path must not bypass its warm interval or backlog check.
                    if a.kind!='review' and worker.lease and time.monotonic()-worker.last_work>5:
                        waiting=[]
                        for request_path in Path(cfg['scheduler']).glob('*.request.json'):
                            try:
                                pending=read(request_path)
                                if identity(pending['pid'])==pending['identity'] and not same_small_pool_request(pending,runtime,cfg):waiting.append(pending)
                            except FileNotFoundError:pass
                        if waiting:break
                    worker.status('warm_idle' if worker.lease else 'ready_on_cpu');time.sleep(.2);continue
                dest,r=selected;worker.current={k:r[k] for k in ['request_id','experiment','round','submitted_at_unix']}
                started=time.time()
                returned=False
                try:
                    value=validate_request(r,cfg)
                    assert (value['operation']=='review')==(a.kind=='review')
                    result=worker.infer(value)
                    reply(r,result=result,worker_pid=os.getpid(),worker_identity=identity(os.getpid()),worker_slot=a.slot,gpus=worker.lease.gpus,started_at_unix=started,completed_at_unix=time.time(),model_loads=worker.loads)
                except YieldForReview:
                    return_claim(shared,dest);returned=True
                    worker.status('yielding_for_review');break
                except Exception as exc:
                    traceback.print_exc()
                    reply(r,error=str(exc),error_type=type(exc).__name__,worker_pid=os.getpid())
                    if type(exc).__name__=='OutOfMemoryError' and a.kind=='small':worker.torch.cuda.empty_cache()
                    else:raise
                finally:
                    if not returned:dest.unlink(missing_ok=True)
                worker.completed+=1;worker.last_work=time.monotonic();worker.status('request_complete')
        finally:worker.close()

if __name__=='__main__':main()
