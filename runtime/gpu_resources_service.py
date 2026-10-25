"""FIFO GPU leases shared by the three corpus experiments; no acceptance logic."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        if s[0]=='Z':return None
        return [p.stat().st_uid,int(s[19])]
    except (OSError,IndexError,ValueError):return None

def scheduling_policy(exp):
    p=Path(exp)/'scheduling_policy.json'
    if not p.exists():return {}
    value=json.loads(p.read_text())
    assert value['mode']=='shared_stage_queue'
    assert value['scheduler_directory']=='/path/to/cospell/Vocabulary/.shared_gpu_scheduler'
    proof=json.loads((Path(exp)/'scheduling_validation.json').read_text())
    assert proof['status']=='passed' and proof['policy_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in proof['runtime_sha256'].items():
        assert hashlib.sha256((Path(exp)/name).read_bytes()).hexdigest()==digest,('Unvalidated scheduling code',name)
    return value

def memory_snapshot():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    d={}
    for line in r.stdout.splitlines():
        i,u,total,free=[x.strip() for x in line.split(',')]
        d[int(i)]={'uuid':u,'total_mib':int(total),'free_mib':int(free),'compute_pids':[]}
    r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    for line in r.stdout.splitlines():
        u,pid=[x.strip() for x in line.split(',')]
        for v in d.values():
            if v['uuid']==u:v['compute_pids'].append(int(pid))
    return d

class LeaseUnavailable(Exception):
    """Nonblocking capacity probe could not obtain a FIFO GPU lease."""
    pass

class GpuLease:
    def __init__(self,exp,round_dir,stage,gpus,minimum=1,maximum=1,minimum_mib=2048,fraction=0,progress=None,scheduler=None,snapshot=None,wait=True):
        self.exp=Path(exp);self.round=Path(round_dir);self.stage=stage
        policy=scheduling_policy(exp) if scheduler is None else {}
        assert scheduler is not None or policy
        self.directory=Path(scheduler or policy['scheduler_directory'])
        self.directory.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.allowed=list(gpus);self.minimum=minimum;self.maximum=maximum
        self.minimum_mib=minimum_mib;self.fraction=fraction;self.progress=progress
        self.wait=wait
        self.snapshot=snapshot or memory_snapshot
        self.gpus=[];self.handles=[];self.record=None
        self.request=self.directory/(str(os.getpid())+'_'+str(time.time_ns())+'.request.json')
    @property
    def fds(self):return tuple(h.fileno() for h in self.handles)
    def __enter__(self):
        self.record={'pid':os.getpid(),'identity':identity(os.getpid()),'requested_at_ns':time.time_ns(),
                     'experiment':str(self.exp),'round':self.round.name,'stage':self.stage,
                     'minimum':self.minimum,'maximum':self.maximum,'eligible_gpus':self.allowed}
        atomic(self.request,self.record);last=0
        try:
            while True:
                with (self.directory/'queue.lock').open('a') as guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    pending=[]
                    for p in self.directory.glob('*.request.json'):
                        try:r=json.loads(p.read_text())
                        except FileNotFoundError:continue
                        if identity(r['pid'])!=r['identity']:
                            p.unlink(missing_ok=True);continue
                        pending.append((r['requested_at_ns'],p.name,r))
                    pending.sort()
                    if pending and pending[0][1]==self.request.name:
                        memory=self.snapshot()
                        candidates=[g for g in self.allowed if g in memory and not memory[g]['compute_pids']
                                    and memory[g]['free_mib']>=max(self.minimum_mib,memory[g]['total_mib']*self.fraction+256 if self.fraction else 0)]
                        for g in candidates:
                            h=(self.directory/('gpu'+str(g)+'.lock')).open('a')
                            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:h.close();continue
                            self.handles.append(h);self.gpus.append(g)
                            if len(self.gpus)==self.maximum:break
                        if len(self.gpus)>=self.minimum:
                            self.record.update(gpus=self.gpus,acquired_at_unix=time.time(),state='held')
                            atomic(self.exp/'gpu_stage_lease.json',self.record)
                            atomic(self.directory/(str(os.getpid())+'.lease.json'),self.record)
                            self.request.unlink()
                            return self
                        for h in self.handles:h.close()
                        self.handles=[];self.gpus=[]
                if not self.wait:raise LeaseUnavailable()
                if time.monotonic()-last>=10:
                    if self.progress:self.progress('filter',round=self.round.name,step='waiting_shared_gpu',next_step=self.stage,
                        requested_gpu_count=self.minimum,maximum_gpu_count=self.maximum,request=str(self.request),
                        waiting_seconds=round((time.time_ns()-self.record['requested_at_ns'])/1e9,1))
                    last=time.monotonic()
                time.sleep(1)
        except BaseException:
            self.request.unlink(missing_ok=True)
            for h in self.handles:h.close()
            raise
    def __exit__(self,*exc):
        # Model subprocesses inherit the locks; a surviving child keeps its GPU reserved.
        self.record.update(state='released',released_at_unix=time.time())
        atomic(self.exp/'gpu_stage_lease.json',self.record)
        atomic(self.directory/(str(os.getpid())+'.lease.json'),self.record)
        for h in self.handles:h.close()
        self.handles=[]
