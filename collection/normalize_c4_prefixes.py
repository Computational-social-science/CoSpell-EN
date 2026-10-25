"""One-time bounded replay for restored vocabulary, without rewinding main checkpoints."""
import multiprocessing as mp
import queue
import time
from expansion_common import EXP_DIR,config,read_json,write_json,progress,add_sample

def replay_new_vocabulary(db):
    marker=EXP_DIR/'c4_vocabulary_repair.json'
    if not marker.exists():return
    plan=read_json(marker)
    if plan.get('state')=='complete':return
    from collect_sources import init_worker,collect_file,disk_room
    checkpoints=read_json(EXP_DIR/'c4_checkpoints.json')
    assert checkpoints==plan['original_checkpoints'],'Primary C4 checkpoints changed during vocabulary repair'
    remaining={k:v for k,v in plan['prefixes'].items() if k not in plan.get('completed_prefixes',[])}
    ctx=mp.get_context('spawn');out=ctx.Queue(maxsize=config()['collector_queue_batches'])
    added=0;last=0
    with ctx.Pool(config()['c4_workers'],initializer=init_worker,initargs=(plan['added_words'],out)) as workers:
        jobs={path:workers.apply_async(collect_file,(path,0,end)) for path,end in remaining.items()};done={}
        while jobs:
            try:
                kind,path,rows,documents=out.get(timeout=1)
                if kind=='rows':
                    disk_room('c4_vocabulary_repair')
                    for word,sentence,source in rows:added+=add_sample(db,word,sentence,source)
                    db.commit()
                else:
                    assert kind=='done';done[path]=rows
            except queue.Empty:pass
            for path,result in list(jobs.items()):
                if result.ready() and path in done:
                    result.get();db.commit();plan.setdefault('completed_prefixes',[]).append(path)
                    write_json(marker,plan);del jobs[path]
            if time.monotonic()-last>10:
                progress('c4_vocabulary_repair',state='replaying_restored_vocabulary_only',remaining_prefixes=len(jobs),new_rows=added)
                last=time.monotonic()
    assert read_json(EXP_DIR/'c4_checkpoints.json')==checkpoints
    plan.update(state='complete',completed_at_unix=time.time(),added_rows_this_attempt=added,primary_checkpoints_unchanged=True)
    write_json(marker,plan)
    progress('c4_vocabulary_repair',state='complete',prefixes=len(plan['prefixes']),primary_checkpoints_unchanged=True)
