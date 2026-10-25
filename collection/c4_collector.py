"""Preserve primary checkpoints; replay reopened words only in bounded prefixes.

Source rows commit before their checkpoint. Eight required files reach EOF
before additional files are scheduled; no model or extraction filter changes.
"""
from pathlib import Path
import json
import multiprocessing as mp
import queue
import time
from expansion_common import EXP_DIR, config, pool, read_json, write_json, progress, add_sample
from scan_gate import EPOCH, digest

def run():
    from collect_sources import C4, Feedback, init_worker, collect_file, disk_room
    control = config(); plan_path = EXP_DIR/'c4_scan_gate_plan.json'; plan = read_json(plan_path)
    assert plan['policy'] == EPOCH and len(plan['required_files']) == 8
    feedback = Feedback(); words = feedback.roots | feedback.variants
    db = pool(EXP_DIR/'c4_pool.sqlite')
    db.execute('CREATE INDEX IF NOT EXISTS sample_word_hash ON samples(word,sentence_hash)'); db.commit()
    checkpoints_path = EXP_DIR/'c4_checkpoints.json'
    checkpoints = read_json(checkpoints_path)
    replay_path = EXP_DIR/'c4_scan_gate_prefix_replay.json'
    replay = read_json(replay_path)
    inventory = read_json(EXP_DIR/'c4_inventory.json')['files']
    required = {f['path'] for f in plan['required_files']}
    for f in plan['required_files']:
        st = Path(f['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (f['bytes'], f['mtime_ns'])
    complete = {p:json.loads(raw) for p,raw in db.execute('SELECT path,data FROM completed_files')}
    initial = db.execute('SELECT count(*) FROM samples').fetchone()[0]
    added = 0; started = time.monotonic(); last_status = 0
    ctx = mp.get_context('spawn')

    def checkpoint(path, line):
        old = checkpoints.get(path, {}).get('committed_line', 0)
        assert line >= old, 'C4 primary checkpoint regression'
        st = Path(path).stat()
        checkpoints[path] = {'committed_line':line,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
        write_json(checkpoints_path, checkpoints)

    def certificate():
        p = EXP_DIR/'c4_scan_gate_complete.json'
        if p.exists() or replay['state'] != 'complete':return
        if not all(f['path'] in complete and complete[f['path']].get('scanned_to_eof') is True for f in plan['required_files']):return
        proof = {'policy':EPOCH,'plan_sha256':digest(plan_path),
                 'prefix_replay_complete':True,'prefix_replay_sha256':digest(replay_path),
                 'files':[dict(complete[f['path']],**f) for f in plan['required_files']],
                 'c4_high_rowid':db.execute('SELECT coalesce(max(rowid),0) FROM samples').fetchone()[0],
                 'completed_at_unix':time.time()}
        write_json(p, proof)

    def process(task_map, worker_words, is_replay):
        nonlocal added,last_status
        out=ctx.Queue(maxsize=control['collector_queue_batches'])
        with ctx.Pool(control['c4_workers'],initializer=init_worker,initargs=(sorted(worker_words),out)) as workers:
            pending=iter(task_map.items()); jobs={};done={}
            def launch():
                while len(jobs)<control['c4_workers']:
                    task=next(pending,None)
                    if task is None:return
                    path,(resume,end)=task
                    jobs[path]=workers.apply_async(collect_file,(path,resume,end))
            launch()
            while jobs:
                try:
                    kind,path,data,documents=out.get(timeout=1)
                    if kind=='rows':
                        disk_room('c4');feedback.refresh();active=feedback.active
                        for word,sentence,source in data:
                            if word in active:added+=add_sample(db,word,sentence,source)
                        db.commit()
                    elif kind=='checkpoint':
                        if not is_replay:checkpoint(path,data)
                    else:
                        assert kind=='done';done[path]=data
                except queue.Empty:pass
                for path,result in list(jobs.items()):
                    if not result.ready():continue
                    source,stats=result.get();assert source==path
                    if path not in done:continue
                    assert done.pop(path)==stats
                    db.commit()
                    if is_replay:
                        assert stats['final_line']>=task_map[path][1], 'Incomplete prefix replay'
                        replay.setdefault('completed_prefixes',[]).append(path);write_json(replay_path,replay)
                    else:
                        assert stats['complete_file'] and stats['scanned_to_eof'], 'Shard did not reach EOF'
                        checkpoint(path,stats['final_line'])
                        db.execute('INSERT OR REPLACE INTO completed_files VALUES (?,?)',(path,json.dumps(stats)));db.commit()
                        complete[path]=stats;certificate()
                    del jobs[path]
                if time.monotonic()-last_status>=10:
                    progress('c4',state='replaying_reopened_word_prefixes' if is_replay else 'streaming',
                             required_complete_shards=8,completed_required_shards=len(required & complete.keys()),
                             fallback_enabled=(EXP_DIR/'c4_scan_gate_complete.json').exists(),
                             active_shards=len(jobs),collected_sentences=initial+added,
                             primary_checkpoints_preserved=True,elapsed_seconds=time.monotonic()-started)
                    last_status=time.monotonic()
                launch()
        out.close();out.join_thread()

    if replay['state']!='complete':
        tasks={p:(0,end) for p,end in replay['prefixes'].items() if end>0 and p not in replay.get('completed_prefixes',[])}
        if tasks and replay['words']:process(tasks,set(replay['words']),True)
        replay.update(state='complete',completed_at_unix=time.time());write_json(replay_path,replay)
    # The first eight are a frozen random permutation prefix, not eight workers.
    tasks={f['path']:(None,None) for f in plan['required_files'] if not complete.get(f['path'],{}).get('scanned_to_eof')}
    if tasks:process(tasks,words,False)
    certificate();assert (EXP_DIR/'c4_scan_gate_complete.json').exists()
    # Keep the existing wider scan policy after satisfying the minimum scope.
    for offset in range(0,len(inventory),control['c4_workers']):
        feedback.refresh(force=True)
        if not feedback.active:break
        tasks={f['path']:(None,None) for f in inventory[offset:offset+control['c4_workers']]
               if f['path'] not in complete}
        if tasks:process(tasks,words,False)
    feedback.refresh(force=True)
    total=db.execute('SELECT count(*) FROM samples').fetchone()[0];db.close()
    result={'state':'all_train_files_rescanned' if len(complete)==len(inventory) else 'coverage_goal_reached',
            'completed_files':len(complete),'available_train_files':len(inventory),'samples':total,
            'required_complete_shards':8,'scan_gate_certificate':str(EXP_DIR/'c4_scan_gate_complete.json'),
            'partial_files':[],'elapsed_seconds':time.monotonic()-started}
    assert len(complete)==len(inventory) or not feedback.active
    write_json(EXP_DIR/'c4_complete.json',result);progress('c4',**result)
