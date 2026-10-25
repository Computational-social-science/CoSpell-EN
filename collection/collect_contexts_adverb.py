"""Stream exact source spans to durable pools while independent model filtering runs."""
import argparse
from collections import Counter
import csv
import gzip
import multiprocessing as mp
import queue
import random
import shutil
import time
from expansion_common import *

C4=Path('/data/liujunhui/c4/en')
BIANKA=Path('/data1/bianka/full_inv_natural_from_cloud/sampling_tools_v2/out/trial.k10000.seed2026.sent.tsv.gz')
OUT=None
AUTOMATON=None
NLP=None
FEEDBACK=None


class Feedback:
    def __init__(self):
        self.checked=0;self.roots=set();self.variants=set();self.refresh(force=True)

    def refresh(self,force=False):
        if force or time.monotonic()-self.checked>=15:
            value=read_json(EXP_DIR/'coverage_feedback.json')
            self.roots=set(value['roots']);self.variants=set(value['variants'])
            self.active=self.roots|self.variants
            self.checked=time.monotonic()
        return self.roots,self.variants


def word_char(char):return char.isalnum() or char in "_'\u2019-"


def init_worker(words,out):
    global OUT,AUTOMATON,NLP,FEEDBACK
    import ahocorasick
    import spacy
    OUT=out;AUTOMATON=ahocorasick.Automaton()
    for word in words:AUTOMATON.add_word(word,word)
    AUTOMATON.make_automaton()
    NLP=spacy.blank('en');NLP.add_pipe('sentencizer');NLP.max_length=2000000
    FEEDBACK=Feedback()


def collect_file(path, resume_override=None, last_line=None):
    import orjson
    path=Path(path);rows=[];documents=0;errors=Counter();last=time.monotonic();count=0;finished=True
    batch_limit=config()['collector_batch_rows']
    checkpoints=read_json(EXP_DIR/'c4_checkpoints.json')
    checkpoint=checkpoints.get(str(path),{})
    resume_line=checkpoint.get('committed_line',0) if resume_override is None else resume_override
    if checkpoint:
        st=path.stat()
        assert (checkpoint['bytes'],checkpoint['mtime_ns'])==(st.st_size,st.st_mtime_ns)
    line_no=resume_line;last_checkpoint=time.monotonic()
    with gzip.open(path,'rb') as stream:
        for line_no,line in enumerate(stream,1):
            if last_line is not None and line_no>last_line:finished=False;break
            if line_no<=resume_line:continue
            FEEDBACK.refresh();active=FEEDBACK.active
            # Even with no uncovered words, required shards must reach EOF.
            # Empty active vocabulary skips extraction, not source completion.
            try:record=orjson.loads(line)
            except orjson.JSONDecodeError:errors['json']+=1;continue
            text=record.get('text','');documents+=1
            if not text or len(text)>2000000:errors['empty_or_oversized_document']+=1;continue
            lower=text.translate(ASCII_LOWER)
            matched={word for end,word in AUTOMATON.iter(lower) if word in active
                     and (end+1==len(text) or not word_char(text[end+1]))
                     and (end-len(word)<0 or not word_char(text[end-len(word)]))}
            if matched:
                document_hash=sid(text)
                for span in NLP(text).sents:
                    raw=span.text;sentence=raw.strip()
                    if not 4<=len(sentence.split())<=60:continue
                    start=span.start_char+len(raw)-len(raw.lstrip());end=start+len(sentence)
                    assert text[start:end]==sentence
                    words={word for _,word in AUTOMATON.iter(sentence.translate(ASCII_LOWER)) if word in matched}
                    for word in sorted(words):
                        if structural(sentence,word) is None:continue
                        rows.append((word,sentence,{'corpus':'c4_en_train','source_file':str(path),'source_line':line_no,
                            'document_hash':document_hash,'url':record.get('url'),'sentence_start':start,'sentence_end':end,
                            'source_sentence_hash':sid(sentence),'extraction':'spaCy blank English sentencizer; every eligible exact sentence span; no per-document or per-file word cap'}))
                        count+=1
            if rows and (len(rows)>=batch_limit or time.monotonic()-last>=5):
                OUT.put(('rows',str(path),rows,documents));rows=[];last=time.monotonic()
                OUT.put(('checkpoint',str(path),line_no,documents));last_checkpoint=time.monotonic()
            if time.monotonic()-last_checkpoint>=5:
                if rows:OUT.put(('rows',str(path),rows,documents));rows=[]
                OUT.put(('checkpoint',str(path),line_no,documents));last_checkpoint=time.monotonic()
    if rows:OUT.put(('rows',str(path),rows,documents))
    stats={'documents':documents,'candidate_rows_emitted':count,'errors':dict(errors),
           'complete_file':finished,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
    stats.update(final_line=min(line_no,last_line) if last_line is not None else line_no,
                 scanned_to_eof=finished,resumed_after_line=resume_line)
    OUT.put(('done',str(path),stats,documents))
    return str(path),stats


def disk_room(stage):
    minimum=config()['minimum_free_disk_gib']*1024**3
    while shutil.disk_usage(EXP_DIR).free<minimum:
        progress(stage,state='waiting_for_disk_space',minimum_free_gib=config()['minimum_free_disk_gib'])
        time.sleep(30)


def c4():
    from c4_gate_collector import run
    return run()

def legacy_c4_unused():
    control=config();feedback=Feedback();roots,variants=feedback.refresh();words=roots|variants
    db=pool(EXP_DIR/'c4_pool.sqlite')
    db.execute('CREATE INDEX IF NOT EXISTS sample_word_hash ON samples(word,sentence_hash)');db.commit()
    files=sorted(C4.glob('c4-train*.json.gz'));random.Random(control['c4_file_shuffle_seed']).shuffle(files)
    inventory=[{'path':str(f),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in files]
    write_json(EXP_DIR/'c4_inventory.json',{'split':'train','files':inventory,'available_train_files':len(files),
               'sampling':'streaming coverage feedback; no per-word or per-file cap','workers':control['c4_workers']})
    complete={r[0] for r in db.execute('SELECT path FROM completed_files')}
    pending=iter([f for f in files if str(f) not in complete]);initial=db.execute('SELECT count(*) FROM samples').fetchone()[0]
    bootstrap=EXP_DIR/'bootstrap_complete.json'
    baseline_rows=read_json(bootstrap)['pools']['c4'] if bootstrap.exists() else initial
    total=initial;added=0;last_status=0;started=time.monotonic();partial=[]
    ctx=mp.get_context('spawn');out=ctx.Queue(maxsize=control['collector_queue_batches'])
    with ctx.Pool(control['c4_workers'],initializer=init_worker,initargs=(sorted(words),out)) as workers:
        jobs={};finished_messages={}
        def launch():
            while len(jobs)<control['c4_workers']:
                roots,variants=feedback.refresh()
                if not roots and not variants:return
                path=next(pending,None)
                if path is None:return
                jobs[str(path)]=workers.apply_async(collect_file,(str(path),))
        launch()
        while jobs:
            try:
                kind,path,rows,documents=out.get(timeout=1)
                if kind=='done':
                    finished_messages[path]=rows
                else:
                    assert kind=='rows';disk_room('c4')
                    feedback.refresh();active=feedback.active
                    for word,sentence,source in rows:
                        if word in active:added+=add_sample(db,word,sentence,source)
                    db.commit();total=initial+added
            except queue.Empty:pass
            for path,result in list(jobs.items()):
                if not result.ready():continue
                source_path,stats=result.get();assert source_path==path
                if path not in finished_messages:continue
                assert finished_messages.pop(path)==stats
                # Per-producer queue ordering guarantees every preceding batch is committed.
                if stats['complete_file']:
                    db.execute('INSERT OR REPLACE INTO completed_files VALUES (?,?)',(path,json.dumps(stats)));complete.add(path)
                else:partial.append({'path':path,**stats})
                db.commit();del jobs[path]
            if time.monotonic()-last_status>=10:
                progress('c4',state='streaming',completed_files=len(complete),total_files=len(files),
                         inherited_pool_rows=baseline_rows,new_candidate_rows=total-baseline_rows,collected_sentences=total,
                         active_shards=len(jobs),active_words=len(feedback.roots|feedback.variants),elapsed_seconds=round(time.monotonic()-started,1))
                last_status=time.monotonic()
            launch()
        db.commit()
    total=db.execute('SELECT count(*) FROM samples').fetchone()[0];db.close()
    roots,variants=feedback.refresh(force=True)
    result={'state':'all_train_files_rescanned' if len(complete)==len(files) else 'coverage_goal_reached',
            'completed_files':len(complete),'available_train_files':len(files),'partial_files':partial,
            'inherited_pool_rows':baseline_rows,'new_candidate_rows':total-baseline_rows,'samples':total,
            'no_fixed_word_pool_cap':True,'elapsed_seconds':time.monotonic()-started}
    assert len(complete)==len(files) or not roots and not variants
    write_json(EXP_DIR/'c4_complete.json',result);progress('c4',**result)


def bianka():
    control=config();feedback=Feedback();db=pool(EXP_DIR/'bianka_pool.sqlite')
    db.execute('CREATE INDEX IF NOT EXISTS sample_word_hash ON samples(word,sentence_hash)');db.commit()
    if (EXP_DIR/'bianka_complete.json').exists():return
    total=db.execute('SELECT count(*) FROM samples').fetchone()[0];added=0;malformed=Counter();last_commit=time.monotonic();last_status=0
    started=time.monotonic();line=1;finished=True;csv.field_size_limit(2000000)
    with gzip.open(BIANKA,'rt',encoding='utf-8',newline='') as stream:
        header=next(csv.reader([stream.readline()],delimiter='\t'));assert header==['wid','word','df','sid','sentence']
        for line,raw in enumerate(stream,2):
            roots,_=feedback.refresh()
            if not roots:finished=False;break
            # The word is an ASCII clean lemma; this prefilter avoids parsing unrelated rows.
            fields=raw.split('\t',2)
            if len(fields)<3 or fields[1].strip('"') not in roots:continue
            if '\x00' in raw or len(raw)>2000000:malformed['nul_or_oversized_row']+=1;continue
            try:parts=next(csv.reader([raw],delimiter='\t',strict=True))
            except csv.Error:malformed['invalid_tsv_row']+=1;continue
            if len(parts)!=len(header):malformed['wrong_column_count']+=1;continue
            row=dict(zip(header,parts));word=row['word'];sentence=row['sentence']
            if word in roots and structural(sentence,word) is not None:
                added+=add_sample(db,word,sentence,{'corpus':'bianka_root','source_file':str(BIANKA),'source_line':line,
                    'source_sid':row['sid'],'source_wid':row['wid'],'source_sentence_hash':sid(sentence)})
            if time.monotonic()-last_commit>=control['collector_commit_seconds']:
                db.commit();disk_room('bianka');last_commit=time.monotonic()
            if time.monotonic()-last_status>=15:
                progress('bianka',state='streaming',lines=line,collected_sentences=total+added,
                         remaining_roots=len(roots),elapsed_seconds=round(time.monotonic()-started,1));last_status=time.monotonic()
    db.commit();total=db.execute('SELECT count(*) FROM samples').fetchone()[0];db.close()
    result={'state':'source_scan_complete' if finished else 'coverage_goal_reached','lines':line,'collected_sentences':total,
            'malformed_matching_rows':dict(malformed),'source_bytes':BIANKA.stat().st_size,'source_mtime_ns':BIANKA.stat().st_mtime_ns,
            'no_fixed_word_pool_cap':True,'elapsed_seconds':time.monotonic()-started}
    write_json(EXP_DIR/'bianka_complete.json',result);progress('bianka',**result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',choices=['c4','bianka']);args=parser.parse_args()
    try:globals()[args.source]()
    except Exception as exc:progress(args.source,state='failed',error=repr(exc));raise
