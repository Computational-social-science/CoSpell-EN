from review_policy import POLICY as MODEL_POLICY, required_stages, seed_evidence_files, verify_record
from scan_gate import selected_baseline, validate_record
"""Independently replay source spans and raw model acceptance before publication."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import sqlite3
from expansion_common import EXP_DIR, TARGET, DATA, PREVIOUS, BASELINE_SHA, digest, sid, read_json, read_jsonl, write_json, progress

def norm(text):return re.sub(r'\s+',' ',text).strip()

def verify_c4_file(job):
    path,requested,identity=job
    p=Path(path);assert p.stat().st_size==identity['bytes'] and p.stat().st_mtime_ns==identity['mtime_ns']
    by_line=defaultdict(list)
    for item in requested:by_line[item['source']['source_line']].append(item)
    remaining=set(by_line); verified=0
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line,raw in enumerate(f,1):
            if line not in remaining:continue
            doc=json.loads(raw);text=doc['text']
            for item in by_line[line]:
                s=item['source'];sentence=item['sentence']
                assert sid(text)==s['document_hash']
                assert doc.get('url')==s['url']
                assert text[s['sentence_start']:s['sentence_end']]==sentence
                assert sid(sentence)==s['source_sentence_hash']
                verified+=1
            remaining.remove(line)
            if not remaining:break
    assert not remaining,(path,remaining)
    return {'source_file':path,'verified_source_sentences':verified}

def verify_sources(records):
    unique={}
    for r in records:
        sentence=r['root_sentence'] if r.get('is_synthetic') or r['candidate_kind']=='seed' else r['variant_sentence']
        source=r['source'];assert sid(sentence)==source['source_sentence_hash']
        key=json.dumps([source,sentence],sort_keys=True,ensure_ascii=False)
        unique[key]={'source':source,'sentence':sentence,'word':r['root'] if r.get('is_synthetic') or r['candidate_kind']=='seed' else r['variant']}
    c4=defaultdict(list);bianka={};legacy=[]
    for item in unique.values():
        source=item['source'];corpus=source['corpus']
        if corpus=='c4_en_train':c4[source['source_file']].append(item)
        elif corpus=='bianka_root':bianka[(source['source_sid'],item['word'],item['sentence'])]=item
        elif corpus=='legacy_jsonl_root':legacy.append(item)
        else:raise AssertionError(corpus)
    raw_path=DATA/'goole_variants_sentences_adv.jsonl'
    assert digest(raw_path)=='668de71f77928a99b2f0997d98ad29f003c82013653600c0304256b088ad6118'
    raw=read_jsonl(raw_path)
    for item in legacy:
        s=item['source'];assert s['source_file']==str(raw_path)
        row=raw[s['source_line']-1]
        assert row['root']==row['variant']==item['word']
        assert row['sentences'][s['sentence_index']]==item['sentence']
    del raw
    inventory={i['path']:i for i in read_json(EXP_DIR/'c4_inventory.json')['files']}
    jobs=[(path,items,inventory[path]) for path,items in sorted(c4.items())]
    completed=[]
    if jobs:
        with mp.get_context('spawn').Pool(min(16,len(jobs))) as pool:
            for result in pool.imap_unordered(verify_c4_file,jobs):
                completed.append(result)
                progress('source_audit',step='c4',files_completed=len(completed),files_total=len(jobs))
    if bianka:
        paths={i['source']['source_file'] for i in bianka.values()};assert len(paths)==1
        path=Path(next(iter(paths)));meta=read_json(EXP_DIR/'bianka_complete.json')
        assert path.stat().st_size==meta['source_bytes'] and path.stat().st_mtime_ns==meta['source_mtime_ns']
        remaining=set(bianka);csv.field_size_limit(2000000)
        with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
            header=next(csv.reader([f.readline()],delimiter='\t'))
            for line,raw in enumerate(f,2):
                if '\x00' in raw or len(raw)>2000000:continue
                try:parts=next(csv.reader([raw],delimiter='\t',strict=True))
                except csv.Error:continue
                if len(parts)!=len(header):continue
                row=dict(zip(header,parts));key=(row['sid'],row['word'],row['sentence'])
                if key in remaining:
                    assert row.get('wid')==bianka[key]['source']['source_wid'];remaining.remove(key)
                    if not remaining:break
                if line%5000000==0:progress('source_audit',step='bianka',lines=line,remaining=len(remaining))
        assert not remaining,('Bianka sources missing',len(remaining))
    result={'unique_source_records_verified':len(unique),'c4_source_records':sum(len(v) for v in c4.values()),'c4_files':len(c4),
            'legacy_root_records':len(legacy),'bianka_source_records':len(bianka),'c4_file_results':sorted(completed,key=lambda r:r['source_file']),
            'bianka_identity':'source SID, word, WID and exact sentence; recorded line is supplementary'}
    write_json(EXP_DIR/'source_audit.json',result);return result

def verify_output(output,accepted_path=None):
    baseline_path=EXP_DIR/'baseline/final_adversarial_advs_organized.json'
    assert digest(baseline_path)==BASELINE_SHA
    baseline=read_json(baseline_path);baseline_report=read_json(EXP_DIR/'baseline/build_report.json')
    manifest=read_json(EXP_DIR/'bootstrap_complete.json')
    assert digest(EXP_DIR/'baseline/build_report.json')==manifest['baseline_report_sha256']
    for name,expected in manifest['previous_proofs'].items():assert digest(PREVIOUS/name)==expected
    proof=read_json(PREVIOUS/'publication_complete.json')
    assert proof['validation']=='passed' and proof['output_sha256']==BASELINE_SHA
    baseline_count=sum(len(v['parallel_corpus']) for r in baseline for v in r['variants'])
    assert baseline_count==manifest['baseline_sentence_pairs']==66169
    assert read_json(PREVIOUS/'experiment_report.json')['validation']['status']=='passed'
    accepted=read_jsonl(accepted_path or EXP_DIR/'accepted_records.jsonl')
    assert len(accepted)==len({r['sid'] for r in accepted})
    with sqlite3.connect('file:'+str(EXP_DIR/'expansion_state.sqlite')+'?mode=ro',uri=True) as db:
        registered={r[0]:json.loads(r[1]) for r in db.execute('SELECT sid,record FROM accepted')}
        seeds={r[0]:json.loads(r[1]) for r in db.execute('SELECT sid,record FROM seeds')}
    # A published, immutable batch remains valid while later batches enter the state DB.
    assert all(registered.get(r['sid'])==r for r in accepted)
    by_id={v['id']:(r['lemma'],v['word']) for r in baseline for v in r['variants']}
    targets={v['id'] for r in baseline for v in r['variants'] if not selected_baseline(EXP_DIR,v)}
    assert all(r['record_id'] in targets for r in accepted)
    assert len(accepted)==len({r['record_id'] for r in accepted})
    for r in accepted:
        if r['is_synthetic']:validate_record(r,EXP_DIR)
    needed_seeds={r['seed_id'] for r in accepted if r['is_synthetic']}
    assert needed_seeds<=seeds.keys()
    used_seeds=[seeds[key] for key in sorted(needed_seeds)]
    records=accepted+used_seeds
    round_data={};caches={}
    def cached(base,name,source):
        key=(str(base),name)
        if key not in caches:caches[key]=sqlite3.connect('file:'+str(base/(name+'.sqlite'))+'?mode=ro',uri=True)
        row=caches[key].execute('SELECT source,output FROM outputs WHERE key=?',(hashlib.sha256(source.encode()).hexdigest(),)).fetchone()
        assert row and row[0]==source,(name,source)
        return json.loads(row[1])
    def decisions(base,round_name):
        key=(str(base),round_name)
        if key not in round_data:
            folder=base/'rounds'/round_name
            round_data[key]={stage:{r['sid']:r for r in read_jsonl(folder/(stage+'_decisions_full.jsonl'))} for stage in ['gec','rwse','qwen']}
            round_data[key]['marker']=read_json(folder/'round_complete.json')
            round_data[key]['rwse_output']={r['sid']:r for r in read_jsonl(folder/'rwse_pass_full.jsonl')}
            round_data[key]['gec_report']=read_json(folder/'gec_report_full.json')
            f=folder/'synthetic_admission_decisions_full.jsonl'
            round_data[key]['synthetic_admission']={r['sid']:r for r in read_jsonl(f)} if f.exists() else {}
        return round_data[key]
    for r in records:
        seed=r['candidate_kind']=='seed';sentence=r['variant_sentence'];root=r['root_sentence']
        if seed:assert sentence==root and r['variant']==r['root']
        else:
            assert by_id[r['record_id']]==(r['root'],r['variant'])
            assert r['sid']==sid(r['root'],r['variant'],sentence)
            hits=list(re.finditer(r"(?<![\w'\u2019-])"+re.escape(r['variant'])+r"(?![\w'\u2019-])",sentence,re.I))
            if not r['is_synthetic']:assert len(hits)==1
            targets=[hit for hit in hits if hit.span()==(r['target_start'],r['target_end'])]
            assert len(targets)==1
            hit=targets[0];assert hit.group()==r['target_surface']
            assert sentence[:hit.start()]+r['replacement_surface']+sentence[hit.end():]==root and root!=sentence
            assert r['replacement_surface'].lower()==r['root']
            surface=r['target_surface']
            expected_surface=r['root'] if surface==surface.lower() else r['root'].capitalize() if surface==surface.capitalize() else r['root'].upper() if surface.isupper() else None
            assert r['replacement_surface']==expected_surface
            if r['is_synthetic']:
                s=seeds[r['seed_id']];assert root==s['root_sentence'] and r['source']==s['source'] and r['root']==s['root']
                assert r['source_target_start']==s['target_start'] and r['source_target_end']==s['target_end']
                assert sentence==root[:s['target_start']]+r['target_surface']+root[s['target_end']:]
                assert r['candidate_kind']=='synthetic'
            else:assert r['source']['corpus']=='c4_en_train' and r['candidate_kind']=='natural'
        assert r['language_score']>=.9
        base=Path(r.get('validation_experiment',str(EXP_DIR)))
        assert base in [EXP_DIR,PREVIOUS,DATA/'parallel_corpus_expansion_20261012']
        if not seed:assert base==EXP_DIR
        d=decisions(base,r['validated_in_round'])
        constructed=r.get('acceptance_policy')=='verified_root_direct_substitution_20261019'
        no_qwen=r.get('acceptance_policy')==MODEL_POLICY
        if no_qwen:verify_record(r,d['marker'],d['rwse_output'],d['qwen'])
        if not constructed:
            stages=['rwse'] if no_qwen else ['rwse','qwen']
            assert all(d[stage][r['sid']]['pass'] is True for stage in stages)
        if r.get('acceptance_policy') in ['verified_root_substitution_no_gec_20261019','verified_root_direct_substitution_20261019']:
            assert not seed and r['is_synthetic'] is True and r['candidate_kind']=='synthetic'
            assert r.get('gec_evaluated') is False and r['sid'] not in d['gec']
            admission=d['synthetic_admission'][r['sid']]
            assert admission['pass'] is True and admission['gec_evaluated'] is False
            assert admission['acceptance_policy']==r['acceptance_policy']
            assert admission['reason']=='validated_root_single_span_substitution'
            proof=r['root_validation'];s=seeds[r['seed_id']]
            assert admission['root_validation']==proof and proof['seed_id']==s['sid']
            seed_base=Path(s.get('validation_experiment',str(EXP_DIR)))
            assert Path(proof['experiment'])==seed_base and proof['round']==s['validated_in_round']
            required=seed_evidence_files(s)
            assert set(proof['sha256'])==required
            for name,expected in proof['sha256'].items():
                assert digest(seed_base/'rounds'/proof['round']/name)==expected
            # Every referenced seed is also replayed against raw model caches below.
            assert s in used_seeds
            if constructed:
                assert r.get('pair_origin')=='constructed_from_verified_root'
                for flag in ['gec_evaluated','rwse_evaluated','independent_review_evaluated']:
                    assert r.get(flag) is False and admission.get(flag) is False
                assert all(r['sid'] not in d[stage] for stage in ['gec','rwse','qwen'])
                folder=base/'rounds'/r['validated_in_round']
                stored={x['sid']:x for x in read_jsonl(folder/'synthetic_admission_pass_full.jsonl')}
                assert stored[r['sid']]=={k:v for k,v in r.items() if k!='validated_in_round'}
                # The unchanged seed is replayed against its own raw GEC, RWSE,
                # POS/root-review evidence in the same records loop below.
                # No claim is made that a model corrected this constructed variant.
                continue
        else:
            assert no_qwen or not r.get('acceptance_policy'), 'Unknown acceptance policy'
            assert d['gec'][r['sid']]['pass'] is True
            shards=len(d['gec_report']['shards']);shard=int(hashlib.sha256(sentence.encode()).hexdigest(),16)%shards
            for source in [sentence,root]:
                value=cached(base,f'gec_cache_shard_{shard}',source)
                assert not value.get('error') and value['truncated'] is False and norm(value['text'])==norm(root)
        value=cached(base,'rwse_cache',json.dumps([root,r['target_start'],len(r['replacement_surface'])]))
        assert not value.get('error') and all(f['log_probability_ratio']<math.log(10) for f in value['findings'])
        assert d['rwse'][r['sid']]['flags_by_mu']['10'] is False
        if no_qwen:continue
        if not seed:
            response=cached(base,'qwen_cache',json.dumps(['blind',sentence],ensure_ascii=False));b=response['parsed']
            assert response['finish_reason']=='stop' and norm(b['corrected_sentence'])==norm(root)
            assert b['is_complete_english_sentence'] is True and b['contains_spelling_error'] is True
            assert b['ambiguous'] is False and b['proper_name_or_foreign_expression_changed'] is False
            assert type(b['necessary_edit_count']) is int and b['necessary_edit_count']==1
        response=cached(base,'qwen_cache',json.dumps(['root',root,r['target_start'],r['replacement_surface']],ensure_ascii=False));b=response['parsed']
        assert response['finish_reason']=='stop' and norm(b['corrected_sentence'])==norm(root)
        assert b['is_complete_english_sentence'] is True and b['target_is_adverb'] is True
        assert b['ambiguous'] is False and b['has_any_spelling_or_grammar_error'] is False
    for conn in caches.values():conn.close()
    expected=defaultdict(list)
    for r in sorted(accepted,key=lambda r:r['sid']):expected[r['record_id']].append({'root_sentence':r['root_sentence'],'variant_sentence':r['variant_sentence']})
    assert len(output)==len(baseline)
    identities=set();ambiguities=defaultdict(set);count=0;covered=0
    for root,old in zip(output,baseline):
        assert {k:v for k,v in root.items() if k!='variants'}=={k:v for k,v in old.items() if k!='variants'}
        assert len(root['variants'])==len(old['variants'])
        for v,ov in zip(root['variants'],old['variants']):
            assert {k:x for k,x in v.items() if k!='parallel_corpus'}=={k:x for k,x in ov.items() if k!='parallel_corpus'}
            assert v['parallel_corpus']==selected_baseline(EXP_DIR,ov)+expected.get(v['id'],[])
            assert len(v['parallel_corpus'])<=1
            covered+=bool(v['parallel_corpus'])
            for pair in v['parallel_corpus']:
                assert set(pair)=={'root_sentence','variant_sentence'}
                key=(v['id'],pair['variant_sentence']);assert key not in identities;identities.add(key)
                ambiguities[(v['word'],pair['variant_sentence'])].add(root['lemma']);count+=1
    assert all(len(options)==1 for options in ambiguities.values())
    source_audit=verify_sources(records)
    return {'sentence_pairs_verified':count,'variants_with_corpus':covered,'baseline_sentence_pairs_archived':baseline_count,'baseline_pairs_retained':sum(len(selected_baseline(EXP_DIR,v)) for root in baseline for v in root['variants']),
            'new_sentence_pairs_verified':len(accepted),'used_root_seeds_verified':len(used_seeds),'synthetic_without_gec_verified':sum(r.get('acceptance_policy')=='verified_root_substitution_no_gec_20261019' for r in accepted),'constructed_from_verified_root_verified':sum(r.get('acceptance_policy')=='verified_root_direct_substitution_20261019' for r in accepted),'all_original_fields_unchanged':True,
            'baseline_validation':'SHA-256 anchored to previously validated publication; at most one previously qualified baseline pair selected per relation; natural and historical model-filtered origins preserved separately',
            'all_required_natural_and_seed_raw_model_decisions_rechecked':True,
            'gec_rwse_without_qwen_verified':sum(r.get('acceptance_policy')==MODEL_POLICY for r in accepted),
            'new_policy_qwen_and_contextual_pos_evaluated':False,
            'constructed_pairs_have_no_variant_model_decision_claim':True,'all_new_pairs_have_one_replacement_span':True,'all_new_source_sentences_exact':True,
            'source_audit':source_audit,'quality':'mixed natural model-filtered and explicitly constructed pairs; verified-root evidence is reused for constructed pairs; no claim of observed variant occurrence or unique semantic error'}
