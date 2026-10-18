"""Recover received responses after local interruption without resending them.

Original attempt metadata remains immutable; recovery gets a linked local ID.
"""
import argparse
import shutil
import sqlite3
from pathlib import Path
from llm_kt.artifacts import unique_id, utc_now, finish_stage
from llm_kt.budget import Budget
from llm_kt.common import canonical, digest, file_hash, read_json, write_json
from llm_kt.config import load_config
from llm_kt.providers import make_provider


def recover(config, root, budget):
    provider = make_provider(config['model'])
    db = sqlite3.connect(root / 'state.sqlite')
    recovered = []
    for path in sorted((root / 'attempts').glob('*.json')):
        evidence = read_json(path)
        sid = evidence['sample_id']
        job = db.execute('SELECT status FROM jobs WHERE sample_id=?',(sid,)).fetchone()
        if not job or job[0] == 'success' or evidence.get('recovered_from'):
            continue
        original = evidence['attempt_id']
        response_path = root / 'responses' / (original + '.json')
        if not response_path.exists():
            continue
        response = read_json(response_path)
        try:
            probability = provider.probability(response)
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        local_id = unique_id('recovery')
        for folder in ('requests','responses'):
            for source in (root / folder).glob(original + '.*'):
                shutil.copyfile(source, source.with_name(local_id + source.name[len(original):]))
        usage = response.get('usage') or {}
        result = {'sample_id':sid,'p_correct':probability,'status':'success',
                  'probability_source':config['model']['probability_strategy'],'model_requested':config['model']['model_id'],
                  'model_returned':response.get('model'),'provider':response.get('provider'),
                  'request_id':response.get('id',original),'local_request_id':local_id,'attempt_count':evidence['attempt_number'],
                  'request_hash':evidence['request_hash'],'usage':usage,'cost':usage.get('cost'),'latency_ms':None,
                  'recovered_from':original,'recovery_kind':'offline_response_parse_no_http'}
        linked = {**evidence,'attempt_id':local_id,'status':'success','recovered_from':original,
                  'original_evidence_sha256':file_hash(path),'recovered_at':utc_now(),'result_hash':digest(result),'files':{}}
        for folder in ('requests','responses'):
            for source in (root/folder).glob(local_id + '.*'):
                linked['files'][source.relative_to(root).as_posix()] = file_hash(source)
        write_json(root / 'attempts' / (local_id + '.json'),linked)
        db.execute("UPDATE jobs SET status='success',result=?,error=NULL WHERE sample_id=?",(canonical(result),sid))
        db.execute('INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)',(original,sid,'success',None,canonical(usage),None))
        budget.settle(original,usage.get('cost'))
        db.commit()
        recovered.append({'original_attempt':original,'recovery_id':local_id,'new_http_requests':0})
    db.close()
    folder=root/'recoveries'/unique_id('recovery')
    write_json(folder/'report.json',{'recovered':recovered,'at':utc_now()})
    finish_stage(folder,'offline_response_recovery',{'count':len(recovered)},['report.json'])
    print('offline_recovered',len(recovered),'new_http_requests',0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True);p.add_argument('--budget',required=True)
    a=p.parse_args();recover(load_config(a.config),Path(a.run),Budget(a.budget,'0.40'))
