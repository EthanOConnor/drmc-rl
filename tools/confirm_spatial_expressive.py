"""Evaluate fixed spatial proposal heads on entirely reserved replay identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.human.spatial_proposer import SCHEMA, SpatialProposer
from tools.build_expressive_sequences import write_progress
from tools.fit_spatial_expressive import compare_arms, evaluate, prepare_data, sha256


def source_identities(path):
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
    return {str(r['session']) for r in metadata['sources']},{r['sha256'] for r in metadata['sources']}


def run(config):
    study_path=Path(config['study'])
    study=json.loads(study_path.read_text())
    if study.get('status')!='Complete' or study.get('schema')!=SCHEMA:
        raise ValueError('completed fixed spatial study required')
    earlier=Path(study['config']['source'])
    checkpoint=Path(study['config']['checkpoint'])
    if sha256(earlier)!=study['source_sha256'] or sha256(checkpoint)!=study['competitive_sha256']:
        raise ValueError('frozen source or competitive checkpoint changed')
    old_sessions,old_blobs=source_identities(earlier)
    fresh_sessions,fresh_blobs=source_identities(config['source'])
    if old_sessions & fresh_sessions or old_blobs & fresh_blobs:
        raise ValueError('confirmation reuses development sessions or replay content')
    output=Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(int(config.get('threads',1)))
    started=time.monotonic()
    report=dict(schema='drmc-spatial-expressive-confirmation-v1',status='Running',config=config,
        source_sha256=sha256(config['source']),competitive_sha256=study['competitive_sha256'],
        study_sha256=sha256(study_path),train_source_sha256=study['source_sha256'],
        excluded_development_sessions=len(old_sessions),confirmation_sessions=len(fresh_sessions),
        session_overlap=0,blob_overlap=0,console_frames_trained=0,optimizer_updates=0,
        action_presentations=0,diagnostic_only=True,quality_admission=False,arms={})
    write_progress(output,report)
    try:
        data=prepare_data({**config,'checkpoint':str(checkpoint)},output,report)
        ids=np.arange(len(data['windows']))
        report['evaluated_sessions']=len(set(data['windows'][:,3]))
        device=config.get('device','cpu')
        for name in ('persistent','stateless'):
            path=study_path.parent/(name+'-final.pt')
            if sha256(path)!=study['arms'][name]['checkpoint_sha256']:
                raise ValueError('fixed-final proposal checkpoint changed')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if (saved['schema']!=SCHEMA or saved['source_sha256']!=study['source_sha256']
                    or saved['competitive_sha256']!=study['competitive_sha256']
                    or saved['feature_dim']!=data['features'].shape[1]
                    or saved['persistent']!=(name=='persistent')):
                raise ValueError('proposal checkpoint contract differs from the completed study')
            model=SpatialProposer(saved['feature_dim'],saved['width'],persistent=saved['persistent']).to(device)
            model.load_state_dict(saved['state_dict'],strict=True)
            model.eval().requires_grad_(False)
            priors={k:v.to(device) for k,v in saved['training_priors'].items()}
            report.update(phase='evaluating',current_arm=name)
            write_progress(output,report)
            metrics=evaluate(model,data,ids,device,int(config.get('batch_windows',32)),saved['persistent'],priors)
            report['arms'][name]=dict(status='Complete',checkpoint_sha256=sha256(path),final=metrics,
                evaluated_windows=len(ids),evaluated_actions=int(data['windows'][:,1].sum()))
            write_progress(output,report)
        compare_arms(report)
        report.update(status='Complete',phase='complete',elapsed_seconds=time.monotonic()-started,
            comparison_scope='Fresh whole-session and replay-content-disjoint recorded-prefix confirmation. No optimizer, model selection or new training. Individual paired intervals; actual quality-admitted persistent play, strength and preference remain untested.')
    except BaseException as error:
        report.update(status='Failed',error=str(error))
        raise
    finally:
        write_progress(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(json.loads(parser.parse_args().config.read_text()))
