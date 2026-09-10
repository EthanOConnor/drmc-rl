"""Finite registered source extraction followed by fixed-epoch proposal fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.build_expressive_sequences import write_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='drmc-expressive-study-v1',status='Running',phase='extracting',config=config)
    write_progress(output,report)
    try:
        for phase,recipe,key in [('extracting','trainer-expressive-sequences','expressive_sequences_config'),
                                  ('fitting','trainer-expressive-proposer','expressive_proposer_config')]:
            report['phase'] = phase
            write_progress(output,report)
            with (output/(phase+'.log')).open('ab') as stream:
                subprocess.run([sys.executable,'-m','tools.program','launch',recipe,
                                '--set',key+'='+config[key]],check=True,stdout=stream,stderr=subprocess.STDOUT)
        report.update(status='Complete',phase='complete')
    except BaseException as exc:
        report.update(status='Failed',error=str(exc))
        write_progress(output,report)
        raise
    write_progress(output,report)


if __name__ == '__main__':
    main()
