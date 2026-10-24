"""Exercise the shell-to-evaluation chain using explicit synthetic records."""
import csv
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import json
import numpy as np

repo = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='biomoax shell ') as tmp:
    root = Path(tmp)
    np.save(root/'query.npy', [[1.,0.],[0.,1.]])
    np.save(root/'gallery.npy', [[1.,0.],[0.,1.]])
    def csv_file(name, header, rows):
        with (root/name).open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
    csv_file('q.csv',['record_id'],[['q1'],['q2']])
    csv_file('g.csv',['record_id'],[['g1'],['g2']])
    csv_file('positive.csv',['query_id','gallery_id'],[['q1','g1'],['q2','g2']])
    values = dict(CORE_PYTHON=sys.executable, RUN_ROOT=str(root/'run'),
                  QUERY_FEATURES=str(root/'query.npy'), GALLERY_FEATURES=str(root/'gallery.npy'),
                  QUERY_IDS=str(root/'q.csv'), GALLERY_IDS=str(root/'g.csv'),
                  POSITIVE_PAIRS=str(root/'positive.csv'))
    config = root/'config.env'
    config.write_text('\n'.join(k+'='+shlex.quote(v) for k,v in values.items())+'\n')
    command = ['bash',str(repo/'scripts/06_evaluate.sh'),str(config)]
    subprocess.run(command, check=True, cwd=tmp)
    metrics = json.loads((root/'run/metrics.json').read_text())
    assert metrics['n_eligible']==2 and metrics['top1_percent']==100
    assert (root/'run/logs/evaluate.complete').is_file()
    assert subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0
    # Unknown positive IDs must stop before producing evaluation output.
    csv_file('positive.csv',['query_id','gallery_id'],[['missing','g1']])
    config.write_text(config.read_text().replace(str(root/'run'),str(root/'invalid_run')))
    assert subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0
    assert not (root/'invalid_run/logs/evaluate.complete').exists()
    # Exercise task mode with shuffled feature pools: IDs, not row positions, decide alignment.
    task=root/'task'; task.mkdir()
    csv_file('task/query_ids.csv',['record_id'],[['q1'],['q2']])
    csv_file('task/gallery_ids.csv',['record_id'],[['g1'],['g2']])
    csv_file('task/positive_pairs.csv',['query_id','gallery_id'],[['q1','g1'],['q2','g2']])
    csv_file('task/excluded_pairs.csv',['query_id','gallery_id'],[])
    csv_file('qpool.csv',['record_id'],[['q2'],['q1']])
    np.save(root/'qpool.npy',[[0.,1.],[1.,0.]])
    values.update(RUN_ROOT=str(root/'task_run'),TASK_DIR=str(task),
                  QUERY_POOL=str(root/'qpool.npy'),QUERY_POOL_IDS=str(root/'qpool.csv'),
                  GALLERY_POOL=str(root/'gallery.npy'),GALLERY_POOL_IDS=str(root/'g.csv'))
    config.write_text('\n'.join(k+'='+shlex.quote(v) for k,v in values.items())+'\n')
    subprocess.run(command,check=True)
    assert json.loads((root/'task_run/metrics.json').read_text())['top1_percent']==100
print('PASS: shell evaluation, paths with spaces, overwrite protection, invalid-ID failure')
