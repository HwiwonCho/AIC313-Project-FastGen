"""Verify observation does not change updates/RNG and all logging is durable."""
import csv
import tempfile
from pathlib import Path
import torch
from . import models
from .recipe import UpdateEngine,Config
from .recording import Recorder
from torch_utils import training_stats

def main():
    torch.set_num_threads(4)
    snapshots=[]
    temp_root=Path(__file__).resolve().parent.parent/'runs'
    temp_root.mkdir(exist_ok=True)
    for observe in (False,True):
        training_stats._counters.clear();training_stats._cumulative.clear()
        torch.manual_seed(42)
        channels={4:16,8:16,16:16,32:8,64:8}
        e=UpdateEngine(models.Generator(channels,False),models.Discriminator(channels,False),
                       Config(batch=4,micro=4,fast_regularization=True))
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            run=Path(temp)
            assert run.resolve().parent==temp_root.resolve()
            r=Recorder(run,e,Path(__file__).resolve().parent.parent) if observe else None
            for _ in range(5):
                e.step(torch.rand(4,3,64,64)*2-1,torch.arange(4))
                if r:r.update(.1,0)
            snapshot={n:{k:v.detach().clone() for k,v in m.state_dict().items()}
                      for n,m in [('G',e.G),('D',e.D),('EMA',e.ema),('ADA',e.augment)]}
            snapshot['rng']=torch.get_rng_state().clone();snapshot['pl']=e.loss.pl_mean.clone()
            snapshots.append(snapshot)
            if r:
                r.close()
                with (run/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
                assert [int(x['step']) for x in rows]==[1,2,3,4,5]
                assert rows[1]['Loss/G/reg']=='' and rows[1]['Loss/r1_penalty']==''
                assert rows[0]['Loss/G/reg']!='' and rows[0]['Loss/r1_penalty']!=''
                assert (run/'loss_curve.png').is_file() and (run/'source_snapshot/model.py').is_file()
                assert (run/'environment.json').is_file() and (run/'timing.json').is_file()
                # Simulate rollback to step 3: retain orphan evidence and count
                # already-spent update time rather than erasing it on resume.
                e.iteration=3;e.images_seen=12
                e.recording_state={'update_seconds':.3,'active_wall_seconds':.01}
                resumed=Recorder(run,e,Path(__file__).resolve().parent.parent,resume=True)
                e.step(torch.rand(4,3,64,64)*2-1,torch.arange(4))
                resumed.update(.1,0);resumed.close()
                with (run/'metrics.csv').open() as f:recovered=list(csv.DictReader(f))
                assert [int(x['step']) for x in recovered]==[1,2,3,4]
                assert float(recovered[-1]['total_update_seconds'])>=.59
                assert 'uncheckpointed_metrics' in (run/'events.jsonl').read_text()
                from .train import preview
                e.iteration=100;preview(e,run)
                assert len(list((run/'early_samples/step_0000100').glob('*.png')))==8
                assert (run/'samples/step_0000100.json').exists()
    for name in ['G','D','EMA','ADA']:
        for key,value in snapshots[0][name].items():
            torch.testing.assert_close(value,snapshots[1][name][key],rtol=0,atol=0)
    for name in ['rng','pl']:assert torch.equal(snapshots[0][name],snapshots[1][name])
    print('PASS recording: exact CPU updates/RNG/EMA/ADA/PL; steps 1-5; absent lazy phases blank; source/time/curve evidence')

if __name__=='__main__':main()
