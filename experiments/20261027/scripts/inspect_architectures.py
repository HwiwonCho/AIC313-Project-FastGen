"""Inspect both public model definitions without creating CUDA tensors."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--our-root',type=Path,required=True)
    parser.add_argument('--peer-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.our_root.resolve()))
    sys.path.insert(0,str(args.peer_root.resolve()))
    import torch
    from networks.soflow_model import SoFlow64
    from scaled_network import FlowUNet,UNetConfig,Attention
    with torch.device('meta'):
        ours=SoFlow64(num_categories=151,base_channels=128,channel_mult=[1,2,2,4],num_blocks=3)
        peer=FlowUNet(UNetConfig(conditioning='cfg'))
    result={'ours':{'parameters':sum(p.numel() for p in ours.parameters()),
        'widths':[128,256,256,512],'encoder_blocks_per_level':3,'decoder_blocks_per_level':4,
        'attention':[{'path':name,'heads':module.num_heads,'channels':module.out_channels}
                     for name,module in ours.named_modules() if getattr(module,'num_heads',0)>0],
        'dropout':sorted({float(m.dropout) for m in ours.modules()
                          if isinstance(getattr(m,'dropout',None),(float,int))})},
        'yunseo_scaled':{'parameters':sum(p.numel() for p in peer.parameters()),'configuration':asdict(peer.config),
        'widths':[128,256,384,512],'encoder_blocks_per_level':2,'decoder_blocks_per_level':3,
        'attention':[{'path':name,'heads':module.heads} for name,module in peer.named_modules() if isinstance(module,Attention)],
        'dropout':sorted({m.p for m in peer.modules() if isinstance(m,torch.nn.Dropout)})},
        'CUDA_initialized':torch.cuda.is_initialized()}
    assert result['ours']['parameters']==95524611
    assert result['yunseo_scaled']['parameters']==98402179
    assert not result['CUDA_initialized']
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
