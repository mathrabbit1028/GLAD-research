#!/usr/bin/env python3
"""Post-run attribution: repeat Level B with its initial, untrained residual AE.

No tuning or selection uses these test scores. This separates the soft input
policy's initial effect from improvements due to outer AE optimization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '1'
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from src.utils.config import _wrap, load_config, to_plain
    from src.utils.seed import set_seed
    from src.utils.device import require_cuda
    from src.data.datamodule import TUDataModule
    from src.models.factory import create_model
    from src.models.candidates.prior_mixture import CandidateAdapter as Proposal
    from src.models.candidates.prior_mixture_impl.sampling import PriorBank
    from src.models.candidates.ae_prior_mixture_impl.core import mmd
    torch.set_num_threads(1)
    device = require_cuda('cuda:0')
    bank = torch.load(args.root/'bank.pt', weights_only=True, map_location='cpu')
    cfg = _wrap(bank['model_config'])
    cfg.mode = 'no_ae'
    data = TUDataModule(load_config(['dataset=mutag']).dataset, 0, split=bank['split'])
    graphs = Proposal._clean_training(data.train_dataloader(shuffle=False))
    set_seed(args.seed)
    model = create_model(data.dataset.num_features, cfg, device)
    model.fit_prepared(graphs, bank, args.seed)
    reference = torch.load(args.root/f'seed{args.seed}/no_ae/checkpoint.pt', weights_only=True, map_location=device)
    for name, value in model.network.state_dict().items():
        torch.testing.assert_close(value, reference['model_state']['network'][name], rtol=2e-5, atol=2e-6)
    cfg.mode = 'level_b'
    with torch.no_grad():
        weights = model._mixtures()
    batches = model._batches(model.train_pool, cfg.inner.steps, cfg.inner.per_component, cfg.sampling.seed+3)
    params = model._inner(weights, batches, False)
    model.network.load_state_dict({**model.network.state_dict(), **params})
    model.weights = weights
    fit = [graphs[i] for i in bank['roles']['fit']]
    pilots, ids = PriorBank(fit, cfg.proposal.prior, 84000).labelled(64)
    logits = model.score_graphs(pilots, transform=False)
    wn, wa = [w.detach().cpu().numpy()[ids.numpy()]/64 for w in weights]
    pilot_auc = roc_auc_score(np.r_[np.zeros(len(logits)),np.ones(len(logits))],
                             np.r_[logits, logits], sample_weight=np.r_[wn,wa])
    test = [data.dataset[i] for i in data.split['test_indices']]
    labels = [int(int(g.y.item()) != data.split['normal_label']) for g in test]
    soft, raw = model.score_graphs(test), model.score_graphs(test, transform=False)
    select = [graphs[i] for i in bank['roles']['select']]
    raw_select, soft_select = model.score_graphs(select, False), model.score_graphs(select)
    with torch.no_grad():
        p, pw = torch.tensor(logits, device=device), torch.tensor(wn, device=device)
        gap_raw = float(mmd(torch.tensor(raw_select, device=device), p, model.scale, wy=pw))
        gap_soft = float(mmd(torch.tensor(soft_select, device=device), p, model.scale, wy=pw))
    result = dict(seed=args.seed, condition='untrained_near_identity_AE_with_full_inner_refit',
                  raw_initialization_reproduction='passed', synthetic_auc=float(pilot_auc),
                  real_auc=float(roc_auc_score(labels,soft)), real_raw_auc=float(roc_auc_score(labels,raw)),
                  select_raw_mmd=gap_raw, select_soft_mmd=gap_soft,
                  weights=[w.detach().cpu().tolist() for w in weights],
                  test_indices=data.split['test_indices'], test_labels=labels, test_scores=soft.tolist(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = args.root/f'seed{args.seed}/initialization_audit.json'
    if path.exists():
        raise ValueError('Attribution audit already exists')
    path.write_text(json.dumps(result, indent=2)+'\n')
    torch.save(dict(model_state=model.state_dict(), model_config=to_plain(cfg)),
               args.root/f'seed{args.seed}/initialization_audit.pt')
    print('INITIALIZATION_AUDIT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
