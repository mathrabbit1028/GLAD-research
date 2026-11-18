"""Audit synthetic/real normal mismatch and probe a frozen GIN; no fitting."""
import os
for variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp, spearmanr
from src.utils.config import _wrap
from src.utils.device import require_cuda
from src.utils.seed import set_seed
from src.data.datamodule import TUDataModule
from src.models.factory import create_model
from src.models.candidates.prior_mixture import CandidateAdapter as ProposalAdapter
from src.models.candidates.prior_mixture_gin import sample_mixture, graph_digest
from src.models.candidates.prior_mixture_impl.sampling import PriorBank


def attributes(graph):
    n=graph.num_nodes
    adjacency=np.zeros((n,n),dtype=float)
    edges=graph.edge_index.numpy()
    adjacency[edges[0],edges[1]]=1
    degree=adjacency.sum(1)
    count=connected_components(adjacency,directed=False,return_labels=False)
    category=graph.x.argmax(1).numpy()
    histogram=np.bincount(category,minlength=graph.x.shape[1])/n
    edge_count=edges.shape[1]//2
    return dict(nodes=n,mean_degree=float(degree.mean()),max_degree=float(degree.max()),
                components=int(count),connected=float(count==1),isolated_fraction=float((degree==0).mean()),
                degree_over4_fraction=float((degree>4).mean()),triangle_count=float(np.trace(adjacency@adjacency@adjacency)/6),
                cycle_rank=int(edge_count-n+count),
                label_homophily=float((category[edges[0]]==category[edges[1]]).mean()) if edges.shape[1] else 0.,
                **{f'category_{i}':float(value) for i,value in enumerate(histogram)})


def describe(graphs):
    rows=[attributes(graph) for graph in graphs]
    columns={key:np.array([row[key] for row in rows]) for key in rows[0]}
    frequencies=torch.cat([graph.x for graph in graphs]).mean(0).numpy()
    return dict(n=len(graphs),node_category_frequencies=frequencies.tolist(),
                moments={key:dict(mean=float(values.mean()),median=float(np.median(values)),
                                  q05_q95=np.quantile(values,[.05,.95]).tolist()) for key,values in columns.items()}),columns


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    set_seed(70001)
    device=require_cuda('cuda:0')
    started=time.monotonic()
    checkpoint=torch.load(args.run/'classifier_checkpoint.pt',weights_only=True,map_location='cpu')
    cfg=_wrap(checkpoint['config'])
    data=TUDataModule(cfg.dataset,int(cfg.seed),split=checkpoint['split'])
    graphs=ProposalAdapter._clean_training(data.train_dataloader())
    model=create_model(data.dataset.num_features,cfg.model,device)
    model.load_state_dict(checkpoint['model_state'])
    spec=model.proposals
    synthetic,components=sample_mixture(graphs,cfg.model.proposal.prior,spec['normal'],8192,52001)
    saved=json.loads((args.run/'result.json').read_text())
    assert graph_digest(graphs)==model.diagnostics['training_graph_digest']
    assert graph_digest(synthetic)==saved['pilot_sha256']['normal']
    normal_label=checkpoint['split']['normal_label']
    test_normal=[data.dataset[i] for i in checkpoint['split']['test_indices'] if int(data.dataset[i].y)==normal_label]
    populations=dict(real_train_normal=graphs,synthetic_normal=synthetic,real_test_normal=test_normal)
    summaries,columns,logits={},{},{}
    raw=[]
    for name,population in populations.items():
        summaries[name],columns[name]=describe(population)
        logits[name]=model.score_graphs(population)
        summaries[name]['logit']=dict(mean=float(logits[name].mean()),median=float(np.median(logits[name])),
                                     q05_q95=np.quantile(logits[name],[.05,.95]).tolist())
        for i in range(len(population)):
            raw.append(dict(population=name,index=i,logit=float(logits[name][i]),**{k:float(v[i]) for k,v in columns[name].items()}))
        print('BASELINE '+name+' '+json.dumps(summaries[name]),flush=True)
    comparisons={key:dict(ks=float(ks_2samp(columns['real_train_normal'][key],columns['synthetic_normal'][key]).statistic),
                          real_mean=summaries['real_train_normal']['moments'][key]['mean'],
                          synthetic_mean=summaries['synthetic_normal']['moments'][key]['mean']) for key in columns['synthetic_normal']}
    # Independently generated component pilots check whether reweighting can plausibly repair moments.
    component_summaries=[]
    bank=PriorBank(graphs,cfg.model.proposal.prior,71001)
    for component in range(cfg.model.proposal.prior.components):
        samples=[bank.sample(component) for _ in range(256)]
        description,_=describe(samples)
        component_summaries.append(dict(component=component,normal_weight=float(spec['normal'][component]),**description))
    print('COMPONENT_AUDIT complete',flush=True)
    # Same topology, varying node-label mechanisms. These are diagnostic perturbations,
    # not molecularly valid graphs or a retrained model. The IID cross compares marginals
    # with both mechanisms under the same independence assumption.
    proportions={name:np.asarray(summaries[name]['node_category_frequencies']) for name in ['real_train_normal','synthetic_normal']}
    interventions=[]
    for source in ['real_train_normal','synthetic_normal']:
        population=populations[source]
        for mechanism in ['shuffle_within_graph','iid_real_marginal','iid_synthetic_marginal','connect_components']:
            records=[]
            for repeat in range(4):
                rng=np.random.default_rng(72001+repeat)
                variants=[]
                for graph in population:
                    variant=graph.clone()
                    if mechanism=='shuffle_within_graph':
                        variant.x=graph.x[rng.permutation(graph.num_nodes)]
                    elif mechanism.startswith('iid'):
                        target='real_train_normal' if mechanism=='iid_real_marginal' else 'synthetic_normal'
                        labels=rng.choice(graph.x.shape[1],graph.num_nodes,p=proportions[target]/proportions[target].sum())
                        variant.x=torch.nn.functional.one_hot(torch.tensor(labels),graph.x.shape[1]).float()
                    else:
                        n=graph.num_nodes
                        adj=np.zeros((n,n))
                        adj[graph.edge_index[0].numpy(),graph.edge_index[1].numpy()]=1
                        number,membership=connected_components(adj,directed=False)
                        if number>1:
                            representatives=[int(rng.choice(np.flatnonzero(membership==c))) for c in range(number)]
                            links=torch.tensor([representatives[:-1],representatives[1:]],dtype=torch.long)
                            variant.edge_index=torch.cat([graph.edge_index,links,links.flip(0)],1)
                    variants.append(variant)
                values=model.score_graphs(variants)
                difference=values-logits[source]
                records.append(dict(mean=float(values.mean()),median=float(np.median(values)),
                                    mean_paired_change=float(difference.mean()),median_paired_change=float(np.median(difference))))
            entry=dict(source=source,mechanism=mechanism,repeats=records,
                       average_mean=float(np.mean([r['mean'] for r in records])),
                       average_paired_change=float(np.mean([r['mean_paired_change'] for r in records])))
            interventions.append(entry)
            print('INTERVENTION '+json.dumps(entry),flush=True)
    correlations={key:float(spearmanr(values,logits['synthetic_normal']).statistic)
                  for key,values in columns['synthetic_normal'].items() if len(np.unique(values))>1}
    with (output/'graph_statistics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    result=dict(populations=summaries,comparisons=comparisons,component_pilots=component_summaries,
                interventions=interventions,synthetic_logit_spearman=correlations,seconds=time.monotonic()-started,
                checkpoints=dict(source=str(args.run.resolve()),sha256=hashlib.sha256((args.run/'classifier_checkpoint.pt').read_bytes()).hexdigest()),
                caveat='Frozen classifier seed 0; interventions test score sensitivity, not physical or uniquely identified causes. No fitting or test-label selection.')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'diagnose_prior_mixture_gap.py').write_text(Path(__file__).read_text())
    print('COMPLETE '+str(output),flush=True)

if __name__=='__main__':
    main()
