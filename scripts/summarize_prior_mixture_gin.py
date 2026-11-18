"""Verify and summarize GIN replacement runs without pooling repeated pilots."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.candidates.prior_mixture_impl.diagnostics import separation_auc


def wilson(success, total):
    z=1.959963984540054
    p=success/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=[]
    pilots=[]
    real=[]
    manifests=[]
    for run in args.runs:
        result=json.loads((run/'result.json').read_text())
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest['status']!='completed':
            raise ValueError('Incomplete run')
        with (run/'pilot_scores.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        populations={name:{score:np.array([float(row[column]) for row in rows if row['mixture']==name])
                           for score,column in [('gin','gin_logit'),('ellipsoid','ellipsoid_score'),('ratio','log_mixture_ratio')]}
                     for name in ['normal','abnormal','uniform']}
        for name,values in populations.items():
            if len(values['gin'])!=manifest['pilots_per_mixture']:
                raise ValueError(f'Pilot count mismatch: {name}')
        for score in ['gin','ellipsoid','ratio']:
            actual=separation_auc(populations['normal'][score],populations['abnormal'][score])
            if not np.isclose(actual,result['relative'][score]['auc'],atol=1e-12,rtol=0):
                raise ValueError('Pilot AUROC mismatch')
        with (run/'real_scores.csv').open() as handle:
            real_rows=list(csv.DictReader(handle))
        training_normals=np.array([float(row['gin_logit']) for row in real_rows if row['split']=='train' and row['is_anomaly']=='0'])
        if len(training_normals)!=result['coverage']['real_training']['n'] or not np.isfinite(training_normals).all():
            raise ValueError('Training normal scores are incomplete or non-finite')
        normals=np.array([float(row['gin_logit']) for row in real_rows if row['split']=='test' and row['is_anomaly']=='0'])
        anomalies=np.array([float(row['gin_logit']) for row in real_rows if row['split']=='test' and row['is_anomaly']=='1'])
        if not np.isclose(separation_auc(normals,anomalies),result['real_test']['auc'],atol=1e-12,rtol=0):
            raise ValueError('Real test AUROC mismatch')
        threshold=result['real_normal99_threshold']
        result['real_test']['confusion']=dict(tp=int((anomalies>=threshold).sum()),fn=int((anomalies<threshold).sum()),
                                             fp=int((normals>=threshold).sum()),tn=int((normals<threshold).sum()))
        coverage=result['coverage']['real_test_normal']
        coverage['synthetic95_wilson_interval']=wilson(coverage['accepted_at_synthetic_normal95'],coverage['n'])
        results.append(result)
        pilots.append(populations)
        real.append(dict(normal=normals,abnormal=anomalies,training_normal=training_normals))
        manifests.append(manifest)
    for result,manifest in zip(results[1:],manifests[1:]):
        for key in ['reference_sha256','data_sha256','train_digest','sampling_seed','split_seed']:
            if manifest[key]!=manifests[0][key]:
                raise ValueError(f'Runs differ in {key}')
        if result['pilot_sha256']!=results[0]['pilot_sha256']:
            raise ValueError('Evaluation graphs differ across seeds')
        for key in ['train_data','validation_data']:
            if result['diagnostics'][key]!=results[0]['diagnostics'][key]:
                raise ValueError('Classifier datasets differ across seeds')
    def describe(values):
        return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                    minimum=float(min(values)),maximum=float(max(values)))
    summary=dict(runs=[str(run.resolve()) for run in args.runs],
                 classifier_seeds=[result['classifier_seed'] for result in results],
                 pilot_gin_auc=describe([result['relative']['gin']['auc'] for result in results]),
                 real_anomaly_auc=describe([result['real_test']['auc'] for result in results]),
                 results=results,
                 verification='Raw-score AUROCs, sample counts, frozen priors, split, train data and pilot hashes agree',
                 limitation='Three classifier seeds on one fixed prior fit and split; score acceptance is not density-support coverage')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/prior-mixture-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    def roc(n,a):
        values=np.r_[n,a]
        labels=np.r_[np.zeros(len(n)),np.ones(len(a))]
        order=np.argsort(-values,kind='stable')
        positions=np.r_[np.flatnonzero(np.diff(values[order])),len(values)-1]
        tp=np.cumsum(labels[order])[positions]
        fp=(positions+1)-tp
        return np.r_[0,fp/len(n)],np.r_[0,tp/len(a)]
    for result,values in zip(results,pilots):
        x,y=roc(values['normal']['gin'],values['abnormal']['gin'])
        axes[0].plot(x,y,label=f"GIN seed {result['classifier_seed']}: {result['relative']['gin']['auc']:.3f}")
    for score,label,color in [('ellipsoid','Previous ellipsoid','#777777'),('ratio','Frozen mixture ratio','#a1489e')]:
        x,y=roc(pilots[0]['normal'][score],pilots[0]['abnormal'][score])
        axes[0].plot(x,y,ls='--',color=color,label=f"{label}: {results[0]['relative'][score]['auc']:.3f}")
    axes[0].plot([0,1],[0,1],':',color='#bbbbbb')
    axes[0].set(title='Fresh synthetic pilots · same graphs for every score',xlabel='Normal pilots flagged as abnormal',
                ylabel='Abnormal pilots detected',xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=8,loc='lower right')
    for values,label,color,style in [(pilots[0]['normal']['gin'],'Synthetic normal (8,192)','#2374ab','-'),
                                (pilots[0]['abnormal']['gin'],'Synthetic abnormal (8,192)','#e87924','-'),
                                (real[0]['training_normal'],f"Real train normal ({len(real[0]['training_normal'])})",'#8652a1','--'),
                                (real[0]['normal'],'Real test normal (13)','#2d965c','-'),
                                (real[0]['abnormal'],'Real test anomaly (63)','#bc4040','-')]:
        values=np.sort(values)
        axes[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color,linestyle=style)
    axes[1].axvline(results[0]['synthetic_normal95_threshold'],ls=':',color='#444444',label='Synthetic normal 95% cutoff')
    axes[1].set(title='Transfer to real graphs · classifier seed 0',xlabel='GIN anomaly logit (higher is more abnormal)',
                ylabel='Fraction at or below score',ylim=(0,1))
    axes[1].legend(fontsize=8,loc='lower right')
    fig.savefig(output/'gin_comparison.png',dpi=180)
    fig.savefig(output/'gin_comparison.pdf')
    print(json.dumps({key:summary[key] for key in ['pilot_gin_auc','real_anomaly_auc','verification']},indent=2))


if __name__=='__main__':
    main()
