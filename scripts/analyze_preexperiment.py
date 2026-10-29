"""Recompute the fixed 36-run pilot analysis using only the standard library.

Reads saved validation results, never trains or evaluates the test set.
Replaces derived CSV/JSON files in reports/preexperiment_analysis_v1;
does not regenerate the narrative DOCX or modify original runs.
"""

from pathlib import Path
import json,csv,statistics as st,collections,hashlib
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/preexperiment_analysis_v1'
OUT.mkdir(parents=True, exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return st.mean(v)
def sd(v):return st.stdev(v) if len(v)>1 else 0.
def csvout(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Only successful global updates count as household contributions.
runs=[];households=[];rounds=[];audits=[];sources={};curves={};traces={}
for path in sorted((ROOT/'runs/laptop_validation_v1').iterdir()):
 if not (path/'config.json').exists():continue
 cfg=load(path/'config.json');meta=load(path/'metadata.json');status=load(path/'status.json')
 ms=[load(p) for p in sorted((path/'rounds').glob('[0-9]*/metrics.json'))]
 assert status['status']=='complete' and len(ms)==50 and [m['round'] for m in ms]==list(range(1,51))
 selected=collections.Counter();contrib=collections.Counter();profile={};drops=late=0;selprofile=collections.Counter();sucprofile=collections.Counter()
 for m in ms:
  mp=path/'rounds'/f'{m["round"]:04}'/'metrics.json';cp=mp.parent/'clients.json';sources[str(mp.relative_to(ROOT))]=sha(mp);sources[str(cp.relative_to(ROOT))]=sha(cp)
  clients=load(cp)
  for e in clients:
   cid=e['client'];profile[cid]=e['profile'];selected[cid]+=int(e['selected']);contrib[cid]+=int(e['survived'] and m['update_applied'])
   if e['selected']:
    selprofile[e['profile']]+=1;drops+=int(e['dropout']);late+=int(not e['dropout'] and e['actual_s']>cfg['simulation']['deadline_s'])
   if e['survived'] and m['update_applied']:sucprofile[e['profile']]+=1
  rounds.append({'run':path.name,'round':m['round'],'val_mae':m['validation']['macro_mae_kwh'],'val_rmse':m['validation']['macro_rmse_kwh'],'p90':m['validation']['p90_household_mae_kwh'],'virtual_s':m['virtual_elapsed_s'],'round_wall_s':m['measured_round_s'],'cumulative_wall_s':m['measured_wall_s'],'update_applied':m['update_applied']})
 vals=[m['validation']['macro_mae_kwh'] for m in ms];curves[path.name]=ms
 x=list(contrib.values());jain=sum(x)**2/(len(x)*sum(v*v for v in x)) if sum(x) else 0.
 item={'run':path.name,'scheduler':cfg['train']['scheduler'],'deadline':cfg['simulation']['deadline_s'],'seed':cfg['seed'],'mae10':vals[9],'mae50':vals[-1],'best_mae':min(vals),'last10_mae':mean(vals[-10:]),'last5_minus_prev5':mean(vals[-5:])-mean(vals[-10:-5]),'rmse50':ms[-1]['validation']['macro_rmse_kwh'],'p90_50':ms[-1]['validation']['p90_household_mae_kwh'],'virtual_s':ms[-1]['virtual_elapsed_s'],'wall_s':ms[-1]['measured_wall_s'],'round_wall_sum_s':sum(m['measured_round_s'] for m in ms),'wall_gap_s':ms[-1]['measured_wall_s']-sum(m['measured_round_s'] for m in ms),'failed':sum(not m['update_applied'] for m in ms),'dropout_events':drops,'non_dropout_timeout_events':late,'zero_contributors':sum(v==0 for v in x),'jain_contributions':jain,'constrained_contrib':sucprofile['constrained'],'intermediate_contrib':sucprofile['intermediate'],'capable_contrib':sucprofile['capable'],'modeled_bytes':sum(m['modeled_round_bytes'] for m in ms),'peak_rss_mb':max(m['peak_process_rss_mb'] for m in ms)}
 runs.append(item)
 for cid in sorted(profile):households.append({'run':path.name,'client':cid,'profile':profile[cid],'selected':selected[cid],'contributed':contrib[cid]})
 trace=load(path/'trace.json');traces[path.name]=trace
 audits.append({'run':path.name,'status':status['status'],'rounds':len(ms),'dataset_fingerprint':meta['dataset_fingerprint'],'source_fingerprint':hashlib.sha256(json.dumps(meta['source_files'],sort_keys=True).encode()).hexdigest(),'config_matches_plan':cfg==load(ROOT/'reports/plans/laptop_validation_v1'/(path.name+'.json')),'test_exists':(path/'test').exists(),'integrity_exists':(path/'integrity.json').exists()})
 for p in [path/'config.json',path/'metadata.json',path/'status.json',path/'trace.json']:sources[str(p.relative_to(ROOT))]=sha(p)
# Preserve seeds as the replication unit; deadlines with identical curves
# are separate scenarios, not extra independent replications.
assert len(runs) == 36, 'Expected the fixed 36-run validation matrix'
groups=[]
keys=[k for k in runs[0] if k not in ['run','scheduler','deadline','seed']]
for d in [1,3,10,60]:
 for s in ['random','fastest','coverage']:
  rows=[r for r in runs if r['deadline']==d and r['scheduler']==s];assert len(rows)==3
  g={'deadline':d,'scheduler':s,'n':3}
  for k in keys:g[k+'_mean']=mean([r[k] for r in rows]);g[k+'_sd']=sd([r[k] for r in rows])
  groups.append(g)
paired=[]
for d in [1,3,10,60]:
 for other in ['coverage','fastest']:
  for seed in [7,17,27]:
   a=next(r for r in runs if r['deadline']==d and r['scheduler']=='random' and r['seed']==seed);b=next(r for r in runs if r['deadline']==d and r['scheduler']==other and r['seed']==seed)
   paired.append({'deadline':d,'seed':seed,'comparison':other+' minus random','mae50_delta':b['mae50']-a['mae50'],'last10_mae_delta':b['last10_mae']-a['last10_mae'],'virtual_s_delta':b['virtual_s']-a['virtual_s']})
# Post-hoc descriptive comparisons, NOT a predeclared optimization target.
time_rows=[]
for d in [1,3,10,60]:
 for seed in [7,17,27]:
  names=[f'{s}_d{d}_s{seed}' for s in ['random','fastest','coverage']];common=min(curves[n][-1]['virtual_elapsed_s'] for n in names)
  for n in names:
   eligible=[m for m in curves[n] if m['virtual_elapsed_s']<=common];m=eligible[-1] if eligible else None
   time_rows.append({'deadline':d,'seed':seed,'scheduler':n.split('_')[0],'common_budget_s':common,'round':m['round'] if m else None,'mae':m['validation']['macro_mae_kwh'] if m else None})
worst=sorted(rounds,key=lambda r:r['round_wall_s'],reverse=True)[:10]
summary={'audits':audits,'groups':groups,'paired':paired,'same_time':time_rows,'top_wall_rounds':worst,'runs':runs,
 'curves_identical_d3_d10_d60':all([m['validation'] for m in curves[f'{s}_d3_s{k}']]==[m['validation'] for m in curves[f'{s}_d10_s{k}']]==[m['validation'] for m in curves[f'{s}_d60_s{k}']] for s in ['random','fastest','coverage'] for k in [7,17,27]),
 'traces_matched_per_seed':all(traces[f'random_d3_s{k}']==traces[f'{s}_d{d}_s{k}'] for k in [7,17,27] for s in ['random','fastest','coverage'] for d in [1,3,10,60]),
 'total_wall_hours':sum(r['wall_s'] for r in runs)/3600,'median_run_wall_s':st.median(r['wall_s'] for r in runs),'median_round_wall_s':st.median(r['round_wall_s'] for r in rounds)}
for name,rows in [('runs.csv',runs),('groups.csv',groups),('paired.csv',paired),('same_time.csv',time_rows),('household_participation.csv',households),('rounds.csv',rounds)]:csvout(name,rows)
(OUT/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'evidence_sha256.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['curves_identical_d3_d10_d60','traces_matched_per_seed','total_wall_hours','median_run_wall_s','median_round_wall_s','top_wall_rounds']},indent=2))
print('GROUPS',[(g['deadline'],g['scheduler'],round(g['mae50_mean'],6),round(g['jain_contributions_mean'],3),round(g['last5_minus_prev5_mean'],6)) for g in groups])
print('PAIRED D3',[p for p in paired if p['deadline']==3])
print('SAME TIME D3',[p for p in time_rows if p['deadline']==3])
