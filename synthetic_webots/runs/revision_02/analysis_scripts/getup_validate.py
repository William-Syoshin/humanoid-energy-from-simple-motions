from getup_probe import *

def main():
 out=ROOT/'development'/('getup_validate_'+time.strftime('%Y%m%d_%H%M%S'));records=[];selected={}
 variants=[('get_up_x1p2','get_up',sequence(ROLL+[dict(f,duration=f['duration']*1.2) for f in UP],hold=3.),'up'),('lie_down','lie_down',sequence(DOWN,STAND,hold=4.),'down')]
 with Session(out) as session:
  for name,action,frames,kind in variants:
   group=[]
   for seed in range(91001,91011):
    spec=development_spec(action,seed=seed,duration=frames[-1][0],frames=frames)
    result,d,path=session.run(spec,name+'_'+str(seed));m=metrics(d,kind);row=dict(name=name,spec=spec,metrics=m,result=result,path=str(path));records.append(row);group.append(row)
    print(name,seed,'passed',m['passed'],'z',round(m['final_height_m'],5),'tilt',round(m['tail_max_tilt_deg'],3),'v',m['tail_mean_linear_speed'],'omega',m['tail_mean_angular_speed'],flush=True);(out/'summary.json').write_text(json.dumps(records,indent=2))
   if all(r['metrics']['passed'] for r in group):selected[action]=dict(frames=frames,duration=frames[-1][0],development_passes=len(group),development_seeds=list(range(91001,91011)),name=name)
  if 'get_up' not in selected:
   for factor in [1.5,2.]:
    name='get_up_x'+str(factor);frames=sequence(ROLL+[dict(f,duration=f['duration']*factor) for f in UP],hold=3.);group=[]
    for seed in range(91001,91011):
     spec=development_spec('get_up',seed=seed,duration=frames[-1][0],frames=frames)
     result,d,path=session.run(spec,name+'_'+str(seed));m=metrics(d,'up');row=dict(name=name,spec=spec,metrics=m,result=result,path=str(path));records.append(row);group.append(row)
     print(name,seed,'passed',m['passed'],'z',round(m['final_height_m'],5),'tilt',round(m['tail_max_tilt_deg'],3),'v',m['tail_mean_linear_speed'],'omega',m['tail_mean_angular_speed'],flush=True);(out/'summary.json').write_text(json.dumps(records,indent=2))
    if all(r['metrics']['passed'] for r in group):selected['get_up']=dict(frames=frames,duration=frames[-1][0],development_passes=len(group),development_seeds=list(range(91001,91011)),name=name);break
 (out/'adoptable_sequences.json').write_text(json.dumps(dict(sequences=selected,selection_criterion='All ten development perturbations pass terminal mechanics; no energy metric evaluated',original_source=str(SOURCE)),indent=2))
 print('OUTPUT',out,'SELECTED',list(selected),flush=True)
if __name__=='__main__':main()
