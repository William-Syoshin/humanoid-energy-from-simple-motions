from getup_probe import *

def scaled(fs,factor):return [dict(f,duration=f['duration']*factor) for f in fs]
def main():
 out=ROOT/'development'/('getup_search_'+time.strftime('%Y%m%d_%H%M%S'));records=[]
 variants=[]
 for scale in [.6,.8,1.2,1.5,2.]:variants.append((f'up_duration_x{scale}',ROLL+scaled(UP,scale)))
 candidate=json.loads((SOURCE.parent/'verified_getup_candidate.json').read_text())['frames']
 variants.append(('candidate_split_hold',ROLL+candidate))
 variants.append(('roll_plus_crawl_plus_up',[f for f in ORIGINAL['frames'] if f['stage'] in ['right_roll','crawl','get_up']]))
 with Session(out) as session:
  for name,seq in variants:
   frames=sequence(seq,hold=3.);spec=development_spec('get_up',seed=91002,duration=frames[-1][0],frames=frames)
   result,d,path=session.run(spec,name+'_91002');m=metrics(d,'up');row=dict(name=name,spec=spec,metrics=m,result=result,path=str(path));records.append(row)
   print(name,json.dumps(m),flush=True);(out/'summary.json').write_text(json.dumps(records,indent=2))
 print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
