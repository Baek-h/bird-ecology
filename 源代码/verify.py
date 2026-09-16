"""Check integrity, splits, cache/inference equivalence, batch and reproducibility."""
from pathlib import Path
import json,hashlib,tempfile
import numpy as np,pandas as pd,joblib
from core import process_image,read_image,write_image
from inference import Engine,flat_result,BASE

def main():
 cfg=json.loads((BASE/'config.json').read_text(encoding='utf-8')); root=Path(cfg['data_root']); r=BASE/'results'; checks=[]
 def check(name,condition):
  checks.append({'check':name,'passed':bool(condition)}); print(name,condition,flush=True)
  if not condition:raise AssertionError(name)
 df=pd.read_csv(r/'image_index.csv'); schema=json.loads((r/'feature_schema.json').read_text()); a=np.load(r/'features.npz'); x=a['features'];
 check('11788 images and 200 classes',len(df)==11788 and df.class_id.nunique()==200)
 check('Official split 5994/5794',int(df.is_train.sum())==5994 and int((df.is_train==0).sum())==5794)
 check('All 11788 image and mask files exist',all((root/'CUB_200_2011/images'/p).is_file() and (root/'segmentations'/Path(p).with_suffix('.png')).is_file() for p in df.relative_path))
 check('All five feature matrices finite',x.shape==(11788,5,795) and np.isfinite(x).all())
 check('All segmentation metrics in [0,1]',np.isfinite(a['scores']).all() and (a['scores']>=0).all() and (a['scores']<=1).all())
 split=pd.read_csv(r/'split_manifest.csv'); check('Train validation test disjoint and exhaustive',split.image_id.nunique()==11788 and len(split)==11788 and (split.split=='official_test').sum()==5794)
 eco=json.loads((r/'ecology_split.json').read_text()); check('Ecology held-out species disjoint',not set(eco['train_species'])&set(eco['held_out_species']))
 check('Full official test predictions for all nine experiments',pd.read_csv(r/'test_predictions.csv').shape==(5794,12))
 indices=[0,135,2750,6000,9000,11787]
 for idx in indices:
  p=root/'CUB_200_2011/images'/df.iloc[idx].relative_path
  for method in ['full','raw','otsu']:
   actual=process_image(p,method)['vector']; check(f'Feature cache equals inference ID {df.iloc[idx].image_id} {method}',np.allclose(actual,x[idx,schema['methods'].index(method)],atol=1e-6,rtol=1e-6))
 engine=Engine(); rec=df[df.is_train==0].iloc[0]; path=root/'CUB_200_2011/images'/rec.relative_path; r1=engine.analyze(path); r2=engine.analyze(path)
 check('Repeated inference deterministic',r1['ranking']==r2['ranking'] and r1['diagnostics']==r2['diagnostics'])
 pred=pd.read_csv(r/'test_predictions.csv'); selected=json.loads((r/'selected_model.json').read_text())['experiment']; check('Saved model reproduces stored test prediction',int(pred.loc[pred.image_id==rec.image_id,selected].iloc[0])==r1['ranking'][0]['class_id'])
 with tempfile.TemporaryDirectory(dir=root) as tmp:
  renamed=Path(tmp)/'无标签新名称.png'; write_image(renamed,read_image(path)); rn=engine.analyze(renamed)
  check('Filename-independent inference',r1['ranking']==rn['ranking'])
  corrupt=Path(tmp)/'invalid.jpg'; corrupt.write_text('not an image')
  try:engine.analyze(corrupt); rejected=False
  except ValueError:rejected=True
  check('Corrupt image rejected explicitly',rejected)
 rows=[flat_result(engine.analyze(root/'CUB_200_2011/images'/s)) for s in df[df.is_train==0].relative_path.iloc[:3]]; pd.DataFrame(rows).to_csv(r/'batch_smoke.csv',index=False,encoding='utf-8-sig'); check('Batch inference returns all 3 records',len(rows)==3)
 (r/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8'); print('VERIFICATION PASSED',flush=True)
if __name__=='__main__':main()
