"""Shared inference service for desktop GUI, batch use and verification."""
from pathlib import Path
import json,time
import joblib,numpy as np,pandas as pd
from core import process_image,SEED

BASE=Path(__file__).resolve().parent
class Engine:
 def __init__(self):
  self.model=joblib.load(BASE/'models/production.joblib')
  self.ecology=joblib.load(BASE/'models/ecology.joblib')
  self.cross=pd.read_csv(BASE/'metadata/species_crosswalk.csv').set_index('class_id')
  self.schema=self.model['feature_schema']; self.names=self.schema['feature_names']
 def analyze(self,path):
  start=time.perf_counter(); method=self.model['method']; p=process_image(path,method)
  z=p['vector'][self.model['columns']].reshape(1,-1)
  pipeline=self.model['pipeline']; scores=pipeline.decision_function(z)[0]; order=np.argsort(scores)[-5:][::-1]; cls=pipeline.classes_
  ranking=[{'class_id':int(cls[i]),'name':str(self.cross.loc[cls[i],'common_name']),'svm_score':float(scores[i])} for i in order]
  row=self.cross.loc[ranking[0]['class_id']]; matched=bool(row.mapped)
  keys=['Species2','Mass','Hand-Wing.Index','Beak.Length_Nares','Habitat','Habitat.Density','Migration','Trophic.Niche','Primary.Lifestyle']
  reference={k:None if pd.isna(row.get(k)) else (row[k].item() if hasattr(row[k],'item') else row[k]) for k in keys} if matched else {}
  full=p if method=='full' else process_image(path,'full')
  ez=full['vector'].reshape(1,-1); e=self.ecology['target_scaler'].inverse_transform(self.ecology['pipeline'].predict(ez))[0]
  estimates={'log10_mass_g':float(e[0]),'mass_g':float(10**np.clip(e[0],-3,7)),'hand_wing_index':float(e[1]),'beak_nares_mm':float(e[2])}
  diagnostics={k:float(full['vector'][self.names.index(k)]) for k in ['foreground_ratio','edge_box_dimension','horizontal_mask_symmetry','vertical_mask_symmetry','horizontal_texture_correlation','vertical_texture_correlation']}
  return {'path':str(path),'method':method,'ranking':ranking,'avonet_reference':reference,'match_note':str(row.match_note),'image_regression':estimates,'diagnostics':diagnostics,'segmentation_fallback':bool(full['fallback']),'seconds':time.perf_counter()-start,'images':p,'detection_images':full,'notes':'SVM scores are not probabilities. AVONET values are species averages conditional on correct identification. Regression is exploratory and does not measure this individual; no ecological fitness score is claimed.'}

def flat_result(result):
 return {'file':result['path'],'predicted_class':result['ranking'][0]['class_id'],'predicted_name':result['ranking'][0]['name'],'method':result['method'],'svm_score':result['ranking'][0]['svm_score'],**{f'avonet_{k}':v for k,v in result['avonet_reference'].items()},**{f'regression_{k}':v for k,v in result['image_regression'].items()},**result['diagnostics'],'fallback':result['segmentation_fallback'],'seconds':result['seconds']}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('images',nargs='+'); ap.add_argument('--csv',required=True); args=ap.parse_args()
 engine=Engine(); rows=[]
 for p in args.images:
  try: rows.append(flat_result(engine.analyze(p)))
  except Exception as exc: rows.append({'file':p,'error':str(exc)})
 pd.DataFrame(rows).to_csv(args.csv,index=False,encoding='utf-8-sig')
