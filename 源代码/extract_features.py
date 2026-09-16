"""Full-dataset, resumable feature extraction. Ground truth is scoring-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,concurrent.futures,json,time,hashlib
import cv2,numpy as np,pandas as pd
from core import METHODS,SEED,read_image,resize,segment,feature_vector,segmentation_scores,write_image

BASE=Path(__file__).resolve().parent
def metadata(root):
 cub=root/'CUB_200_2011'
 def read(name,cols): return pd.read_csv(cub/name,sep=' ',header=None,names=cols)
 df=read('images.txt',['image_id','relative_path']).merge(read('image_class_labels.txt',['image_id','class_id']),on='image_id').merge(read('train_test_split.txt',['image_id','is_train']),on='image_id')
 assert len(df)==11788 and df.class_id.nunique()==200 and df.image_id.nunique()==11788
 assert df.is_train.sum()==5994
 return df.sort_values('image_id').reset_index(drop=True)

def worker(task):
 root,record,cache=task; ident,rel,cls,train=record; cv2.setNumThreads(1)
 dest=Path(cache)/f'{ident:05d}.npz'
 if dest.exists(): return ident,None
 start=time.perf_counter(); im=resize(read_image(Path(root)/'CUB_200_2011/images'/rel))
 gt=cv2.imdecode(np.fromfile(str(Path(root)/'segmentations'/Path(rel).with_suffix('.png')),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
 if gt is None: raise ValueError(f'Missing ground truth {ident}')
 vectors=[]; scores=[]; times=[]; falls=[]; masks={}; filtered={}
 for method in METHODS:
  t=time.perf_counter()
  if method=='no_align': mask,flt,fall=masks['full'],filtered['full'],falls[METHODS.index('full')]
  else: mask,flt,fall=segment(im,method,SEED) # identical seed at training and inference
  vec,names,groups,crop,cm=feature_vector(flt,mask,method not in ('raw','no_align'))
  vectors.append(vec); scores.append(segmentation_scores(mask,gt)); times.append(time.perf_counter()-t); falls.append(fall)
  masks[method]=mask; filtered[method]=flt
 np.savez_compressed(dest,features=np.stack(vectors),scores=scores,times=times,fallback=falls)
 return ident,time.perf_counter()-start

def run(root,workers=4):
 root=Path(root); cache=root/'feature_cache_v1'; cache.mkdir(exist_ok=True)
 results=BASE/'results'; results.mkdir(exist_ok=True)
 df=metadata(root); df.to_csv(results/'image_index.csv',index=False)
 todo=[(str(root),tuple(row),str(cache)) for row in df.itertuples(index=False,name=None)]
 start=time.time(); completed=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
  for ident,seconds in ex.map(worker,todo,chunksize=8):
   completed+=1
   if completed%100==0: print(f'FEATURES {completed}/{len(df)} elapsed {(time.time()-start)/60:.1f} min',flush=True)
 features=[]; scores=[]; times=[]; falls=[]
 for ident in df.image_id:
  with np.load(cache/f'{ident:05d}.npz') as a:
   features.append(a['features']); scores.append(a['scores']); times.append(a['times']); falls.append(a['fallback'])
 features=np.stack(features)
 assert features.shape[:2]==(11788,len(METHODS)) and np.isfinite(features).all()
 np.savez_compressed(results/'features.npz',features=features,scores=scores,times=times,fallback=falls)
 first=resize(read_image(root/'CUB_200_2011/images'/df.relative_path.iloc[0])); m,flt,_=segment(first)
 _,names,groups,_,_=feature_vector(flt,m)
 schema={'methods':METHODS,'feature_names':names,'groups':groups,'seed':SEED,'image_count':len(df),'feature_count':len(names),'size':160,'workers':workers,'extraction_elapsed_seconds':time.time()-start,'annotation_policy':'Segmentation masks used only to compute evaluation scores; bounding boxes, part locations, attributes and labels never enter image feature computation.'}
 (results/'feature_schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
 print('ALL FEATURES COMPLETE',features.shape,flush=True)

if __name__=='__main__':
 cfg=json.loads((BASE/'config.json').read_text(encoding='utf-8')); ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=cfg['workers']); args=ap.parse_args(); run(cfg['data_root'],args.workers)
