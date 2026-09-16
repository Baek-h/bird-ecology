from pathlib import Path
import json
import numpy as np,pandas as pd,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import process_image,read_image,segmentation_scores,write_image

BASE=Path(__file__).resolve().parent
def main():
 cfg=json.loads((BASE/'config.json').read_text(encoding='utf-8')); root=Path(cfg['data_root']); results=BASE/'results'; fig=results/'figures'; fig.mkdir(exist_ok=True)
 df=pd.read_csv(results/'image_index.csv'); scores=pd.read_csv(results/'segmentation_per_image.csv'); pred=pd.read_csv(results/'test_predictions.csv'); selected=json.loads((results/'selected_model.json').read_text())['experiment']
 test=scores[scores.is_train==0].sort_values('full_IoU'); positions=[0,len(test)//4,len(test)//2,3*len(test)//4,len(test)-1]; ids=test.iloc[positions].image_id.tolist()
 rows=[]; plt.figure(figsize=(10,11))
 for row,ident in enumerate(ids):
  rec=df[df.image_id==ident].iloc[0]; path=root/'CUB_200_2011/images'/rec.relative_path; p=process_image(path); gt=read_image(root/'segmentations'/Path(rec.relative_path).with_suffix('.png')); gt=cv2.resize(gt,(p['image'].shape[1],p['image'].shape[0]),interpolation=cv2.INTER_NEAREST)
  masked=p['image'].copy(); masked[p['mask']==0]=(masked[p['mask']==0]*.15).astype(np.uint8)
  iou=float(test[test.image_id==ident].full_IoU.iloc[0]); rows.append({'image_id':int(ident),'relative_path':rec.relative_path,'full_IoU':iou,'selection':'Official-test full-IoU order statistics: min, Q1, median, Q3, max'})
  for col,(im,title) in enumerate([(p['image'],f'ID {ident} | input'),(gt,'Ground truth'),(masked,f'Automatic | IoU {iou:.3f}'),(p['crop'],'Aligned crop')]):
   ax=plt.subplot(5,4,row*4+col+1); ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.set_title(title,fontsize=9); ax.axis('off')
 plt.tight_layout(); plt.savefig(fig/'segmentation_examples.png',dpi=170,bbox_inches='tight'); plt.close()
 (results/'example_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 correct=pred[pred[selected]==pred.class_id]; wrong=pred[pred[selected]!=pred.class_id]; chosen=pd.concat([correct.iloc[:3],wrong.iloc[:3]])
 cross=pd.read_csv(BASE/'metadata/species_crosswalk.csv').set_index('class_id'); plt.figure(figsize=(10,6))
 for i,(_,r) in enumerate(chosen.iterrows()):
  im=read_image(root/'CUB_200_2011/images'/r.relative_path); ax=plt.subplot(2,3,i+1); ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.axis('off'); ax.set_title(f"True: {cross.loc[r.class_id,'common_name']}\nPred: {cross.loc[r[selected],'common_name']}",fontsize=8)
 plt.tight_layout(); plt.savefig(fig/'classification_examples.png',dpi=170,bbox_inches='tight'); plt.close()
 counts=df.groupby(['class_id','is_train']).size().unstack(); plt.figure(figsize=(10,3)); plt.bar(counts.index,counts[1],label='Train',color='#276c79'); plt.bar(counts.index,counts[0],bottom=counts[1],label='Test',color='#8bb4aa'); plt.xlabel('Class ID'); plt.ylabel('Image count'); plt.title('All 200 classes retained'); plt.legend(); plt.tight_layout(); plt.savefig(fig/'class_distribution.png',dpi=160); plt.close()
 print('EXAMPLE FIGURES READY',flush=True)
if __name__=='__main__':main()
