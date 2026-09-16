"""Validation-selected classical classifiers and leakage-controlled evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json,time,platform,sys
import cv2,joblib,numpy as np,pandas as pd,sklearn
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report,mean_absolute_error,r2_score
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import SEED

BASE=Path(__file__).resolve().parent; R=BASE/'results'; FIG=R/'figures'; M=BASE/'models'
def dump(name,data): (R/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def savefig(name): plt.tight_layout(); plt.savefig(FIG/name,dpi=160,bbox_inches='tight'); plt.close()
def bootstrap_mean(v,seed=SEED):
 rng=np.random.default_rng(seed); v=np.asarray(v); boot=[rng.choice(v,len(v),replace=True).mean() for _ in range(1000)]
 return np.quantile(boot,[.025,.975]).tolist()

def run():
 FIG.mkdir(parents=True,exist_ok=True); M.mkdir(exist_ok=True)
 schema=json.loads((R/'feature_schema.json').read_text()); a=np.load(R/'features.npz'); x=a['features']; seg=a['scores']; timing=a['times']; falls=a['fallback']
 df=pd.read_csv(R/'image_index.csv'); y=df.class_id.to_numpy(); train=df.is_train.to_numpy()==1; test=~train
 ids=np.arange(len(df)); tr,va=train_test_split(ids[train],test_size=.2,stratify=y[train],random_state=SEED)
 pd.DataFrame({'image_id':df.image_id,'split':np.where(test,'official_test',np.where(np.isin(ids,va),'validation','development_train'))}).to_csv(R/'split_manifest.csv',index=False)
 groups=schema['groups']; full=schema['methods'].index('full')
 experiments=[(m,i,np.arange(x.shape[2])) for i,m in enumerate(schema['methods'])]
 for remove in ['texture','complexity','hog']:
  s,e=groups[remove]; experiments.append((f'without_{remove}',full,np.r_[0:s,e:x.shape[2]]))
 experiments.append(('color_only',full,np.arange(*groups['color'])))
 metrics=[]; preds={}; decisions={}; best_validation=-1; selected_name=None; started=time.time()
 for name,mi,cols in experiments:
  z=x[:,mi,:][:,cols]; best=None; trials=[]
  for c in [1.,10.]:
   pipe=make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',cache_size=256,decision_function_shape='ovr',random_state=SEED))
   t=time.perf_counter(); pipe.fit(z[tr],y[tr]); score=accuracy_score(y[va],pipe.predict(z[va])); trials.append({'C':c,'validation_accuracy':score})
   print(f'VALIDATE {name} C={c:g} acc={score:.4f}',flush=True)
   if best is None or score>best[0]: best=(score,c)
  # The production method is selected only by validation, before viewing test scores.
  if name in schema['methods'] and best[0]>best_validation: best_validation=best[0]; selected_name=name
  pipe=make_pipeline(StandardScaler(),SVC(C=best[1],kernel='rbf',gamma='scale',cache_size=256,decision_function_shape='ovr',random_state=SEED))
  t=time.perf_counter(); pipe.fit(z[train],y[train]); fit_seconds=time.perf_counter()-t
  t=time.perf_counter(); dec=pipe.decision_function(z[test]); pred=pipe.classes_[np.argmax(dec,axis=1)]; infer_seconds=time.perf_counter()-t
  top5=np.take(pipe.classes_,np.argsort(dec,axis=1)[:,-5:]); acc=accuracy_score(y[test],pred)
  item={'experiment':name,'feature_count':len(cols),'C':best[1],'validation_accuracy':best[0],'test_accuracy':acc,'macro_f1':f1_score(y[test],pred,average='macro',zero_division=0),'top5_accuracy':float((top5==y[test,None]).any(1).mean()),'accuracy_ci95':bootstrap_mean((pred==y[test]).astype(float)),'fit_seconds':fit_seconds,'test_prediction_seconds':infer_seconds,'trials':trials}
  metrics.append(item); preds[name]=pred; decisions[name]=dec
  joblib.dump({'pipeline':pipe,'method':schema['methods'][mi],'columns':cols,'seed':SEED,'feature_schema':schema,'validation_accuracy':best[0]},M/f'{name}.joblib',compress=3)
  print(f'TEST {name} acc={acc:.4f} f1={item["macro_f1"]:.4f} top5={item["top5_accuracy"]:.4f}',flush=True)
  dump('classification_metrics.json',metrics)
 pd.DataFrame([{k:v for k,v in r.items() if k not in ('trials','accuracy_ci95')} for r in metrics]).to_csv(R/'classification_metrics.csv',index=False)
 predframe=df.loc[test,['image_id','class_id','relative_path']].copy()
 for k,v in preds.items(): predframe[k]=v
 predframe.to_csv(R/'test_predictions.csv',index=False)
 selected=joblib.load(M/f'{selected_name}.joblib'); joblib.dump(selected,M/'production.joblib',compress=3)
 selected_pred=preds[selected_name]
 dump('selected_model.json',{'experiment':selected_name,'selection_rule':'Highest development validation accuracy among five processing pipelines; C selected on validation; refit on all 5994 official training images. No test-based selection.','validation_accuracy':best_validation})
 report=classification_report(y[test],selected_pred,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(R/'per_class_report.csv')
 cm=confusion_matrix(y[test],selected_pred,labels=np.arange(1,201)); pd.DataFrame(cm,index=np.arange(1,201),columns=np.arange(1,201)).to_csv(R/'confusion_matrix.csv')
 plt.figure(figsize=(8,7)); plt.imshow(cm,cmap='Blues',interpolation='nearest'); plt.colorbar(label='Image count'); plt.xlabel('Predicted class index'); plt.ylabel('True class index'); plt.title(f'200-class confusion matrix | {selected_name}'); savefig('confusion_matrix.png')
 segrows=[]
 for i,m in enumerate(schema['methods']):
  segrows.append({'method':m,'test_count':int(test.sum()),'mean_IoU':float(seg[test,i,0].mean()),'mean_Dice':float(seg[test,i,1].mean()),'median_IoU':float(np.median(seg[test,i,0])),'IoU_ci95':bootstrap_mean(seg[test,i,0]),'fallback_count_all':int(falls[:,i].sum()),'mean_compute_seconds':float(timing[:,i].mean())})
 dump('segmentation_metrics.json',segrows); pd.DataFrame(segrows).to_csv(R/'segmentation_metrics.csv',index=False)
 perimage=df[['image_id','class_id','is_train']].copy()
 for i,m in enumerate(schema['methods']): perimage[f'{m}_IoU']=seg[:,i,0]; perimage[f'{m}_Dice']=seg[:,i,1]
 perimage.to_csv(R/'segmentation_per_image.csv',index=False)
 dif=(preds['full']==y[test]).astype(float)-(preds['raw']==y[test]).astype(float)
 dump('paired_comparisons.json',{'full_minus_raw_accuracy':float(dif.mean()),'image_bootstrap_ci95':bootstrap_mean(dif),'full_minus_otsu_IoU':float((seg[test,full,0]-seg[test,1,0]).mean()),'IoU_difference_ci95':bootstrap_mean(seg[test,full,0]-seg[test,1,0]),'note':'Intervals resample test images and do not model species-level dependence.'})
 plt.figure(figsize=(10,4.5)); xx=np.arange(len(metrics)); plt.bar(xx,[r['test_accuracy']*100 for r in metrics],color='#276c79'); plt.xticks(xx,[r['experiment'] for r in metrics],rotation=30,ha='right'); plt.ylabel('Top-1 accuracy (%)'); plt.title('Full CUB official test split | classical features + RBF SVM'); savefig('classification_comparison.png')
 plt.figure(figsize=(7,4)); plt.bar([r['method'] for r in segrows],[r['mean_IoU'] for r in segrows],color='#559b81'); plt.ylim(0,1); plt.ylabel('Mean IoU'); plt.title('Automatic segmentation | 5,794 official test images'); savefig('segmentation_comparison.png')
 # PCA preprocessing fitted exclusively on official train images.
 scaler=StandardScaler().fit(x[train,full]); pca=PCA(n_components=12,random_state=SEED).fit(scaler.transform(x[train,full])); pc=pca.transform(scaler.transform(x[:,full])); joblib.dump({'scaler':scaler,'pca':pca},M/'pca.joblib')
 pdf=df[['image_id','class_id','is_train']].copy(); pdf['PC1']=pc[:,0]; pdf['PC2']=pc[:,1]; pdf.to_csv(R/'pca_coordinates.csv',index=False)
 cross=pd.read_csv(BASE/'metadata/species_crosswalk.csv').set_index('class_id'); ecological_analysis(x[:,full],df,cross)
 colors=df.class_id.map(cross['Habitat.Density']); plt.figure(figsize=(8,5)); sc=plt.scatter(pc[test,0],pc[test,1],c=colors[test].fillna(0),s=5,cmap='viridis',alpha=.5); plt.colorbar(sc,label='AVONET habitat density category (0=unmatched)'); plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'); plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'); plt.title('Test-image feature space | association, not evolutionary inference'); savefig('pca_habitat.png')
 dump('pca_summary.json',{'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'fit_split':'5994 official training images','interpretation':'Exploratory image-feature space; not a phylogeny or proof of adaptation.'})
 dump('run_environment.json',{'python':sys.version,'platform':platform.platform(),'opencv':cv2.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'seed':SEED,'train':int(train.sum()),'test':int(test.sum()),'images':len(df),'classes':200,'training_evaluation_seconds':time.time()-started})
 print('ALL EXPERIMENTS COMPLETE',flush=True)

def ecological_analysis(z,df,cross):
 targets=['Mass','Hand-Wing.Index','Beak.Length_Nares']; table=cross[targets].apply(pd.to_numeric,errors='coerce'); usable=table.dropna().index
 classes=df.class_id.to_numpy(); valid=np.isin(classes,usable); train=df.is_train.to_numpy()==1; test=~train
 yy=table.reindex(classes).to_numpy(); yy[:,0]=np.log10(yy[:,0])
 # Hold entire species out, with no image from them in training or model selection.
 train_species,test_species=train_test_split(np.array(sorted(usable)),test_size=.2,random_state=SEED)
 tr=train&np.isin(classes,train_species); te=test&np.isin(classes,test_species)
 alpha=1000. # fixed a priori; no tuning on held-out species
 target_scaler=StandardScaler().fit(yy[tr]); pipe=make_pipeline(StandardScaler(),Ridge(alpha=alpha)); pipe.fit(z[tr],target_scaler.transform(yy[tr])); predicted=target_scaler.inverse_transform(pipe.predict(z[te])); truth=yy[te]; baseline=np.tile(yy[tr].mean(0),(te.sum(),1))
 rows=[]
 for j,n in enumerate(['log10_mass_g','hand_wing_index','beak_nares_mm']):
  rows.append({'target':n,'test_images':int(te.sum()),'train_species':len(train_species),'test_species':len(test_species),'MAE':mean_absolute_error(truth[:,j],predicted[:,j]),'R2':r2_score(truth[:,j],predicted[:,j]),'mean_baseline_MAE':mean_absolute_error(truth[:,j],baseline[:,j]),'mean_baseline_R2':r2_score(truth[:,j],baseline[:,j])})
 dump('ecology_metrics.json',rows)
 pred_df=df.loc[te,['image_id','class_id']].copy()
 for j,n in enumerate(targets): pred_df[f'true_{n}']=truth[:,j]; pred_df[f'pred_{n}']=predicted[:,j]
 pred_df.to_csv(R/'ecology_predictions.csv',index=False)
 # Also report equal-species-weighted errors; image counts differ by species.
 agg=pred_df.groupby('class_id').mean(numeric_only=True); equal=[]
 for n in targets: equal.append({'target':n,'species_mean_prediction_MAE':mean_absolute_error(agg[f'true_{n}'],agg[f'pred_{n}']),'species_mean_prediction_R2':r2_score(agg[f'true_{n}'],agg[f'pred_{n}'])})
 dump('ecology_species_metrics.json',equal)
 dump('ecology_split.json',{'train_species':train_species.tolist(),'held_out_species':test_species.tolist(),'targets':targets,'mass_transform':'log10(g)','ridge_alpha':alpha,'matched_species':len(usable),'train_images':int(tr.sum()),'test_images':int(te.sum()),'protocol':'Training uses official train images of training species; evaluation uses official test images of wholly held-out species. All unmatched images remain included in the 200-class image and segmentation experiments.'})
 fig,axes=plt.subplots(1,3,figsize=(11,3.5))
 for j,ax in enumerate(axes):
  ax.scatter(truth[:,j],predicted[:,j],s=4,alpha=.25,color='#276c79'); lo=min(truth[:,j].min(),predicted[:,j].min()); hi=max(truth[:,j].max(),predicted[:,j].max()); ax.plot([lo,hi],[lo,hi],'--',color='gray'); ax.set_title(rows[j]['target']); ax.set_xlabel('AVONET species mean'); ax.set_ylabel('Image-based prediction')
 savefig('ecology_predictions.png')
 # Production ecological regressor refit on all matched official training images.
 fit=train&valid; target_scaler=StandardScaler().fit(yy[fit]); pipe=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(z[fit],target_scaler.transform(yy[fit])); joblib.dump({'pipeline':pipe,'target_scaler':target_scaler,'targets':targets,'mass_log10':True,'method':'full','evaluation':rows},M/'ecology.joblib',compress=3)

if __name__=='__main__':
 with threadpool_limits(limits=2): run()
