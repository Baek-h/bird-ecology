"""Train-fitted K-means in PCA space; full test-set silhouette, no subsampling."""
from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import SEED

BASE=Path(__file__).resolve().parent
def main():
 r=BASE/'results'; df=pd.read_csv(r/'image_index.csv'); x=np.load(r/'features.npz')['features'][:,2,:]; model=joblib.load(BASE/'models/pca.joblib'); z=model['pca'].transform(model['scaler'].transform(x)); train=df.is_train.to_numpy()==1
 km=KMeans(n_clusters=8,n_init=10,random_state=SEED).fit(z[train]); labels=km.predict(z)
 score=float(silhouette_score(z[~train],labels[~train]))
 out=df[['image_id','class_id','is_train']].copy();out['cluster']=labels;out.to_csv(r/'cluster_assignments.csv',index=False)
 cross=pd.read_csv(BASE/'metadata/species_crosswalk.csv');joined=out.merge(cross[['class_id','Habitat','Trophic.Niche','Mass','Hand-Wing.Index']],on='class_id'); summary=joined.groupby('cluster').agg(images=('image_id','size'),species=('class_id','nunique'),mean_species_mass=('Mass','mean'),mean_hand_wing_index=('Hand-Wing.Index','mean'));summary.to_csv(r/'cluster_summary.csv')
 pd.crosstab(joined.cluster,joined.Habitat).to_csv(r/'cluster_habitat_counts.csv')
 joblib.dump(km,BASE/'models/kmeans.joblib')
 (r/'cluster_metrics.json').write_text(json.dumps({'clusters':8,'PCA_dimensions':12,'silhouette_full_test':score,'test_images':int((~train).sum()),'seed':SEED,'fit_split':'official training images only','interpretation':'Exploratory clusters of image features; no phylogenetic or causal evolutionary inference. Ecological summaries repeat species means per image and are descriptive only.'},indent=2),encoding='utf-8')
 plt.figure(figsize=(8,5));sc=plt.scatter(z[~train,0],z[~train,1],c=labels[~train],cmap='tab10',s=6,alpha=.5);plt.colorbar(sc,label='K-means cluster');plt.xlabel('PC1');plt.ylabel('PC2');plt.title(f'8 exploratory clusters | test silhouette {score:.3f}');plt.tight_layout();plt.savefig(r/'figures/pca_clusters.png',dpi=160);plt.close();print('CLUSTER ANALYSIS COMPLETE',score,flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
