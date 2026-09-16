"""Choose the deployed candidate using validation accuracy only, including ablations."""
from pathlib import Path
import json,joblib
BASE=Path(__file__).resolve().parent
def main():
 rows=json.loads((BASE/'results/classification_metrics.json').read_text())
 # No test metric enters this selection.
 selected=max(rows,key=lambda r:r['validation_accuracy'])
 name=selected['experiment'];obj=joblib.load(BASE/f'models/{name}.joblib');joblib.dump(obj,BASE/'models/production.joblib',compress=3)
 record={'experiment':name,'selection_rule':'Highest development validation accuracy across all nine predeclared candidates, including feature subsets; no test metric enters selection. Candidate refit on all 5994 official training images.','validation_accuracy':selected['validation_accuracy']}
 (BASE/'results/selected_model.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 # Refresh the production-model classwise report and confusion matrix.
 import pandas as pd,numpy as np
 from sklearn.metrics import classification_report,confusion_matrix
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 df=pd.read_csv(BASE/'results/test_predictions.csv');y=df.class_id;pred=df[name]
 pd.DataFrame(classification_report(y,pred,output_dict=True,zero_division=0)).T.to_csv(BASE/'results/per_class_report.csv')
 cm=confusion_matrix(y,pred,labels=np.arange(1,201));pd.DataFrame(cm,index=np.arange(1,201),columns=np.arange(1,201)).to_csv(BASE/'results/confusion_matrix.csv')
 plt.figure(figsize=(8,7));plt.imshow(cm,cmap='Blues',interpolation='nearest');plt.colorbar(label='Image count');plt.xlabel('Predicted class index');plt.ylabel('True class index');plt.title(f'200-class confusion matrix | {name}');plt.tight_layout();plt.savefig(BASE/'results/figures/confusion_matrix.png',dpi=160,bbox_inches='tight');plt.close()
 print('PRODUCTION SELECTED BY VALIDATION',name,selected['validation_accuracy'],flush=True)
if __name__=='__main__':main()
