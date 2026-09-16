from pathlib import Path
import json,subprocess,sys

base=Path(__file__).resolve().parent
def main():
 cfg=json.loads((base/'config.json').read_text(encoding='utf-8'))
 for script in ['download_data.py','prepare_metadata.py','extract_features.py','train_evaluate.py','select_production.py','cluster_analysis.py','make_examples.py','verify.py']:
  print(f'RUN {script}',flush=True)
  args=['--root',cfg['data_root']] if script=='download_data.py' else []
  subprocess.run([sys.executable,str(base/script),*args],cwd=base,check=True)

if __name__=='__main__': main()
