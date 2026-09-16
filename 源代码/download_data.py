"""Download complete official datasets, verify MD5, safely unpack archives."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, tarfile, time
import requests

SOURCES = [
 ('CUB_200_2011.tgz','https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1','97eceeb196236b17998738112f37df78'),
 ('segmentations.tgz','https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1','4d47ba1228eae64f2fa547c47bc65255'),
 ('AVONET Supplementary dataset 1.xlsx','https://ndownloader.figshare.com/files/34480856','1445afdcfb6df784010c2ca034544bc8'),
]

def digest(p):
 h=hashlib.md5()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def download(item,root):
 name,url,expected=item; p=root/name
 if not p.exists() or digest(p)!=expected:
  part=p.with_suffix(p.suffix+'.part')
  for attempt in range(5):
   try:
    offset=part.stat().st_size if part.exists() else 0
    r=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120))
    r.raise_for_status()
    if r.status_code not in (200,206): raise RuntimeError(f'Unexpected response {r.status_code}')
    append=offset>0 and r.status_code==206
    total=int(r.headers.get('Content-Length',0))+(offset if append else 0); n=offset if append else 0; last=time.time()
    with part.open('ab' if append else 'wb') as f:
     for b in r.iter_content(1024*1024):
      f.write(b); n+=len(b)
      if time.time()-last>20: print(f'{name}: {n/1e6:.1f}/{total/1e6:.1f} MB',flush=True); last=time.time()
    if digest(part)!=expected:
     part.unlink() # discard only this downloader's corrupt partial file
     raise RuntimeError('MD5 mismatch; restarting download')
    part.replace(p); break
   except Exception as e:
    print(f'{name}: attempt {attempt+1}: {e}',flush=True)
    if attempt==4: raise
    time.sleep(2)
 print(f'VERIFIED {name} {expected}',flush=True)
 if name.endswith('.tgz'):
  sentinel=root/(name+'.extracted')
  if not sentinel.exists():
   with tarfile.open(p) as tar:
    for m in tar:
     dest=(root/m.name).resolve()
     if not dest.is_relative_to(root.resolve()) or m.issym() or m.islnk(): raise ValueError('Unsafe archive member')
     tar.extract(m,root)
   sentinel.write_text(expected)
   print(f'EXTRACTED {name}',flush=True)
 return {'name':name,'url':url,'md5':expected,'bytes':p.stat().st_size}

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default=r'D:\图像处理综合实践\datasets'); args=ap.parse_args()
 root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: records=list(ex.map(lambda s:download(s,root),SOURCES))
 (root/'download_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 print('ALL DATASETS READY',flush=True)
