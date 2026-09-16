"""Classical OpenCV pipeline. No annotation, class name or filename enters features."""
from pathlib import Path
import cv2
import numpy as np

SEED=2026
SIZE=160
METHODS=('raw','otsu','full','no_filter','no_align')

def read_image(path):
 im=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
 if im is None: raise ValueError(f'Cannot decode image: {path}')
 return im

def write_image(path,im):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 ok,buf=cv2.imencode(path.suffix,im)
 if not ok: raise ValueError('Image encoding failed')
 buf.tofile(str(path))

def resize(im):
 h,w=im.shape[:2]; scale=SIZE/max(h,w)
 return cv2.resize(im,(max(8,round(w*scale)),max(8,round(h*scale))),interpolation=cv2.INTER_AREA)

def clean(mask):
 m=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 n,labels,stats,centers=cv2.connectedComponentsWithStats(m,8)
 if n<=1: return m
 h,w=m.shape
 score=stats[1:,cv2.CC_STAT_AREA]/(1+2*((centers[1:,0]/w-.5)**2+(centers[1:,1]/h-.5)**2))
 return (labels==1+np.argmax(score)).astype('uint8')

def segment(im,method='full',seed=SEED):
 h,w=im.shape[:2]
 if method=='raw': return np.ones((h,w),np.uint8),im.copy(),False
 filtered=cv2.bilateralFilter(im,5,35,35) if method!='no_filter' else im.copy()
 lab=cv2.cvtColor(filtered,cv2.COLOR_BGR2LAB).astype(np.float32)
 boundary=np.concatenate([lab[:3].reshape(-1,3),lab[-3:].reshape(-1,3),lab[:,:3].reshape(-1,3),lab[:,-3:].reshape(-1,3)])
 # Eight boundary colour centres model a multimodal background.
 cv2.setRNGSeed(int(seed))
 _,_,centres=cv2.kmeans(boundary,8,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,15,.5),1,cv2.KMEANS_PP_CENTERS)
 dist=np.sqrt(((lab[:,:,None,:]-centres[None,None,:,:])**2).sum(3).min(2))
 yy,xx=np.mgrid[:h,:w]; prior=np.exp(-((xx/w-.5)**2+(yy/h-.5)**2)/.22)
 score=dist*(.5+.5*prior)
 score=np.uint8(np.clip(score/(np.percentile(score,99)+1e-6)*255,0,255))
 _,binary=cv2.threshold(score,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 m=clean(binary)
 fallback=False
 if method!='otsu':
  gm=np.full((h,w),cv2.GC_PR_BGD,np.uint8)
  gm[m>0]=cv2.GC_PR_FGD
  eroded=cv2.erode(m,np.ones((3,3),np.uint8))
  gm[(eroded>0)&(score>=np.percentile(score,75))]=cv2.GC_FGD
  gm[:2]=gm[-2:]=cv2.GC_BGD; gm[:,:2]=gm[:,-2:]=cv2.GC_BGD
  if not (gm==cv2.GC_FGD).any(): gm[h//2,w//2]=cv2.GC_FGD
  try:
   cv2.setRNGSeed(int(seed))
   cv2.grabCut(filtered,gm,None,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
   m=clean(((gm==1)|(gm==3)).astype('uint8'))
  except cv2.error: fallback=True
 if m.sum()<20:
  fallback=True; m=np.zeros((h,w),np.uint8); cv2.ellipse(m,(w//2,h//2),(max(2,w//4),max(2,h//4)),0,0,360,1,-1)
 return m,filtered,fallback

def aligned_crop(im,mask,align=True):
 ys,xs=np.nonzero(mask); h,w=mask.shape
 if align and len(xs)>20:
  pts=np.column_stack([xs,ys]).astype(np.float64)
  val,vec=np.linalg.eigh(np.cov(pts.T)); v=vec[:,np.argmax(val)]
  angle=np.degrees(np.arctan2(v[1],v[0])); angle=(angle+90)%180-90
  side=int(np.ceil(np.hypot(h,w)))+4; cx,cy=xs.mean(),ys.mean()
  mat=cv2.getRotationMatrix2D((float(cx),float(cy)),float(angle),1)
  mat[:,2]+=[side/2-cx,side/2-cy]
  im=cv2.warpAffine(im,mat,(side,side),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  mask=cv2.warpAffine(mask,mat,(side,side),flags=cv2.INTER_NEAREST)
  ys,xs=np.nonzero(mask)
 x0,x1=xs.min(),xs.max()+1; y0,y1=ys.min(),ys.max()+1
 crop=im[y0:y1,x0:x1].copy(); m=mask[y0:y1,x0:x1]
 # Letterbox to square without altering aspect ratio.
 side=max(crop.shape[:2]); square=np.zeros((side,side,3),np.uint8); sm=np.zeros((side,side),np.uint8)
 dy=(side-crop.shape[0])//2; dx=(side-crop.shape[1])//2
 square[dy:dy+crop.shape[0],dx:dx+crop.shape[1]]=crop; sm[dy:dy+crop.shape[0],dx:dx+crop.shape[1]]=m
 square=cv2.resize(square,(64,64)); sm=cv2.resize(sm,(64,64),interpolation=cv2.INTER_NEAREST)
 square[sm==0]=0
 return square,sm

def glcm_features(gray,mask):
 q=(gray//16).astype(int); result=[]; names=[]; a,b=np.mgrid[:16,:16]
 for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
  ya=slice(max(0,-dy),min(64,64-dy)); yb=slice(max(0,dy),min(64,64+dy))
  xa=slice(max(0,-dx),min(64,64-dx)); xb=slice(max(0,dx),min(64,64+dx))
  valid=(mask[ya,xa]>0)&(mask[yb,xb]>0)
  counts=np.bincount((q[ya,xa][valid]*16+q[yb,xb][valid]),minlength=256).reshape(16,16).astype(float)
  p=(counts+counts.T); p/=max(p.sum(),1)
  ma=(p*a).sum(); mb=(p*b).sum(); sa=np.sqrt((p*(a-ma)**2).sum()); sb=np.sqrt((p*(b-mb)**2).sum())
  result.extend([(p*(a-b)**2).sum()/225,(p/(1+(a-b)**2)).sum(),(p*p).sum(),(p*(a-ma)*(b-mb)).sum()/(sa*sb+1e-9)])
  names.extend([f'glcm_{dx}_{dy}_{n}' for n in ['contrast','homogeneity','energy','correlation']])
 return result,names

def feature_vector(im,mask,align=True):
 crop,m=aligned_crop(im,mask,align); hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
 values=[]; names=[]; groups={}
 def add(group,v,n):
  start=len(values); values.extend(np.ravel(v).tolist()); names.extend(n); groups[group]=(start,len(values))
 color=[]; cn=[]
 for i,(y0,y1,x0,x1) in enumerate([(0,64,0,64),(0,32,0,32),(0,32,32,64),(32,64,0,32),(32,64,32,64)]):
  reg=hsv[y0:y1,x0:x1]; rm=m[y0:y1,x0:x1]
  for ch,bins,upper in [(0,18,180),(1,8,256),(2,8,256)]:
   hist=cv2.calcHist([reg],[ch],rm,[bins],[0,upper]).ravel(); hist=np.sqrt(hist/(hist.sum()+1e-9))
   color.extend(hist); cn.extend([f'color_{i}_{ch}_{j}' for j in range(bins)])
 add('color',color,cn)
 gl,gn=glcm_features(gray,m)
 # LBP via eight fixed neighbours and a 16-bin histogram.
 centre=gray[1:-1,1:-1]; lbp=np.zeros_like(centre)
 for bit,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]): lbp|=((gray[1+dy:63+dy,1+dx:63+dx]>=centre).astype(np.uint8)<<bit)
 hist=np.bincount((lbp[m[1:-1,1:-1]>0]//16),minlength=16).astype(float); hist/=max(hist.sum(),1)
 add('texture',gl+hist.tolist(),gn+[f'lbp_{i}' for i in range(16)])
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); c=max(contours,key=cv2.contourArea)
 area=float(m.sum()); per=cv2.arcLength(c,True); hull=cv2.contourArea(cv2.convexHull(c)); x,y,w,h=cv2.boundingRect(c)
 hu=cv2.HuMoments(cv2.moments(m)).ravel(); hu=-np.sign(hu)*np.log10(np.abs(hu)+1e-12)
 shape=[mask.mean(),w/max(h,1),area/(w*h),4*np.pi*area/(per*per+1e-6),cv2.contourArea(c)/(hull+1e-6)]+hu.tolist()
 add('shape',shape,['foreground_ratio','aspect_ratio','extent','compactness','solidity']+[f'hu_{i}' for i in range(7)])
 edge=(cv2.Canny(gray,60,140)>0)&(m>0); sizes=[2,4,8,16]; counts=[np.any(edge.reshape(64//s,s,64//s,s),axis=(1,3)).sum() for s in sizes]
 fractal=float(np.polyfit(np.log(64/np.array(sizes)),np.log(np.maximum(counts,1)),1)[0])
 sym=[]
 for axis in [0,1]:
  flip=np.flip(m,axis); sym.append(float(np.minimum(m,flip).sum()/max(np.maximum(m,flip).sum(),1)))
  valid=(m>0)&(flip>0); a=gray[valid].astype(float); b=np.flip(gray,axis)[valid].astype(float)
  corr=float(np.corrcoef(a,b)[0,1]) if len(a)>4 and a.std()>1e-6 and b.std()>1e-6 else 0.
  sym.append(corr)
 add('complexity',[fractal]+sym,['edge_box_dimension','horizontal_mask_symmetry','horizontal_texture_correlation','vertical_mask_symmetry','vertical_texture_correlation'])
 hog=cv2.HOGDescriptor((64,64),(16,16),(16,16),(8,8),9).compute(gray).ravel()
 add('hog',hog,[f'hog_{i}' for i in range(len(hog))])
 return np.nan_to_num(np.array(values,np.float32)),names,groups,crop,m

def process_image(path,method='full',seed=SEED):
 cv2.setNumThreads(1)
 im=resize(read_image(path)); mask,filtered,fallback=segment(im,method,seed)
 vec,names,groups,crop,cm=feature_vector(filtered,mask,method not in ('raw','no_align'))
 return {'vector':vec,'names':names,'groups':groups,'image':im,'filtered':filtered,'mask':mask,'crop':crop,'crop_mask':cm,'fallback':fallback}

def segmentation_scores(mask,gt):
 gt=cv2.resize(gt,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)>127; p=mask>0
 inter=np.logical_and(p,gt).sum(); union=np.logical_or(p,gt).sum()
 return float(inter/max(union,1)),float(2*inter/max(p.sum()+gt.sum(),1))
