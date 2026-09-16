"""Offline Tk desktop interface: browse, retrieve, analyze, batch and statistics."""
from pathlib import Path
import argparse,json,queue,threading,time
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import pandas as pd
import cv2
from PIL import Image,ImageTk
from inference import Engine,flat_result,BASE
from core import read_image

BG='#f3f6f7'; INK='#18313d'; TEAL='#176d73'
class App:
 def __init__(self,root):
  self.root=root; root.title('鸟类图像处理与生态评价 · 综合实践'); root.geometry('1280x820'); root.minsize(1100,740); root.configure(bg=BG)
  self.cfg=json.loads((BASE/'config.json').read_text(encoding='utf-8')); self.data=Path(self.cfg['data_root'])
  self.index=pd.read_csv(BASE/'results/image_index.csv'); self.cross=pd.read_csv(BASE/'metadata/species_crosswalk.csv'); self.index=self.index.merge(self.cross[['class_id','common_name']],on='class_id')
  self.engine=Engine(); self.queue=queue.Queue(); self.photos=[]; self.current=None; self.batch_rows=[]; self.page=0; self.filtered=self.index; self.last_result=None; self.busy=False
  style=ttk.Style(); style.theme_use('clam'); style.configure('.',font=('Microsoft YaHei UI',10)); style.configure('TFrame',background=BG); style.configure('TLabel',background=BG,foreground=INK); style.configure('TButton',padding=(12,7)); style.configure('Accent.TButton',background=TEAL,foreground='white'); style.configure('Treeview',rowheight=27,fieldbackground='white',background='white'); style.configure('Treeview.Heading',font=('Microsoft YaHei UI',10,'bold'),padding=6); style.configure('TNotebook.Tab',padding=(18,9))
  header=tk.Frame(root,bg=INK,height=78); header.pack(fill='x'); header.pack_propagate(False)
  tk.Label(header,text='鸟类图像处理与生态评价',font=('Microsoft YaHei UI',21,'bold'),fg='white',bg=INK).pack(side='left',padx=24,pady=18)
  tk.Label(header,text='CUB 200 · OpenCV · AVONET',font=('Segoe UI',11),fg='#b9d4d8',bg=INK).pack(side='right',padx=24)
  self.tabs=ttk.Notebook(root); self.tabs.pack(fill='both',expand=True,padx=18,pady=14)
  self.analysis=ttk.Frame(self.tabs,padding=12); self.batch=ttk.Frame(self.tabs,padding=16); self.stats=ttk.Frame(self.tabs,padding=16)
  self.tabs.add(self.analysis,text='图像浏览与分析'); self.tabs.add(self.batch,text='批量处理'); self.tabs.add(self.stats,text='数据与实验统计')
  self.build_analysis(); self.build_batch(); self.build_stats()
  self.status=tk.StringVar(value='就绪 · 模型已加载 · 所有处理均在本机运行'); ttk.Label(root,textvariable=self.status,padding=(20,5)).pack(fill='x')
  self.refresh(); root.after(100,self.poll)
 def build_analysis(self):
  left=ttk.Frame(self.analysis,width=310); left.pack(side='left',fill='y',padx=(0,16)); left.pack_propagate(False)
  ttk.Label(left,text='数据集检索',font=('Microsoft YaHei UI',14,'bold')).pack(anchor='w',pady=(0,8))
  self.search=tk.StringVar(); entry=ttk.Entry(left,textvariable=self.search); entry.pack(fill='x'); entry.bind('<Return>',lambda e:self.filter())
  controls=ttk.Frame(left); controls.pack(fill='x',pady=8)
  self.split=tk.StringVar(value='全部'); ttk.Combobox(controls,textvariable=self.split,values=['全部','训练集','测试集'],state='readonly',width=9).pack(side='left'); ttk.Button(controls,text='检索',command=self.filter).pack(side='right')
  self.tree=ttk.Treeview(left,columns=('id','name'),show='headings',height=10,selectmode='browse'); self.tree.heading('id',text='编号'); self.tree.heading('name',text='鸟类名称'); self.tree.column('id',width=55,stretch=False); self.tree.column('name',width=222); self.tree.pack(fill='both',expand=True); self.tree.bind('<<TreeviewSelect>>',self.choose)
  nav=ttk.Frame(left); nav.pack(fill='x',pady=7); ttk.Button(nav,text='上一页',command=lambda:self.paginate(-1)).pack(side='left'); ttk.Button(nav,text='下一页',command=lambda:self.paginate(1)).pack(side='right'); self.count=tk.StringVar(); ttk.Label(left,textvariable=self.count).pack(anchor='w')
  ttk.Button(left,text='导入本地图像',command=self.open_image).pack(fill='x',pady=(12,5)); self.runbutton=ttk.Button(left,text='开始分析',style='Accent.TButton',command=self.analyze); self.runbutton.pack(fill='x')
  right=ttk.Frame(self.analysis); right.pack(side='left',fill='both',expand=True)
  self.title=tk.StringVar(value='选择图像，查看分割与生态性状'); ttk.Label(right,textvariable=self.title,font=('Microsoft YaHei UI',14,'bold')).pack(anchor='w',pady=(0,10))
  previews=ttk.Frame(right); previews.pack(fill='x'); self.image_labels=[]; self.blank=ImageTk.PhotoImage(Image.new('RGB',(250,210),'#e3eaed'))
  for name in ['输入图像','自动前景区域','特征输入区域']:
   cell=ttk.Frame(previews); cell.pack(side='left',fill='both',expand=True,padx=3); ttk.Label(cell,text=name).pack(pady=5); lab=tk.Label(cell,bg='#e3eaed',image=self.blank,width=250,height=210); lab.pack(fill='both',expand=True); self.image_labels.append(lab)
  self.detail=tk.Text(right,height=15,wrap='word',font=('Microsoft YaHei UI',10),bg='white',fg=INK,relief='flat',padx=14,pady=12); self.detail.pack(fill='both',expand=True,pady=(12,0)); self.set_detail('使用左侧名称或文件名检索图像，也可以导入 JPG、PNG 等本地图像。\n\n分析包含：候选物种、图像纹理与对称性、AVONET 物种平均性状和探索性回归结果。\n\n生态性状属于物种层面的参考信息，不能据此判断照片中个体的健康或适应能力。')
 def build_batch(self):
  ttk.Label(self.batch,text='多张图像处理与结果导出',font=('Microsoft YaHei UI',16,'bold')).pack(anchor='w',pady=(0,12))
  ttk.Label(self.batch,text='选择多张图像后自动逐张分析。失败文件保留错误信息，其余文件继续处理。').pack(anchor='w')
  bar=ttk.Frame(self.batch); bar.pack(fill='x',pady=14); ttk.Button(bar,text='选择图像并运行',style='Accent.TButton',command=self.batch_open).pack(side='left'); ttk.Button(bar,text='导出结果 CSV',command=self.export_batch).pack(side='left',padx=10)
  self.progress=ttk.Progressbar(self.batch); self.progress.pack(fill='x',pady=10)
  self.batchtree=ttk.Treeview(self.batch,columns=('file','prediction','time','state'),show='headings');
  for c,label,width in [('file','文件',430),('prediction','预测物种',280),('time','耗时 秒',100),('state','状态',170)]: self.batchtree.heading(c,text=label); self.batchtree.column(c,width=width)
  self.batchtree.pack(fill='both',expand=True)
  ttk.Label(self.batch,text='CSV 同时包含图像特征、物种参考性状和回归估计；SVM 分数不表示概率。').pack(anchor='w',pady=12)
 def build_stats(self):
  ttk.Label(self.stats,text='完整数据集与实际实验结果',font=('Microsoft YaHei UI',16,'bold')).pack(anchor='w',pady=(0,14))
  cards=ttk.Frame(self.stats); cards.pack(fill='x')
  for title,value in [('图像总数',f'{len(self.index):,}'),('鸟类类别',str(self.index.class_id.nunique())),('训练 / 测试','5,994 / 5,794'),('AVONET 对应',f'{int(self.cross.mapped.sum())} / 200')]:
   c=tk.Frame(cards,bg='white',padx=22,pady=12); c.pack(side='left',expand=True,fill='x',padx=5); tk.Label(c,text=value,font=('Segoe UI',22,'bold'),fg=TEAL,bg='white').pack(anchor='w'); tk.Label(c,text=title,font=('Microsoft YaHei UI',10),fg=INK,bg='white').pack(anchor='w')
  result=json.loads((BASE/'results/classification_metrics.json').read_text()); chosen=json.loads((BASE/'results/selected_model.json').read_text())['experiment']; msg=f'当前模型：{chosen}（依据验证集选择）\n官方测试集只用于最终评价；分割真值不进入特征提取。'
  ttk.Label(self.stats,text=msg).pack(anchor='w',pady=16)
  tr=ttk.Treeview(self.stats,columns=('method','acc','top5','f1','val'),show='headings',height=10)
  for c,label,width in [('method','实验方案',300),('acc','测试准确率',160),('top5','Top-5',160),('f1','Macro-F1',160),('val','验证准确率',160)]: tr.heading(c,text=label); tr.column(c,width=width)
  for r in result: tr.insert('',tk.END,values=(r['experiment'],f"{r['test_accuracy']:.2%}",f"{r['top5_accuracy']:.2%}",f"{r['macro_f1']:.4f}",f"{r['validation_accuracy']:.2%}"))
  tr.pack(fill='x'); eco=json.loads((BASE/'results/ecology_metrics.json').read_text()); ttk.Label(self.stats,text='跨物种生态回归：'+'；'.join(f"{r['target']} R²={r['R2']:.3f}" for r in eco),wraplength=1100).pack(anchor='w',pady=16)
 def filter(self):
  q=self.search.get().strip().lower(); df=self.index
  if q: df=df[df.common_name.str.lower().str.contains(q,regex=False)|df.relative_path.str.lower().str.contains(q,regex=False)]
  if self.split.get()!='全部': df=df[df.is_train==(1 if self.split.get()=='训练集' else 0)]
  self.filtered=df; self.page=0; self.refresh()
 def paginate(self,delta): self.page=max(0,min(self.page+delta,max(0,(len(self.filtered)-1)//100))); self.refresh()
 def refresh(self):
  self.tree.delete(*self.tree.get_children())
  for _,row in self.filtered.iloc[self.page*100:(self.page+1)*100].iterrows(): self.tree.insert('',tk.END,iid=str(row.image_id),values=(row.image_id,row.common_name))
  self.count.set(f'{len(self.filtered):,} 张图像 · 第 {self.page+1} 页')
 def choose(self,event=None):
  sel=self.tree.selection()
  if not sel:return
  row=self.index[self.index.image_id==int(sel[0])].iloc[0]; self.current=self.data/'CUB_200_2011/images'/row.relative_path; self.title.set(row.common_name); self.show_images([read_image(self.current)]); self.set_detail(f'图像编号：{row.image_id}\n数据划分：'+('训练集' if row.is_train else '测试集')+'\n\n点击“开始分析”运行模型。浏览标签仅用于显示，不参与模型预测。')
 def open_image(self):
  p=filedialog.askopenfilename(filetypes=[('图像','*.jpg *.jpeg *.png *.bmp *.webp'),('所有文件','*.*')])
  if p:
   try:self.current=Path(p); self.title.set(self.current.name); self.show_images([read_image(p)]); self.set_detail('本地图像已载入，点击“开始分析”。')
   except Exception as exc:messagebox.showerror('读取失败',str(exc))
 def show_images(self,images):
  self.photos=[]
  for i,lab in enumerate(self.image_labels):
   if i<len(images):
    im=images[i]; rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB) if im.ndim==3 else im; pil=Image.fromarray(rgb); pil.thumbnail((250,210)); canvas=Image.new('RGB',(250,210),'#e3eaed'); canvas.paste(pil,((250-pil.width)//2,(210-pil.height)//2)); photo=ImageTk.PhotoImage(canvas); self.photos.append(photo); lab.configure(image=photo,width=250,height=210)
   else:lab.configure(image=self.blank,width=250,height=210)
 def set_detail(self,text):self.detail.configure(state='normal'); self.detail.delete('1.0',tk.END); self.detail.insert('1.0',text); self.detail.configure(state='disabled')
 def analyze(self):
  if self.busy:return
  if self.current is None:messagebox.showinfo('选择图像','请先选择或导入图像。'); return
  self.busy=True; self.runbutton.configure(state='disabled'); self.status.set('正在分析图像…'); path=self.current
  def task():
   try:self.queue.put(('result',self.engine.analyze(path)))
   except Exception as e:self.queue.put(('error',str(e)))
  threading.Thread(target=task,daemon=True).start()
 def display(self,r):
  self.last_result=r; p=r['images']; dimg=r['detection_images']; overlay=dimg['image'].copy(); overlay[dimg['mask']==0]=(overlay[dimg['mask']==0]*.18).astype('uint8'); self.show_images([p['image'],overlay,p['crop']]); self.title.set(r['ranking'][0]['name'])
  lines=['候选物种  '+ '  |  '.join(f"{i+1}. {v['name']}" for i,v in enumerate(r['ranking'][:3])),f"处理方案：{r['method']}     耗时：{r['seconds']:.2f} 秒"]
  d=r['diagnostics']; lines+=['',f"图像特征  前景占比 {d['foreground_ratio']:.1%}   边缘盒维数 {d['edge_box_dimension']:.3f}   左右投影对称性 {d['vertical_mask_symmetry']:.3f}"]
  ref=r['avonet_reference']
  if ref:lines+=['',f"AVONET 物种平均值  {ref['Species2']}",f"体重 {ref['Mass']} g   手翼指数 {ref['Hand-Wing.Index']}   喙长 {ref['Beak.Length_Nares']} mm",f"生境 {ref['Habitat']}   营养生态位 {ref['Trophic.Niche']}   生活型 {ref['Primary.Lifestyle']}"]
  else:lines+=['','该标签的物种对应关系未确认，暂不提供 AVONET 参考值。']
  e=r['image_regression']; lines+=['',f"探索性图像回归  体重 {e['mass_g']:.1f} g   手翼指数 {e['hand_wing_index']:.1f}   喙长 {e['beak_nares_mm']:.1f} mm",'以上是模型估计，可能超出合理范围；不是对照片中个体的实测值。','物种参考值以识别正确为前提。投影对称性受姿态影响，不代表生物学双侧对称。']
  if r['segmentation_fallback']:lines+=['注意：分割触发中心区域回退，请检查分割图。']
  self.set_detail('\n'.join(lines)); self.status.set('分析完成 · 原始结果和批量导出保留各项指标')
 def batch_open(self):
  if self.busy:return
  paths=filedialog.askopenfilenames(filetypes=[('图像','*.jpg *.jpeg *.png *.bmp *.webp')])
  if paths:self.start_batch(paths)
 def start_batch(self,paths):
  self.busy=True; self.batch_rows=[]; self.batchtree.delete(*self.batchtree.get_children()); self.progress.configure(maximum=len(paths),value=0)
  def task():
   for p in paths:
    try:self.queue.put(('batch_row',flat_result(self.engine.analyze(p))))
    except Exception as e:self.queue.put(('batch_row',{'file':str(p),'error':str(e)}))
   self.queue.put(('batch_done',None))
  threading.Thread(target=task,daemon=True).start()
 def export_batch(self):
  if not self.batch_rows:return
  path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='鸟类批量分析.csv',filetypes=[('CSV','*.csv')])
  if path:pd.DataFrame(self.batch_rows).to_csv(path,index=False,encoding='utf-8-sig'); self.status.set(f'结果已保存：{path}')
 def poll(self):
  try:
   while True:
    kind,value=self.queue.get_nowait()
    if kind=='result':self.display(value); self.busy=False; self.runbutton.configure(state='normal')
    elif kind=='error':self.busy=False; self.runbutton.configure(state='normal'); self.status.set('处理失败'); messagebox.showerror('处理失败',value)
    elif kind=='batch_row':
     self.batch_rows.append(value); self.batchtree.insert('',tk.END,values=(Path(value['file']).name,value.get('predicted_name',''),f"{value.get('seconds',0):.2f}",value.get('error','完成'))); self.progress.configure(value=len(self.batch_rows)); self.status.set(f'已处理 {len(self.batch_rows)} 张图像')
    elif kind=='batch_done':self.busy=False; self.status.set('批量处理完成，可导出 CSV')
  except queue.Empty:pass
  self.root.after(100,self.poll)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--qa',action='store_true'); args=ap.parse_args(); root=tk.Tk()
 try: app=App(root)
 except Exception as exc:messagebox.showerror('启动失败',f'{exc}\n请先运行完整实验或检查 config.json。'); root.destroy(); raise
 if args.qa:
  # Deterministic UI smoke test and actual-window screenshots, no fabricated UI.
  from window_capture import capture_widget
  out=BASE/'results/figures'; state={'step':0}; tests=app.index[app.index.is_train==0].iloc[:3]
  app.current=app.data/'CUB_200_2011/images'/tests.iloc[0].relative_path
  def qa():
   step=state['step']
   if step==0:app.analyze(); state['step']=1
   elif step==1 and app.last_result is not None and not app.busy:
    capture_widget(root,out/'ui_analysis.png'); app.tabs.select(app.stats); state['step']=2
   elif step==2:
    capture_widget(root,out/'ui_statistics.png'); app.tabs.select(app.batch); app.start_batch([app.data/'CUB_200_2011/images'/r.relative_path for _,r in tests.iterrows()]); state['step']=3
   elif step==3 and not app.busy:
    capture_widget(root,out/'ui_batch.png'); pd.DataFrame(app.batch_rows).to_csv(BASE/'results/gui_batch_smoke.csv',index=False,encoding='utf-8-sig'); (BASE/'results/gui_qa.json').write_text(json.dumps({'single_image':True,'batch_count':len(app.batch_rows),'batch_errors':sum('error' in r for r in app.batch_rows),'screenshots':3,'capture':'Only application client window using PrintWindow'}),encoding='utf-8'); root.destroy(); return
   root.after(1200,qa)
  root.after(1200,qa)
 root.mainloop()
if __name__=='__main__':main()
