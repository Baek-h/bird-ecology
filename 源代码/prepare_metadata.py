"""Documented name matching; unresolved CUB labels remain missing, never guessed."""
from pathlib import Path
import json,re,hashlib
import pandas as pd
import requests

BASE=Path(__file__).resolve().parent
TAX_URL='https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&version=2021'
# Explicit historical spelling / taxonomic equivalences. Ambiguous genus-only
# labels are deliberately not expanded to a species.
OVERRIDES={
 'Forsters_Tern':('Sterna forsteri',"Forster's Tern possessive spelling"),
 'Cardinal':('Cardinalis cardinalis','Northern Cardinal; abbreviated common label'),
 'Green_Violetear':('Colibri thalassinus','Historical broad Green Violetear taxon'),
 'Florida_Jay':('Aphelocoma coerulescens','Florida Scrub-Jay'),
 'White_breasted_Kingfisher':('Halcyon smyrnensis','White-throated Kingfisher synonym'),
 'Mockingbird':('Mimus polyglottos','Northern Mockingbird; abbreviated common label'),
 'White_Pelican':('Pelecanus erythrorhynchos','American White Pelican; North American CUB label'),
 'Whip_poor_Will':('Antrostomus vociferus','Eastern Whip-poor-will; historical name'),
 'Nelson_Sharp_tailed_Sparrow':('Ammospiza nelsoni',"Nelson's Sparrow; historical name"),
 'Cape_Glossy_Starling':('Lamprotornis nitens','Cape Starling synonym'),
 'Artic_Tern':('Sterna paradisaea','Arctic Tern spelling correction'),
 'Warbling_Vireo':('Vireo gilvus','AVONET historical broad species concept'),
 'Myrtle_Warbler':('Setophaga coronata','Myrtle form mapped to AVONET Yellow-rumped Warbler species average'),
 'Yellow_Warbler':('Setophaga petechia','AVONET historical broad species concept'),
 'House_Wren':('Troglodytes aedon','AVONET historical broad species concept'),
}
AMBIGUOUS={'Frigatebird','Nighthawk','Sayornis','Geococcyx','Tree_Sparrow'}
def norm(s): return re.sub('[^a-z]','',s.lower().replace("'s",'').replace('grey','gray'))

def prepare(root):
 root=Path(root); out=BASE/'metadata'; out.mkdir(exist_ok=True)
 taxpath=root/'ebird_taxonomy.csv'
 snapshot=out/'ebird_taxonomy_snapshot.csv'
 if not taxpath.exists() and snapshot.exists(): taxpath.write_bytes(snapshot.read_bytes())
 if not taxpath.exists():
  r=requests.get(TAX_URL,timeout=90); r.raise_for_status()
  if not r.content.startswith(b'SCIENTIFIC_NAME'): raise ValueError('Invalid taxonomy response')
  taxpath.write_bytes(r.content)
 snapshot.write_bytes(taxpath.read_bytes())
 tax=pd.read_csv(taxpath); tax=tax[tax.CATEGORY=='species']
 common={norm(r.COMMON_NAME):r.SCIENTIFIC_NAME for _,r in tax.iterrows()}
 book=root/'AVONET Supplementary dataset 1.xlsx'
 av=pd.read_excel(book,sheet_name='AVONET2_eBird'); av.to_csv(out/'avonet_ebird.csv',index=False,encoding='utf-8-sig')
 pd.read_excel(book,sheet_name='Metadata').to_csv(out/'avonet_variable_definitions.csv',index=False,encoding='utf-8-sig')
 # Crosswalk from the same workbook bridges older names without fuzzy matching.
 cross=pd.read_excel(book,sheet_name='BirdLife-eBird crosswalk')
 aliases=dict(zip(cross.Species1,cross.Species2))
 birdtree=pd.read_excel(book,sheet_name='BirdLife–BirdTree crosswalk')
 aliases.update({r.Species3:aliases.get(r.Species1,r.Species1) for _,r in birdtree.iterrows()})
 lookup=av.set_index('Species2'); rows=[]
 dirs=sorted(x.name for x in (root/'segmentations').iterdir() if x.is_dir())
 assert len(dirs)==200
 for name in dirs:
  ident,label=name.split('.',1); sci=common.get(norm(label)); note='eBird common-name exact match after punctuation and possessive normalization'
  if label in OVERRIDES: sci,note=OVERRIDES[label]
  if label in AMBIGUOUS: sci=None; note='Ambiguous common/genus label; ecology withheld pending taxonomic verification'
  original=sci
  if sci not in lookup.index: sci=aliases.get(sci,sci)
  row={'class_id':int(ident),'cub_label':label,'common_name':label.replace('_',' '),'scientific_name_requested':original,'match_note':note,'mapped':bool(sci in lookup.index)}
  if sci in lookup.index:
   row['Species2']=sci
   match=lookup.loc[sci]
   if isinstance(match,pd.DataFrame): raise ValueError('Non-unique AVONET species')
   row.update(match.to_dict())
  rows.append(row)
 result=pd.DataFrame(rows); result.to_csv(out/'species_crosswalk.csv',index=False,encoding='utf-8-sig')
 (out/'taxonomy_source.json').write_text(json.dumps({'requested_url':TAX_URL,'warning':'API may return latest taxonomy despite version request; response preserved and hashed; explicit historical aliases use AVONET workbook taxonomy.','sha256':hashlib.sha256(taxpath.read_bytes()).hexdigest(),'mapped':int(result.mapped.sum()),'total':len(result)},indent=2),encoding='utf-8')
 print(result.loc[~result.mapped,['class_id','cub_label','scientific_name_requested']].to_string(index=False),flush=True)
 print('MAPPED',int(result.mapped.sum()),'/',len(result),flush=True)
 return result

if __name__=='__main__': prepare(json.loads((BASE/'config.json').read_text(encoding='utf-8'))['data_root'])
