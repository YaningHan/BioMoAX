from pathlib import Path
import sys,os,runpy,json
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'validation'/('access_'+Path(sys.argv[1]).stem+'.json')
events=[]
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve()
 mode=args[1];flags=args[2]
 writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
 allowed_software=any(p.is_relative_to(Path(v).resolve()) for v in [sys.prefix,sys.base_prefix])
 allowed_font=str(p) in {str(Path(os.environ[k]).resolve()) for k in ['BIOMOAX_FONT_REGULAR','BIOMOAX_FONT_BOLD'] if os.environ.get(k)}
 if p.is_relative_to(ROOT.parent) and not p.is_relative_to(ROOT) and not allowed_software and not allowed_font:
  raise RuntimeError('External project dependency forbidden: '+str(p))
 if p.is_relative_to(ROOT):
  rel=p.relative_to(ROOT).as_posix()
  if writing and rel.startswith(('inputs/','assets/','reference/')):raise RuntimeError('Frozen input write forbidden: '+rel)
  events.append(dict(path=rel,write=writing))
sys.addaudithook(audit)
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'scripts'))
import pymupdf as fitz
native_pdf_open=fitz.open
def checked_pdf_open(filename=None,*args,**kwargs):
 if isinstance(filename,(str,os.PathLike)):audit('open',(filename,'r',0))
 return native_pdf_open(filename,*args,**kwargs)
fitz.open=checked_pdf_open
script=sys.argv[1];sys.argv=sys.argv[1:]
try:runpy.run_path(str(ROOT/'scripts'/script),run_name='__main__')
finally:LOG.write_text(json.dumps(events,indent=2))
