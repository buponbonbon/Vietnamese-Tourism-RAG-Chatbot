import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]

class FlaskAbort(Exception):
    def __init__(self, status, description=''):
        self.status, self.description = status, description

class DummyFlask:
    def __init__(self, name):
        self.routes={}
        self.config={}
    def route(self, name):
        def decorator(fn):
            self.routes[name]=fn
            return fn
        return decorator

fake_request=types.SimpleNamespace(args=None)
flask=types.ModuleType('flask')
flask.Flask=DummyFlask
flask.abort=lambda status, description='': (_ for _ in ()).throw(FlaskAbort(status,description))
flask.request=fake_request
flask.render_template=Mock(side_effect=lambda name, **kwargs: {'template':name,**kwargs})
sys.modules['flask']=flask
spec=importlib.util.spec_from_file_location('tourism_backend_for_test',ROOT/'app/local_backend.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Args:
 def __init__(self, query='', page=1): self.query=query;self.page=page
 def get(self, key, default=None, type=None):
  raw = {'q':self.query, 'page':self.page}.get(key,default)
  if type:
   try: return type(raw)
   except: return default
  return raw

# Escape raw HTML while preserving intended highlighting.
snippet = '<script>alert(1)</script> & Hà Nội là thủ đô'
res=mod.highlight_keywords(snippet, 'Hà Nội')
assert '<script>' not in str(res)
assert '&lt;script&gt;' in str(res)
assert '<mark class="keyword-highlight">Hà</mark>' in str(res) or '<mark class="keyword-highlight">Hà Nội</mark>' in str(res)
assert ' &amp; ' in str(res)
print('[PASS] safe HTML highlighted keyword handling')

# Escaped HTML in Jinja without |safe and fake external source URL.
env=Environment(loader=FileSystemLoader(ROOT/'app/templates'),autoescape=select_autoescape(['html']))
safe_html=env.get_template('results.html').render(query='a <svg/onload=alert(1)>',chunks=[{'chunk_id':'aa<bb>','highlighted_text':res,'rerank_score':0.0,'rrf_score':2.0}],total_results=1, total_pages=1,current_page=1,llm_answer='noted')
assert '<script>' not in safe_html and '&lt;script&gt;' in safe_html
assert 'https://vietnamtourism.gov.vn/' not in safe_html
assert '0.0000' in safe_html  # rerank 0 takes precedence over RRF=2
assert '<mark class="keyword-highlight">Hà</mark>' in safe_html and '<mark class="keyword-highlight">Nội</mark>' in safe_html
assert 'aa&lt;bb&gt;' in safe_html
print('[PASS] template XSS and score selection')

# Health should not expose internal path.
mod.rag_pipeline=None
status,status_code=mod.health()
assert status_code==503 and 'project_root' not in status and 'python' not in status
print('[PASS] health offline status and information exposure')

# Test cached RAG output on page navigation, non-mutating across requests.
class FakePipeline:
 def __init__(self): self.calls=0
 def execute_pipeline(self,*args,**kwargs):
  self.calls+=1
  return {'answer':'hello', 'retrieved_context':[{'chunk_id':str(i),'text':'Hà Nội <script>x</script>','rerank_score':i/100} for i in range(6)]}
mod.rag_pipeline=FakePipeline()
mod.execute_query_cached.cache_clear()
fake_request.args=Args('Hà Nội',1)
page1=mod.search()
fake_request.args=Args('Hà Nội',2)
page2=mod.search()
assert mod.rag_pipeline.calls==1
assert len(page1['chunks'])==len(page2['chunks'])==3
assert page1['chunks'][0]['chunk_id']=='0' and page2['chunks'][0]['chunk_id']=='3'
assert 'highlighted_text' not in mod.execute_query_cached('Hà Nội')['retrieved_context'][0]
assert page1['total_pages']==2 and page2['total_pages']==2
print('[PASS] pagination uses cached generation, avoids changing model results')

fake_request.args=Args('hi',0)
try: mod.search(); raise AssertionError('invalid page accepted')
except FlaskAbort as e: assert e.status==400
fake_request.args=Args('X'*1001,1)
try: mod.search(); raise AssertionError('long query accepted')
except FlaskAbort as e: assert e.status==400
print('[PASS] bad input validation')

# Confirms launch code offers independent, locally-bound ports.
text=(ROOT/'app/local_backend.py').read_text(encoding='utf8')
assert '"5000"' in text and 'host="127.0.0.1"' in text
conf=(ROOT/'.streamlit/config.toml').read_text(encoding='utf8')
assert 'enableXsrfProtection = true' in conf and 'enableCORS = true' in conf
print('[PASS] Flask/Streamlit port and security configuration')
