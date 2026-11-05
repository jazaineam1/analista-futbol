"""MatchLab 4 experimental single-instance API.
Run: MATCHLAB_ADMIN_TOKEN=<secret> MATCHLAB_DATA_PATH=/persistent/snapshot.json \
     uvicorn server:app --host 127.0.0.1 --port 8765
Production needs a licensed feed, HTTPS reverse proxy, auth and monitoring.
"""
from __future__ import annotations
import copy
import hmac
import json
import os
import threading
import time
from collections import defaultdict,deque
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from src.solver import point_rank_bounds

ROOT=Path(__file__).resolve().parent
DATA_PATH=Path(os.environ.get('MATCHLAB_DATA_PATH',str(ROOT/'data'/'colombia-2026-ii.json')))
ADMIN_TOKEN=os.environ.get('MATCHLAB_ADMIN_TOKEN','')
app=FastAPI(title='MatchLab 4.0 API',version='4.0.0',docs_url=None,redoc_url=None)
lock=threading.RLock()
hits=defaultdict(deque)

def load_data():
    with lock:
        return json.loads(DATA_PATH.read_text(encoding='utf-8'))

def validate_snapshot(d):
    if not isinstance(d,dict) or not isinstance(d.get('teams'),list) or not isinstance(d.get('fixtures'),list) or not isinstance(d.get('meta'),dict):raise ValueError('Invalid schema')
    teams=d['teams'];fixtures=d['fixtures'];ids=[t['id'] for t in teams];n=len(teams)
    if n<2 or n>40 or len(set(ids))!=n or len(fixtures)>80:raise ValueError('Invalid teams/fixtures')
    for t in teams:
        for x in ['pts','pj','w','d','l','gf','ga']:
            if not isinstance(t.get(x),int) or t[x]<0:raise ValueError('Invalid statistics')
        if t['pj']!=t['w']+t['d']+t['l'] or t['pts']!=3*t['w']+t['d']:raise ValueError('Pts/PJ mismatch')
    if sum(x['gf'] for x in teams)!=sum(x['ga'] for x in teams):raise ValueError('Goal imbalance')
    seen=set();counts={id:0 for id in ids}
    for f in fixtures:
        if f.get('home') not in counts or f.get('away') not in counts or f['home']==f['away'] or f.get('id') in seen:raise ValueError('Invalid or duplicated fixtures')
        seen.add(f['id']);counts[f['home']]+=1;counts[f['away']]+=1
    games=d['meta'].get('gamesPerTeam')
    if games:
        for t in teams:
            if t['pj']+counts[t['id']]!=games:raise ValueError(f'Schedule inconsistent: {t["id"]}')
    return True

class BoundRequest(BaseModel):
    team_id:str=Field(min_length=1,max_length=32)
    fixed:dict[str,int]=Field(default_factory=dict,max_length=80)

class FinalResult(BaseModel):
    fixture_id:str=Field(min_length=1,max_length=64)
    home_goals:int=Field(ge=0,le=15)
    away_goals:int=Field(ge=0,le=15)
    source_url:str=Field(min_length=12,max_length=1000)
    verified_at:str=Field(min_length=10,max_length=64)

@app.middleware('http')
async def default_headers(request:Request,call_next):
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer'
    response.headers['X-Frame-Options']='DENY'
    if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
    return response

@app.get('/',response_class=HTMLResponse)
def home():
    html=(ROOT/'MatchLab_4_2026-11-05.html').read_text(encoding='utf-8')
    return HTMLResponse(html.replace('window.MATCHLAB_API_AUTO=false;','window.MATCHLAB_API_AUTO=true;',1))

@app.get('/api/health')
def health():
    d=load_data();validate_snapshot(d)
    return {'ok':True,'version':'4.0.0','competition':d['meta']['id'],'as_of':d['meta']['asOf'],'fixtures_pending':len(d['fixtures']),'teams':len(d['teams']),'mode':'snapshot_no_live_feed'}

@app.get('/api/state')
def state():
    d=load_data();validate_snapshot(d);return d

@app.post('/api/bounds')
def bounds(payload:BoundRequest,request:Request):
    ip=request.client.host if request.client else 'unknown';now=time.monotonic()
    with lock:
        recent=hits[ip]
        while recent and now-recent[0]>60:recent.popleft()
        if len(recent)>=30:raise HTTPException(429,'Rate limit 30/min')
        recent.append(now)
    d=load_data();validate_snapshot(d)
    if payload.team_id not in {t['id'] for t in d['teams']}:raise HTTPException(422,'Unknown team')
    fixtures={f['id'] for f in d['fixtures']}
    if any(k not in fixtures or v not in(0,1,2) for k,v in payload.fixed.items()):raise HTTPException(422,'Unknown fixture or invalid outcome')
    if len(d['fixtures'])>35:raise HTTPException(422,'Solver limit: 35 fixtures maximum')
    try:
        r=point_rank_bounds(d,payload.team_id,payload.fixed,time_limit=4.0)
    except RuntimeError:
        raise HTTPException(503,'Solver could not prove optimality within time limit')
    return {'team_id':r.team,'best':r.best,'worst':r.worst,'best_route':r.best_route,'worst_route':r.worst_route,'certificate':'exact_point_order_optimistic_and_pessimistic_ties','official_tiebreaks_complete':False}

@app.post('/api/final')
def finalize(payload:FinalResult,x_matchlab_admin:Optional[str]=Header(default=None)):
    if len(ADMIN_TOKEN)<24 or not x_matchlab_admin or not hmac.compare_digest(x_matchlab_admin,ADMIN_TOKEN):raise HTTPException(401,'Admin credentials required')
    if not DATA_PATH.exists():raise HTTPException(500,'Data path missing')
    if not payload.source_url.startswith('https://'):raise HTTPException(422,'Source must be HTTPS')
    with lock:
        d=load_data();validate_snapshot(d)
        idx=next((i for i,f in enumerate(d['fixtures']) if f['id']==payload.fixture_id),None)
        if idx is None:raise HTTPException(409,'Fixture not pending, or already ingested')
        f=d['fixtures'].pop(idx);a=next(t for t in d['teams'] if t['id']==f['home']);b=next(t for t in d['teams'] if t['id']==f['away'])
        hg=payload.home_goals;ag=payload.away_goals
        for t,g,against in [(a,hg,ag),(b,ag,hg)]:t['pj']+=1;t['gf']+=g;t['ga']+=against
        if hg>ag:a['w']+=1;a['pts']+=3;b['l']+=1
        elif hg<ag:b['w']+=1;b['pts']+=3;a['l']+=1
        else:a['d']+=1;b['d']+=1;a['pts']+=1;b['pts']+=1
        d['recentFinals']=[{'date':payload.verified_at[:10],'home':f['home'],'away':f['away'],'homeGoals':hg,'awayGoals':ag,'status':'final','source':payload.source_url}]+d.get('recentFinals',[])
        d['meta']['asOf']=payload.verified_at;d['meta']['status']='instantanea_actualizada_por_administrador_no_feed_en_vivo';d['meta']['sourceNote']='Actualización manual verificada; revisar fecha y proveedor.'
        d.pop('bounds',None) # Old proofs invalid; must recompute before exposing.
        validate_snapshot(d)
        if DATA_PATH.resolve()==(ROOT/'data'/'colombia-2026-ii.json').resolve():raise HTTPException(403,'Use explicit persistent MATCHLAB_DATA_PATH to modify results')
        tmp=DATA_PATH.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,DATA_PATH)
    return {'ok':True,'fixture':payload.fixture_id,'remaining':len(d['fixtures']),'as_of':d['meta']['asOf'],'bounds':'cleared_pending_recomputation'}