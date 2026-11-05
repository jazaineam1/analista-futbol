import itertools
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from src.solver import point_rank_bounds
import server

DATA=json.loads((ROOT/'data'/'colombia-2026-ii.json').read_text())
server.validate_snapshot(DATA)
client=TestClient(server.app)

def test_health():
    r=client.get('/api/health'); assert r.status_code==200; d=r.json();assert d['fixtures_pending']==20 and d['teams']==20 and d['mode']=='snapshot_no_live_feed'
    r=client.get('/api/state');assert r.status_code==200; assert r.json()['meta']['asOf'].startswith('2026-11-05')

def test_frontend():
    r=client.get('/');assert r.status_code==200;assert 'MATCHLAB_API_AUTO=true' in r.text

def test_every_team_bounds_and_witnesses():
    fixture=DATA['fixtures']
    for t in DATA['teams']:
        b=point_rank_bounds(DATA,t['id'])
        proof=DATA['bounds'][t['id']]
        assert (b.best,b.worst)==(proof['best'],proof['worst'])
        for route in [b.best_route,b.worst_route]:
            assert len(route)==len(fixture)
            assert all(x in (0,1,2) for x in route)
    assert DATA['bounds']['JUN']['best']>DATA['meta']['topN']

def test_conditioned_proof():
    f=DATA['fixtures'][0];r=client.post('/api/bounds',json={'team_id':'DIM','fixed':{f['id']:0}})
    assert r.status_code==200,r.text
    assert r.json()['best']<=r.json()['worst']
    assert r.json()['best_route'][0]==0
    assert r.json()['official_tiebreaks_complete'] is False

def test_final_protected():
    f=DATA['fixtures'][0]
    payload={'fixture_id':f['id'],'home_goals':1,'away_goals':0,'source_url':'https://example.org/final','verified_at':'2026-11-06T00:00:00-05:00'}
    assert client.post('/api/final',json=payload).status_code==401

def test_mini_league_exact_exhaustion():
    # 4-club 6-match mini league with 729 exhaustive 1X2 sequences.
    ids=['A','B','C','D'];teams=[dict(id=i,name=i,pts=0,pj=0,w=0,d=0,l=0,gf=0,ga=0) for i in ids]
    pairs=list(itertools.combinations(ids,2));fixtures=[{'id':f'f{i}', 'home':a,'away':b} for i,(a,b) in enumerate(pairs)]
    d={'teams':teams,'fixtures':fixtures,'meta':{'topN':2,'gamesPerTeam':3}}
    bound=point_rank_bounds(d,'A');ranks_strict=set();ranks_tie=set()
    for path in itertools.product(range(3), repeat=6):
        pts={i:0 for i in ids}
        for f,o in zip(fixtures,path):
            if o==0:pts[f['home']]+=3
            elif o==1:pts[f['home']]+=1;pts[f['away']]+=1
            else:pts[f['away']]+=3
        ranks_strict.add(1+sum(pts[i]>pts['A'] for i in ids if i!='A'))
        ranks_tie.add(1+sum(pts[i]>=pts['A'] for i in ids if i!='A'))
    assert bound.best==min(ranks_strict)
    assert bound.worst==max(ranks_tie)


def test_ingest_final_in_isolated_copy():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        p=Path(tmpdir)/'snapshot.json'
        p.write_text(json.dumps(DATA),encoding='utf-8')
        old_path,old_token=server.DATA_PATH,server.ADMIN_TOKEN
        try:
            server.DATA_PATH=p;server.ADMIN_TOKEN='a'*40
            f=DATA['fixtures'][0]
            body={'fixture_id':f['id'],'home_goals':2,'away_goals':1,'source_url':'https://example.org/verified','verified_at':'2026-11-06T21:00:00-05:00'}
            response=client.post('/api/final',json=body,headers={'X-Matchlab-Admin':'a'*40})
            assert response.status_code==200,response.text
            d=json.loads(p.read_text(encoding='utf-8'))
            assert len(d['fixtures'])==19
            assert 'bounds' not in d # stale rank proofs are cleared
            assert len(d['recentFinals'])==3
            assert len([t for t in d['teams'] if t['pj']==18])==2
            assert client.post('/api/final',json=body,headers={'X-Matchlab-Admin':'a'*40}).status_code==409
        finally:
            server.DATA_PATH=old_path;server.ADMIN_TOKEN=old_token

if __name__=='__main__':
  test_health();test_frontend();test_every_team_bounds_and_witnesses();test_conditioned_proof();test_final_protected();test_mini_league_exact_exhaustion();test_ingest_final_in_isolated_copy();print('7/7 backend tests PASS')