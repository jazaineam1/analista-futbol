"""MatchLab 4 — exact 1X2 point-order bounds, not a full official goals tie-break certificate.

Each fixture has a single shared categorical outcome. scipy MILP proves lower/upper rank
bounds under strictly-points-ahead (optimistic tied teams) and points-ahead-or-tied
(pessimistic tied teams). The model does NOT claim all official goal tie-breaks.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

@dataclass
class BoundResult:
    team: str
    best: int
    worst: int
    best_route: list[int]
    worst_route: list[int]


def point_rank_bounds(data: dict, team_id: str, fixed: Optional[dict[str,int]]=None, time_limit: float=6.0) -> BoundResult:
    """1-based mathematically certified bounds under points-only tie advantages.

    fixed maps fixture.id to outcome 0=home win, 1=draw, 2=away win.
    """
    teams=data['teams']; fixtures=data['fixtures']; N=len(teams); F=len(fixtures)
    ids={t['id']:i for i,t in enumerate(teams)}
    ti=ids[team_id]
    T=3*F+(N-1)
    values=np.array([t['pts'] for t in teams],dtype=float)
    P=np.zeros((N,T))
    for fi,f in enumerate(fixtures):
        h=ids[f['home']];a=ids[f['away']]
        P[h,3*fi]+=3;P[h,3*fi+1]+=1
        P[a,3*fi+1]+=1;P[a,3*fi+2]+=3
    # Bounds x[fixture,outcome] in [0,1], one chosen per fixture.
    lb=np.zeros(T);ub=np.ones(T)
    for fi,f in enumerate(fixtures):
        if fixed and f['id'] in fixed:
            o=fixed[f['id']]
            if o not in (0,1,2): raise ValueError('Invalid outcome')
            for k in range(3):ub[3*fi+k]=int(k==o)
    rivals=[i for i in range(N) if i != ti]
    M=100 # all point differences abs <= 57, safe
    def solve(delta: int, maximize:bool):
        rows=[];low=[];high=[]
        for fi in range(F):
            r=np.zeros(T);r[3*fi:3*fi+3]=1
            rows.append(r);low.append(1);high.append(1)
        for j,ri in enumerate(rivals):
            # d=(base_i-base_t-delta)+(P_i-P_t). Want d-M*y in [-M,-1]
            r=P[ri]-P[ti];r[3*F+j]=-M
            base=values[ri]-values[ti]-delta
            rows.append(r);low.append(-M-base);high.append(-1-base)
        objective=np.zeros(T); objective[3*F:]=(-1 if maximize else 1)
        A=np.asarray(rows)
        r=milp(c=objective,integrality=np.ones(T),bounds=Bounds(lb,ub),constraints=LinearConstraint(A,low,high),options={'time_limit':time_limit,'mip_rel_gap':0})
        if r.status != 0:
            raise RuntimeError(f'MILP not proven optimal: {r.message}')
        xs=np.asarray(r.x[:3*F]).reshape((F,3))
        route=np.argmax(xs,axis=1).tolist()
        # Recompute rankings under desired bounds to ensure no mistakes.
        points=values.copy()
        for f,o in zip(fixtures,route):
            if o==0:points[ids[f['home']]]+=3
            elif o==1:points[ids[f['home']]]+=1;points[ids[f['away']]]+=1
            else:points[ids[f['away']]]+=3
        ahead=int(sum((points[i]>points[ti] if delta==1 else points[i]>=points[ti]) for i in rivals))
        claimed=1+int(round(sum(r.x[3*F:])))
        assert claimed==1+ahead,(team_id,claimed,ahead)
        return claimed,route
    best,broute=solve(delta=1,maximize=False)
    worst,wroute=solve(delta=0,maximize=True)
    assert best<=worst
    return BoundResult(team_id,best,worst,broute,wroute)


def compute_all(data):
    return {t['id']:vars(point_rank_bounds(data,t['id'])) for t in data['teams']}

if __name__=='__main__':
    import sys,time
    data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    start=time.monotonic()
    for t in data['teams']:
        r=point_rank_bounds(data,t['id']);print(t['id'],r.best,r.worst, f'{time.monotonic()-start:.1f}s',flush=True)