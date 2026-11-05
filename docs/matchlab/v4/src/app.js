'use strict';
const $=id=>document.getElementById(id);
const fmt=x=>new Intl.NumberFormat('es-CO').format(x);
const pct=x=>Number.isFinite(x)?new Intl.NumberFormat('es-CO',{style:'percent',maximumFractionDigits:x<.01?2:1}).format(x):'—';
const h=(str)=>String(str).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const clone=x=>JSON.parse(JSON.stringify(x));
const DEMO={meta:{id:'demo',name:'Liga demostrativa',country:'Sintética',season:'Ejemplo',asOf:'—',timezone:'America/Bogota',topN:2,gamesPerTeam:3,status:'DATOS SINTÉTICOS'},teams:[{id:'A',name:'Aurora',pts:0,pj:0,w:0,d:0,l:0,gf:0,ga:0},{id:'B',name:'Brisa',pts:0,pj:0,w:0,d:0,l:0,gf:0,ga:0},{id:'C',name:'Cóndor',pts:0,pj:0,w:0,d:0,l:0,gf:0,ga:0},{id:'D',name:'Delta',pts:0,pj:0,w:0,d:0,l:0,gf:0,ga:0}],fixtures:[{id:'1-A-B',round:1,home:'A',away:'B'},{id:'1-C-D',round:1,home:'C',away:'D'},{id:'2-A-C',round:2,home:'A',away:'C'},{id:'2-B-D',round:2,home:'B',away:'D'},{id:'3-A-D',round:3,home:'A',away:'D'},{id:'3-B-C',round:3,home:'B',away:'C'}],recentFinals:[],sources:[]};
let data=clone(window.MATCHLAB_INITIAL_DATA),team='DIM',picks=[],probs=[],goalGrids=[],modelProbs=[],sim=null,tableMode='scenario',busy=false,lastRoute=0n;
let teamIndex=new Map();
function validate(d){
 if(!d||!d.meta||!Array.isArray(d.teams)||!Array.isArray(d.fixtures))throw Error('JSON: se esperan meta, teams, fixtures.');
 if(d.teams.length<2||d.teams.length>40||d.fixtures.length>80)throw Error('Límites de prueba: entre 2 y 40 clubes y hasta 80 partidos pendientes.');
 const ids=new Set();let gf=0,ga=0;
 for(const t of d.teams){if(!t||typeof t.id!=='string'||!/^[A-Za-z0-9_-]{1,32}$/.test(t.id)||typeof t.name!=='string'||ids.has(t.id))throw Error('ID de club no válido o repetido');ids.add(t.id);
  for(const k of ['pts','pj','w','d','l','gf','ga'])if(!Number.isInteger(t[k])||t[k]<0)throw Error('Estadística inválida de '+t.id+': '+k);
  if(t.w+t.d+t.l!==t.pj||t.pts!==3*t.w+t.d)throw Error('Puntos o PJ inconsistentes de '+t.id);gf+=t.gf;ga+=t.ga;
 }
 if(gf!==ga)throw Error('Balance de goles inconsistente: '+gf+'/'+ga);
 if(!Number.isInteger(d.meta.topN)||d.meta.topN<1||d.meta.topN>d.teams.length)throw Error('topN inválido');
 let seen=new Set(),counts=new Map([...ids].map(id=>[id,0]));
 for(const f of d.fixtures){if(!ids.has(f.home)||!ids.has(f.away)||f.home===f.away)throw Error('Encuentro con club inexistente');let key=f.home+'|'+f.away;if(seen.has(key))throw Error('Partido duplicado: '+key);seen.add(key);counts.set(f.home,counts.get(f.home)+1);counts.set(f.away,counts.get(f.away)+1)}
 if(d.meta.gamesPerTeam!=null){for(const t of d.teams)if(t.pj+counts.get(t.id)!==d.meta.gamesPerTeam)throw Error('Calendario incompleto para '+t.name)}
 return true;
}
function initialize(d,initialTeam){validate(d);data=clone(d);teamIndex=new Map(data.teams.map((t,i)=>[t.id,i]));team=(initialTeam&&teamIndex.has(initialTeam))?initialTeam:data.teams[0].id;
 picks=Array(data.fixtures.length).fill(null);probs=[];modelProbs=[];goalGrids=[];data.fixtures.forEach(f=>{const m=forecastMatch(f);modelProbs.push(m.p);probs.push(m.p.slice());goalGrids.push(m.grids)});sim=null;tableMode='scenario';lastRoute=0n;
 renderAll(true);
}
function factorialPMF(lambda,max=10){let a=[Math.exp(-lambda)];for(let k=1;k<=max;k++)a.push(a[k-1]*lambda/k);return a}
function forecastMatch(f){const a=data.teams[teamIndex.get(f.home)],b=data.teams[teamIndex.get(f.away)];const allG=data.teams.reduce((s,t)=>s+t.gf,0),allPJ=data.teams.reduce((s,t)=>s+t.pj,0),avg=allPJ?allG/allPJ:1.25;
 const alpha=8,rate=(g,pj)=>(g+alpha*avg)/(pj+alpha);
 let lh=Math.max(.24,Math.min(3.5,Math.sqrt(rate(a.gf,a.pj)*rate(b.ga,b.pj))*1.11));
 let la=Math.max(.24,Math.min(3.5,Math.sqrt(rate(b.gf,b.pj)*rate(a.ga,a.pj))*.92));
 const ph=factorialPMF(lh),pa=factorialPMF(la),grids=[[],[],[]],w=[0,0,0];
 for(let x=0;x<ph.length;x++)for(let y=0;y<pa.length;y++){
  const o=x>y?0:x===y?1:2,wt=ph[x]*pa[y];grids[o].push([x,y,wt]);w[o]+=wt;
 }
 const sum=w.reduce((a,b)=>a+b,0);return {p:w.map(x=>x/sum),grids};
}
function scoreFromOutcome(o){return o===0?[1,0]:o===1?[0,0]:[0,1]}
function own(tid){return data.teams.find(t=>t.id===tid)}
function outcomeOf(p){return p?.o??null}
function standings(selection){const ts=data.teams.map(t=>({...t}));const ix=new Map(ts.map((t,i)=>[t.id,i]));
 for(let k=0;k<data.fixtures.length;k++){const p=selection[k];if(!p)continue;let f=data.fixtures[k],a=ts[ix.get(f.home)],b=ts[ix.get(f.away)];let [hg,ag]=p.score||scoreFromOutcome(p.o);
 a.pj++;b.pj++;a.gf+=hg;a.ga+=ag;b.gf+=ag;b.ga+=hg;
 if(hg>ag){a.w++;a.pts+=3;b.l++}else if(hg===ag){a.d++;b.d++;a.pts++;b.pts++}else{b.w++;b.pts+=3;a.l++}
 }
 const remaining=new Map(data.teams.map(t=>[t.id,data.fixtures.reduce((n,f,i)=>n+(!selection[i]&&(t.id===f.home||t.id===f.away)?1:0),0)]));
 ts.forEach(t=>{t.dg=t.gf-t.ga;t.max=t.pts+3*remaining.get(t.id)});
 ts.sort((a,b)=>b.pts-a.pts||b.dg-a.dg||b.gf-a.gf||a.name.localeCompare(b.name,'es'));
 return ts;
}
function baseRank(){return standings([])}
function hasAnyPicks(){return picks.some(Boolean)}
function baseBounds(tid){return data.bounds?.[tid]||null}
function hardStatus(tid){const t=own(tid),top=data.meta.topN;
 const bound=baseBounds(tid);
 if(bound){if(bound.best>top)return {kind:'eliminado',text:'Eliminado',details:`Incluso con todos los desempates de puntos a su favor, el mejor puesto posible es ${bound.best}º. Ninguna combinación de los encuentros pendientes permite entrar al top ${top}.`};if(bound.worst<=top)return {kind:'clasificado',text:'Clasificado',details:`En el escenario matemático más adverso, incluyendo empates de puntos tratados en contra, no baja del puesto ${bound.worst}º.`};return {kind:'disputa',text:'En disputa',details:`Cotas por puntos: entre ${bound.best}º (empates favorables) y ${bound.worst}º (empates adversos). La clasificación depende de resultados de otros clubes.`}}
 const ownRemaining=data.fixtures.filter(f=>f.home===tid||f.away===tid).length;const max=t.pts+ownRemaining*3;
 const definitelyAhead=data.teams.filter(v=>v.id!==tid&&v.pts>max).length;
 if(definitelyAhead>=top)return {kind:'eliminado',text:'Eliminado',details:`Al menos ${definitelyAhead} equipos tienen más puntos que su máximo de ${max}.`};
 return {kind:'desconocido',text:'Sin certificar',details:'Para certificar matemáticamente mejores y peores puestos de la liga importada, es necesario ejecutar el optimizador del backend.'};
}
function probabilities(i){return probs[i]}
function renderSelectors(){let select=$('team');const previous=select.value;select.innerHTML=data.teams.slice().sort((a,b)=>a.name.localeCompare(b.name,'es')).map(t=>`<option value="${h(t.id)}">${h(t.name)}</option>`).join('');select.value=team;
 $('competition-info').textContent=`${data.meta.name} · ${data.meta.season} · Corte: ${data.meta.asOf||'no informado'} · ${data.meta.status||'datos importados'}.`;
 $('recent-finals').innerHTML=(data.recentFinals||[]).map(f=>`<span>Final: ${h(own(f.home)?.name||f.home)} ${f.homeGoals}–${f.awayGoals} ${h(own(f.away)?.name||f.away)}</span>`).join('')||'<span>Sin resultados recientes en esta muestra.</span>';
 $('update-pill').textContent=data.meta.status==='DATOS SINTÉTICOS'?'⚠ Liga de prueba':'● Datos de corte · no es en vivo';
 $('filter-round').innerHTML='<option value="all">Todas</option>'+[...new Set(data.fixtures.map(f=>f.round))].sort().map(r=>`<option value="${h(r)}">${h(r)}</option>`).join('');
}
function renderFacts(){const t=own(team),current=baseRank().findIndex(x=>x.id===team)+1,b=baseBounds(team),remaining=data.fixtures.filter(f=>f.home===team||f.away===team).length,st=hardStatus(team);
 $('m-current').textContent=current+'º';$('m-points').textContent=t.pts+' / '+(t.pts+remaining*3);
 $('m-points-sub').textContent=remaining+' partidos restantes';
 $('m-best').textContent=b?b.best+'º':'—';$('m-worst').textContent=b?b.worst+'º':'—';$('m-status').textContent=st.text;$('m-status').className=st.kind==='eliminado'?'red':st.kind==='clasificado'?'green':'yellow';$('m-status-sub').textContent='Clasifican '+data.meta.topN;
 $('m-chance').textContent=sim?pct(sim.top[teamIndex.get(team)]/sim.n):'—';$('m-chance-sub').textContent=sim?`Modelo experimental, ${fmt(sim.n)} muestras`:'Sin simulación';
 const picked=picks.filter(Boolean).length;const stBase=standings(picks),selected=stBase.find(v=>v.id===team),rank=stBase.findIndex(v=>v.id===team)+1;
 let comment=st.details;
 if(picked)comment+=` Tienes ${picked} partidos fijados en el editor; en la tabla parcial, ${h(t.name)} ocupa el puesto ${rank}º con ${selected.pts} puntos y puede llegar a ${selected.max}. Esto NO cambia el pronóstico libre.`;
 else comment+=` Tiene ${remaining} partidos restantes y un máximo de ${t.pts+3*remaining} puntos.`;
 $('reason').innerHTML=`<b>${h(st.text)}.</b> ${comment}<p class="hint" style="margin:9px 0 0">La posición exacta en caso de empates en puntos puede requerir goles visitantes y criterios adicionales de Dimayor.</p>`;
 $('reason-state').textContent=b?'Optimización MILP':'Cota básica';
 renderSimForTeam();
}
function renderSimForTeam(){const el=$('chart');if(!sim){el.innerHTML='<p class="hint">Ninguna simulación ejecutada todavía.</p>';$('expected-points').textContent='—';$('observed-range').textContent='—';$('sim-status').textContent='Sin simulación';$('sim-description').textContent='La distribución aparecerá tras ejecutar Monte Carlo. El modo libre ignora las selecciones del editor.';return}
 const j=teamIndex.get(team),dist=sim.dist[j],n=sim.n;el.innerHTML=dist.map((v,i)=>`<div class="chart-row"><span>${i+1}º</span><div class="bar"><div style="width:${(v/n*100).toFixed(4)}%"></div></div><span class="value">${pct(v/n)}</span></div>`).join('');
 $('expected-points').textContent=(sim.pt[j]/n).toFixed(2);$('observed-range').textContent=(dist.findIndex(x=>x>0)+1)+'º–'+(dist.length-dist.slice().reverse().findIndex(x=>x>0))+'º';
 $('sim-status').textContent=(sim.conditional?'Condicionado':'Libre')+' · '+fmt(n)+' escenarios';
 $('sim-description').textContent=`Entre ${fmt(n)} muestras, ${h(own(team).name)} obtuvo top ${data.meta.topN} en ${pct(sim.top[j]/n)} y lideró la fase en ${pct(sim.first[j]/n)}. Semilla ${sim.seed}. No es la probabilidad de ganar el campeonato.`;
}
function renderTable(){const arr=standings(tableMode==='scenario'?picks:[]);const k=picks.filter(Boolean).length;
 $('table-status').textContent=tableMode==='scenario'?(k?`Tabla parcial con ${k} de ${data.fixtures.length} resultados fijados, el resto sin jugar.`:'Tabla vigente, sin partidos hipotéticos.'):'Tabla vigente sin los resultados simulados.';
 $('table-body').innerHTML=arr.map((t,i)=>`<tr class="${i<data.meta.topN?'cut':'notcut'} ${t.id===team?'selected':''}"><td class="num">${i+1}</td><td class="name"><button class="table-row-btn" data-team="${h(t.id)}">${h(t.name)}</button></td><td class="num">${t.pj}</td><td class="num">${t.w}</td><td class="num">${t.d}</td><td class="num">${t.l}</td><td class="num">${t.gf}</td><td class="num">${t.ga}</td><td class="num">${t.dg>0?'+':''}${t.dg}</td><td class="num"><b>${t.pts}</b></td><td class="num">${t.max}</td></tr>`).join('');
}
function renderFixtures(){const onlyTeam=$('only-team').checked,onlyUnfixed=$('only-unfixed').checked,rd=$('filter-round').value;let count=0;
 $('fixtures').innerHTML=data.fixtures.map((f,i)=>{
  if((onlyTeam&&!(f.home===team||f.away===team))||(onlyUnfixed&&picks[i])||(rd!=='all'&&String(f.round)!==rd))return '';
  count++;const p=picks[i],prob=probs[i],fixed=p!==null;
  const [hg,ag]=p?.score??['',''];const home=own(f.home).name,away=own(f.away).name;
  return `<div class="fixture ${fixed?'sel':''}" data-fixture="${i}"><div class="round">Jornada ${h(f.round)}<br>${h(f.date||'Por confirmar')} ${h(f.time||'')}</div><div class="teams">${h(home)}<small>vs. ${h(away)}</small></div><div class="prob"><div class="probs"><span>1 ${pct(prob[0])}</span><span>X ${pct(prob[1])}</span><span>2 ${pct(prob[2])}</span></div><span>${p?.score?'Marcador exacto fijado':fixed?'Resultado 1X2 fijado':'Modelo Poisson · no verificado'}</span></div><div class="actions"><div class="outcomes"><button data-i="${i}" data-o="0" class="${p?.o===0?'active':''}">1</button><button data-i="${i}" data-o="1" class="${p?.o===1?'active':''}">X</button><button data-i="${i}" data-o="2" class="${p?.o===2?'active':''}">2</button><button data-i="${i}" data-o="clear" class="clear">↺</button></div><div class="score-control"><input type="number" min="0" max="15" data-score-i="${i}" data-side="h" value="${hg}" aria-label="Goles de ${h(home)}" placeholder="G"><span>–</span><input type="number" min="0" max="15" data-score-i="${i}" data-side="a" value="${ag}" aria-label="Goles de ${h(away)}" placeholder="G"><span class="hint">exacto</span></div></div><details class="advanced"><summary>Ajustar P(1/X/2) · ${h(f.note||'modelo estimado')}</summary><div class="triplet"><label>Local %<input type="number" min="0" max="100" step="0.1" data-pr="${i}" data-k="0" value="${(prob[0]*100).toFixed(1)}"></label><label>Empate %<input type="number" min="0" max="100" step="0.1" data-pr="${i}" data-k="1" value="${(prob[1]*100).toFixed(1)}"></label><label>Visita %<input type="number" min="0" max="100" step="0.1" data-pr="${i}" data-k="2" value="${(prob[2]*100).toFixed(1)}"></label><button class="btn small" data-save-prob="${i}">Aplicar</button></div></details></div>`;
 }).join('')||'<p class="hint">Ningún encuentro cumple el filtro.</p>';
 $('preview-remaining').textContent=`${count} visibles de ${data.fixtures.length} pendientes.`;
}
function renderUniverse(){const free=picks.filter(p=>!p).length,total=3n**BigInt(data.fixtures.length),comp=3n**BigInt(free);
 $('total-universe').textContent=fmt(total);$('compatible-universe').textContent=fmt(comp);
 $('universe-explanation').textContent=`Fijados: ${data.fixtures.length-free} de ${data.fixtures.length}. Quedan ${free} partidos libres, por tanto 3^${free} calendarios compatibles (sin detallar marcadores exactos).`;
 $('picks-total').textContent=`${data.fixtures.length-free} de ${data.fixtures.length} resultados fijados`;
 renderPoints();
}
function renderPoints(){let t=own(team),m=new Map([[t.pts,1]]),count=new Map([[t.pts,1n]]);let onlyFix=$('condition').checked;
 data.fixtures.forEach((f,i)=>{if(f.home!==team&&f.away!==team)return;const isHome=f.home===team,choice=onlyFix?picks[i]:null;
 const ps=choice?[0,1,2].map(o=>o===choice.o?1:0):probs[i];const next=new Map();
 for(const [points,weight] of m){for(let o=0;o<3;o++){const add=o===1?1:(o===0?isHome:!isHome)?3:0,p=points+add;next.set(p,(next.get(p)||0)+weight*ps[o]);}}
 // Need do not mutate source count while iterating; use separate nextCounts
 const nextCounts=new Map();for(const [points,c] of count)for(let o=0;o<3;o++){if(choice&&o!==choice.o)continue;let add=o===1?1:(o===0?isHome:!isHome)?3:0,p=points+add;nextCounts.set(p,(nextCounts.get(p)||0n)+c)}
 m=next;count=nextCounts;
 });
 const rows=[...m.entries()].sort((a,b)=>a[0]-b[0]);const sum=rows.reduce((s,a)=>s+a[1],0);
 $('points-distribution').innerHTML=rows.map(([pts,prob])=>`<div class="stat-row"><span>${pts} puntos</span><span><b>${pct(prob/sum)}</b> <small class="muted">(${fmt(count.get(pts)||0n)} rutas propias)</small></span></div>`).join('');
 $('points-note').textContent=onlyFix?'Distribución condicionada por las selecciones del editor, para los partidos del equipo.':'Distribución sin condiciones del editor. Solo modela puntos del equipo, no posiciones conjuntas.';
}
function renderAudit(){const n=data.fixtures.length,counts=new Map(data.teams.map(t=>[t.id,0]));data.fixtures.forEach(f=>{counts.set(f.home,counts.get(f.home)+1);counts.set(f.away,counts.get(f.away)+1)});
 const good=data.teams.filter(t=>!data.meta.gamesPerTeam||t.pj+counts.get(t.id)===data.meta.gamesPerTeam).length;
 const gf=data.teams.reduce((s,t)=>s+t.gf,0),ga=data.teams.reduce((s,t)=>s+t.ga,0);
 $('audit-finals').textContent=String(data.recentFinals?.length||0);$('audit-fixtures').textContent=n; $('audit-teams').textContent=good+'/'+data.teams.length;$('audit-goals').textContent=gf+' / '+ga;
 $('data-meta').textContent=`${data.meta.name} ${data.meta.season}. Corte declarado: ${data.meta.asOf}. ${data.meta.sourceNote||'Requiere verificar procedencia de datos importados.'}`;
 $('source-links').innerHTML=(data.sources||[]).map(x=>`<p style="margin:6px 0"><a href="${h(x.url)}" target="_blank" rel="noreferrer">${h(x.label)}</a></p>`).join('')||'<span class="hint">Liga sintética o importada sin fuentes declaradas.</span>';
}
function renderAll(resetFilter=false){if(resetFilter){renderSelectors();$('only-team').checked=window.matchMedia('(max-width:800px)').matches;$('only-unfixed').checked=false;$('condition').checked=false;$('filter-round').value='all'}else{$('team').value=team}
 renderFacts();renderTable();renderFixtures();renderUniverse();renderAudit();}
function choose(i,o){if(o==='clear'){picks[i]=null}else{const a=Number(o);if(![0,1,2].includes(a))return;picks[i]={o:a,score:null}};
 if($('condition').checked)sim=null;renderAll();}
function setScore(i,homeGoals,awayGoals){if(!Number.isInteger(homeGoals)||!Number.isInteger(awayGoals)||homeGoals<0||awayGoals<0||homeGoals>15||awayGoals>15)throw Error('Marcador: ingresa goles entre 0 y 15 en ambos campos.');const o=homeGoals>awayGoals?0:homeGoals===awayGoals?1:2;picks[i]={o,score:[homeGoals,awayGoals]};if($('condition').checked)sim=null;renderAll();}
function setPreset(name){const n=data.fixtures.length,modal=()=>data.fixtures.map((_,i)=>({o:probs[i].indexOf(Math.max(...probs[i])),score:null}));
 if(name==='clear'){picks=Array(n).fill(null)}
 else if(name==='draw'){picks=Array.from({length:n},()=>({o:1,score:null}))}
 else if(name==='modal'){picks=modal()}
 else if(name==='favorable'||name==='adverse'){picks=modal();data.fixtures.forEach((f,i)=>{if(f.home===team||f.away===team)picks[i]={o:name==='favorable'?(f.home===team?0:2):(f.home===team?2:0),score:null}})}
 else if(name==='best'||name==='worst'){
  let r=baseBounds(team);if(!r){$('preset-description').textContent='La liga importada no trae certificados. Se cargó una ruta ilustrativa.';setPreset(name==='best'?'favorable':'adverse');return}
  const route=name==='best'?r.best_route:r.worst_route;picks=route.map(o=>({o,score:null}));
 }
 $('preset-description').textContent=({draw:'Todos los partidos hipotéticos terminan 0–0 en la previsualización; el modelo de goles sigue siendo probabilístico.',modal:'Se eligió el resultado 1X2 individual más probable de cada partido. No es un certificado de la posición final.',favorable:'Todos los partidos del club seleccionado terminan en victoria, los otros siguen resultados modales. Ruta ilustrativa, no necesariamente óptima.',adverse:'El club seleccionado pierde todos sus encuentros, los otros siguen resultados modales. Ruta ilustrativa, no necesariamente peor posición.',best:'Ruta factible calculada por programación lineal para minimizar rivales estrictamente por delante en puntos (empates favorables).',worst:'Ruta factible calculada por programación lineal para maximizar rivales por delante o empatados en puntos (empates adversos).',clear:'No hay partidos condicionados. Vuelve el espacio de escenarios completo.'})[name]||'';
 if($('condition').checked)sim=null;renderAll();}
function encodeRoute(){let v=0n;for(const p of picks){if(!p)return null;v=v*3n+BigInt(p.o)}return v}
function loadRoute(v){const total=3n**BigInt(data.fixtures.length);if(v<0n||v>=total)throw Error('El índice debe estar entre 0 y 3^N−1.');let a=Array(data.fixtures.length).fill(0);for(let i=a.length-1;i>=0;i--){a[i]=Number(v%3n);v/=3n}picks=a.map(o=>({o,score:null}));if($('condition').checked)sim=null;renderAll();$('route-hint').textContent='Ruta reproducible cargada: los '+a.length+' encuentros están definidos en el editor.';}
function uniformRandomIndex(max){const bits=max.toString(2).length,bytes=Math.ceil(bits/8);while(true){const arr=new Uint8Array(bytes);if(globalThis.crypto?.getRandomValues)crypto.getRandomValues(arr);else arr.forEach((_,i)=>{arr[i]=Math.floor(Math.random()*256)});let v=0n;for(const b of arr)v=v*256n+BigInt(b);if(v<max)return v}}
function rng(seed){let x=(seed>>>0)||0x6d2b79f5;return ()=>{x+=0x6d2b79f5;let t=x;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296}}
function sampleScore(i,o,rnd){const cells=goalGrids[i][o],weight=cells.reduce((s,v)=>s+v[2],0);let u=rnd()*weight;for(const c of cells){u-=c[2];if(u<=0)return [c[0],c[1]]}const c=cells.at(-1);return [c[0],c[1]]}
function clearSimulation(){sim=null;renderFacts();$('progress-text').textContent='Parámetros modificados. Ejecuta nuevamente la simulación.';$('progress-bar').style.width='0'}
async function runSimulation(){if(busy)return;const n=Number($('samples').value),conditioned=$('condition').checked;let seedText=$('seed').value.trim(),seed;
 if(seedText){if(!/^\d{1,10}$/.test(seedText)){alert('Semilla: escribe un entero positivo de hasta 10 dígitos.');return}seed=Number(seedText)>>>0}else{seed=Math.floor(Math.random()*4294967296)>>>0}
 let rnd=rng(seed),N=data.teams.length,F=data.fixtures.length,dist=Array.from({length:N},()=>Array(N).fill(0)),top=Array(N).fill(0),first=Array(N).fill(0),pt=Array(N).fill(0),base=data.teams;busy=true;$('run-sim').disabled=true;$('run-sim').textContent='Calculando...';$('progress-bar').style.width='0';
 const idx=teamIndex;
 for(let iteration=0;iteration<n;iteration++){
  const standingsRun=base.map(t=>({id:t.id,name:t.name,pts:t.pts,gf:t.gf,ga:t.ga}));
  for(let fi=0;fi<F;fi++){
   const f=data.fixtures[fi],preset=conditioned?picks[fi]:null,p=probs[fi];
   let o;if(preset)o=preset.o;else{let r=rnd();o=r<p[0]?0:r<p[0]+p[1]?1:2}
   const [hg,ag]=preset?.score||sampleScore(fi,o,rnd);
   let a=standingsRun[idx.get(f.home)],b=standingsRun[idx.get(f.away)];a.gf+=hg;a.ga+=ag;b.gf+=ag;b.ga+=hg;
   if(hg>ag)a.pts+=3;else if(hg<ag)b.pts+=3;else{a.pts++;b.pts++}
  }
  standingsRun.sort((a,b)=>b.pts-a.pts||(b.gf-b.ga)-(a.gf-a.ga)||b.gf-a.gf||a.name.localeCompare(b.name,'es'));
  for(let pos=0;pos<N;pos++){let t=standingsRun[pos],j=idx.get(t.id);dist[j][pos]++;pt[j]+=t.pts;if(pos<data.meta.topN)top[j]++;if(pos===0)first[j]++}
  if((iteration+1)%250===0){$('progress-text').textContent=`Procesadas ${fmt(iteration+1)} de ${fmt(n)} temporadas (${conditioned?'condicionadas':'libres'}).`;$('progress-bar').style.width=((iteration+1)/n*100).toFixed(1)+'%';await new Promise(r=>setTimeout(r,0))}
 }
 sim={n,seed,conditional:conditioned,dist,top,first,pt};busy=false;$('run-sim').disabled=false;$('run-sim').textContent='▶ Repetir simulación';$('progress-bar').style.width='100%';
 $('progress-text').textContent=`${fmt(n)} temporadas conjuntas. ${conditioned?'Se respetaron solo tus resultados fijados.':'No se impusieron resultados del editor.'} Semilla ${seed}.`;
 renderFacts();
}
function normalizeManual(i,values){if(values.some(v=>!Number.isFinite(v)||v<0||v>100))throw Error('Probabilidades deben estar entre 0 y 100.');const total=values.reduce((s,v)=>s+v,0);if(total<1)throw Error('La suma de probabilidades debe ser positiva.');probs[i]=values.map(v=>v/total);clearSimulation();renderFixtures();renderUniverse()}
function saveExport(){const payload={meta:data.meta,selectedTeam:team,model:'poisson-experimental',fixtures:data.fixtures.map((f,i)=>({...f,selectedResult:picks[i],probabilities:probs[i]})),exportedAt:new Date().toISOString()};let blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});let url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='matchlab-escenario-'+data.meta.season+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1500)}
function switchLeague(v){if(v==='demo')initialize(DEMO,'A');else initialize(window.MATCHLAB_INITIAL_DATA,'DIM')}
function attachEvents(){
 $('league').addEventListener('change',e=>switchLeague(e.target.value));
 $('team').addEventListener('change',e=>{team=e.target.value;renderAll();});
 $('condition').addEventListener('change',()=>{clearSimulation();renderUniverse()});
 $('run-sim').addEventListener('click',runSimulation);
 $('table-base').addEventListener('click',()=>{tableMode='base';renderTable()});$('table-scenario').addEventListener('click',()=>{tableMode='scenario';renderTable()});
 $('only-team').addEventListener('change',renderFixtures);$('only-unfixed').addEventListener('change',renderFixtures);$('filter-round').addEventListener('change',renderFixtures);
 $('clear-picks').addEventListener('click',()=>setPreset('clear'));
 document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>setPreset(b.dataset.preset)));
 $('table-body').addEventListener('click',e=>{const b=e.target.closest('[data-team]');if(b){team=b.dataset.team;renderAll();document.querySelector('#panorama').scrollIntoView({behavior:'smooth'})}});
 $('fixtures').addEventListener('click',e=>{
  const b=e.target.closest('button');if(!b)return;
  if(b.dataset.o!=null){choose(Number(b.dataset.i),b.dataset.o);return}
  if(b.dataset.saveProb!=null){const i=Number(b.dataset.saveProb),element=b.closest('.fixture'),arr=[0,1,2].map(k=>Number(element.querySelector(`[data-pr="${i}"][data-k="${k}"]`).value));try{normalizeManual(i,arr)}catch(er){alert(er.message)}return}
 });
 $('fixtures').addEventListener('change',e=>{if(e.target.dataset.scoreI!=null){const i=Number(e.target.dataset.scoreI),card=e.target.closest('.fixture'),h=card.querySelector(`[data-score-i="${i}"][data-side="h"]`).value,a=card.querySelector(`[data-score-i="${i}"][data-side="a"]`).value;
   if(h===''&&a===''){return}if(h===''||a===''){return}try{setScore(i,Number(h),Number(a))}catch(er){alert(er.message)}
 }});
 $('load-route').addEventListener('click',()=>{try{const s=$('route-index').value.trim();if(!/^\d+$/.test(s))throw Error('Identificador no válido');loadRoute(BigInt(s))}catch(e){$('route-hint').textContent=e.message}});
 $('prev-route').addEventListener('click',()=>{try{let s=BigInt($('route-index').value||'0');s=s>0n?s-1n:0n;$('route-index').value=s.toString();loadRoute(s)}catch(e){$('route-hint').textContent=e.message}});
 $('next-route').addEventListener('click',()=>{try{let s=BigInt($('route-index').value||'0');s=s+1n;let top=3n**BigInt(data.fixtures.length)-1n;if(s>top)s=top;$('route-index').value=s.toString();loadRoute(s)}catch(e){$('route-hint').textContent=e.message}});
 $('random-route').addEventListener('click',()=>{let idx=uniformRandomIndex(3n**BigInt(data.fixtures.length));$('route-index').value=idx.toString();loadRoute(idx)});
 $('reset-probs').addEventListener('click',()=>{probs=modelProbs.map(x=>x.slice());clearSimulation();renderFixtures();renderUniverse()});
 $('demo-button').addEventListener('click',()=>{$('league').value='demo';switchLeague('demo')});
 $('certify-fixed').addEventListener('click',async ()=>{
  const target=$('certify-status');target.textContent='Consultando solucionador...';
  try{
   const fixed={};picks.forEach((p,i)=>{if(p)fixed[data.fixtures[i].id]=p.o});
   const r=await fetch('/api/bounds',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({team_id:team,fixed})});
   if(!r.ok)throw Error('El backend no está conectado o rechazó la solicitud ('+r.status+').');
   const x=await r.json();target.textContent=`Con los resultados fijados: puesto óptimo por puntos ${x.best}º a ${x.worst}º; ${Object.keys(fixed).length} encuentros condicionados. Desempates oficiales adicionales pendientes.`;
  }catch(e){target.textContent='No se pudo certificar en línea: '+e.message+' La versión HTML autónoma solo muestra cotas iniciales.'}
 });
 $('export').addEventListener('click',saveExport);
 $('import').addEventListener('change',async e=>{const file=e.target.files?.[0];if(!file)return;try{if(file.size>1e6)throw Error('El archivo supera 1 MB');const json=JSON.parse(await file.text());validate(json);initialize(json,json.teams[0].id);$('league').innerHTML+='<option value="imported">Liga importada · '+h(json.meta.name||'Sin nombre')+'</option>';$('league').value='imported';$('competition-info').textContent+=' · Importación local sin verificación de fuente.';}catch(err){alert('No se pudo importar: '+err.message)}finally{e.target.value=''}});
}
attachEvents();initialize(window.MATCHLAB_INITIAL_DATA,'DIM');
if(window.MATCHLAB_API_AUTO===true){fetch('/api/state',{cache:'no-store'}).then(r=>r.ok?r.json():Promise.reject('Sin estado')).then(d=>initialize(d,'DIM')).catch(()=>{$('competition-info').textContent+=' · No se pudo cargar el estado del servidor.'})}
window.MatchLabTest={getData:()=>data,getPicks:()=>picks,select:(id)=>{team=id;renderAll()},preset:setPreset,choose,table:standings,probabilities:()=>probs,run:runSimulation,sim:()=>sim,setScore,scoreFromOutcome,encodeRoute,loadRoute,validate};