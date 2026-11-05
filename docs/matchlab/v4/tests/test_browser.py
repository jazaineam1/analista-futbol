from pathlib import Path
from playwright.sync_api import sync_playwright
P=Path(__file__).parents[1]/'MatchLab_4_2026-11-05.html'
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox'])
 for size in [(1400,900),(390,844)]:
  page=browser.new_page(viewport={'width':size[0],'height':size[1]});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.set_content(P.read_text(encoding='utf-8'),wait_until='load')
  page.screenshot(path=str(P.parents[0]/('QA_desktop.png' if size[0]>500 else 'QA_mobile.png')), full_page=True)
  print('viewport',size,'errors',errors,'fixtures',page.locator('.fixture').count(),'teams',page.locator('#team option').count(), 'universe',page.locator('#total-universe').inner_text())
  assert not errors,errors
  assert page.locator('#team option').count()==20
  assert page.locator('.fixture').count()==(20 if size[0]>500 else 2)
  assert '3.486.784.401' in page.locator('#total-universe').inner_text()
  assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), (size,page.evaluate('document.documentElement.scrollWidth'))
  # Current final score, not provisional
  assert 'Final:' in page.locator('#recent-finals').inner_text()
  page.locator('#team').select_option('DIM')
  page.locator('[data-preset="worst"]').click()
  page.locator('#samples').select_option('2000')
  page.locator('#run-sim').click()
  page.wait_for_function('window.MatchLabTest.sim() !== null',timeout=60000)
  dist=page.evaluate("window.MatchLabTest.sim().dist[window.MatchLabTest.getData().teams.findIndex(t=>t.id==='DIM')]")
  print('free for DIM',[(i+1,v) for i,v in enumerate(dist) if v], 'sum',sum(dist))
  assert sum(1 for v in dist if v)>1,dist
  assert sum(dist)==2000
  page.locator('#condition').check()
  page.locator('#run-sim').click()
  page.wait_for_function('window.MatchLabTest.sim() !== null',timeout=60000)
  dist2=page.evaluate("window.MatchLabTest.sim().dist[window.MatchLabTest.getData().teams.findIndex(t=>t.id==='DIM')]")
  print('conditioned DIM',[(i+1,v) for i,v in enumerate(dist2) if v])
  assert sum(dist2)==2000
  # Fix exact scores: deterministic rankings should be truly 100% in one position.
  page.evaluate("window.MatchLabTest.getData().fixtures.forEach((f,i)=>window.MatchLabTest.setScore(i,1,0))")
  page.locator('#run-sim').click()
  page.wait_for_function('window.MatchLabTest.sim() !== null',timeout=60000)
  dist_exact=page.evaluate("window.MatchLabTest.sim().dist[window.MatchLabTest.getData().teams.findIndex(t=>t.id==='DIM')]")
  assert len([v for v in dist_exact if v])==1
  page.screenshot(path=str(P.parents[0]/('QA_simulation_desktop.png' if size[0]>500 else 'QA_simulation_mobile.png')),full_page=True)
  # Ensure fixtures share points
  page.locator('#clear-picks').click()
  p0=page.evaluate("window.MatchLabTest.table([]).find(t=>t.id==='CHI').pts")
  page.locator('#only-team').uncheck()
  page.locator('button[data-i="0"][data-o="0"]').click()
  p1=page.evaluate("window.MatchLabTest.table(window.MatchLabTest.getPicks()).find(t=>t.id==='CHI').pts")
  assert p1==p0+3,(p0,p1)
  # BigInt last scenario
  page.locator('#route-index').fill('3486784400')
  page.locator('#load-route').click()
  assert '0' in page.locator('#compatible-universe').inner_text() or page.evaluate('window.MatchLabTest.getPicks().filter(Boolean).length')==20
  assert page.evaluate('window.MatchLabTest.getPicks().every(p => p.o===2)')
  # demo 4 teams
  page.locator('#league').select_option('demo')
  assert page.locator('#team option').count()==4
  assert page.locator('#total-universe').inner_text()=='729'
  assert not errors,errors
  page.close()
 browser.close()
print('PASS all browser checks')