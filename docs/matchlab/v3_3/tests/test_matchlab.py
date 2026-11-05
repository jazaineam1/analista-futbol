"""Chromium checks for MatchLab 3.3. Run: python tests/test_matchlab.py"""
import asyncio
import shutil
from pathlib import Path
from playwright.async_api import async_playwright
ROOT = Path(__file__).resolve().parents[1]

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=shutil.which("chromium") or None)
        page = await browser.new_page(viewport={'width':1280, 'height':900})
        errors=[]
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.set_content((ROOT/'MatchLab_3_3_Simulacion_Corregida.html').read_text(),wait_until='load')
        await page.locator('#team').select_option('DIM')
        await page.locator('[data-scenario="worst"]').click()
        await page.wait_for_function('!forecastInFlight && !!distribution && distribution.teamId==="DIM"', timeout=30000)
        a=await page.evaluate('({ranked:distribution.dist.filter(x=>x>0).length,locked:picked.filter(Boolean).length,n:data.fixtures.length,mode:document.getElementById("simBasis").value})')
        assert a['mode']=='free' and a['ranked']>1 and a['locked']==a['n'],a
        assert 'Derrota 0–1' in await page.locator('#teamFixtureSummary').inner_text()
        await page.locator('#simBasis').select_option('conditional')
        await page.wait_for_function('!forecastInFlight && !!distribution && distribution.basis==="conditional"', timeout=30000)
        b=await page.evaluate('({ranked:distribution.dist.filter(x=>x>0).length,text:document.getElementById("simState").textContent})')
        assert b['ranked']==1 and 'NO es una predicción' in b['text'],b
        await page.locator('#simBasis').select_option('free')
        await page.locator('#simSeed').fill('20261104')
        await page.locator('#samples').select_option('10000')
        await page.locator('#btnSim').click()
        await page.wait_for_function('!forecastInFlight && distribution?.n===10000 && distribution?.seed===20261104',timeout=30000)
        c=await page.evaluate('({counts:distribution.dist,seed:distribution.seed,sum:forecastCache.matrix.reduce((s,arr)=>s+arr.slice(0,data.meta.topN).reduce((a,b)=>a+b,0),0),top:data.meta.topN,n:forecastCache.n})')
        assert c['sum']==c['top']*c['n'],c
        await page.locator('#btnSim').click()
        await page.wait_for_function('!forecastInFlight && distribution?.n===10000 && distribution?.seed===20261104',timeout=30000)
        assert c['counts']==await page.evaluate('distribution.dist')
        await page.locator('#team').select_option('AME')
        assert await page.evaluate('distribution.seed')==20261104
        await page.locator('#simSeed').fill('20261105')
        await page.locator('#btnSim').click()
        await page.wait_for_function('!forecastInFlight && distribution?.seed===20261105',timeout=30000)
        await page.locator('#fileImport').set_input_files(str(ROOT/'MatchLab_demo_38_partidos_sinteticos.json'))
        await page.wait_for_function('data.fixtures.length===38',timeout=30000)
        assert await page.locator('#uTotal').inner_text()=='1.350.851.717.672.992.089'
        await page.locator('#uIndex').fill(str(3**38-1))
        await page.locator('#uJump').click()
        assert not errors,errors
        mobile=await browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
        mobile.on('pageerror',lambda error:errors.append(str(error)))
        await mobile.set_content((ROOT/'MatchLab_3_3_Simulacion_Corregida.html').read_text(),wait_until='load')
        assert await mobile.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'horizontal overflow'
        assert not errors,errors
        await browser.close()
        print('PASS: 3.3 forecast free/conditional, reproducible seed, shared simulation, 3^38 index, mobile')

if __name__=='__main__':
    asyncio.run(main())