from pathlib import Path
root=Path(__file__).parent
s=(root/'src'/'index.template.html').read_text()
js=(root/'src'/'app.js').read_text()
css=(root/'src'/'style.css').read_text()
data=(root/'data'/'colombia-2026-ii.json').read_text()
s=s.replace('/*STYLE*/',css).replace('/*DATA*/','window.MATCHLAB_API_AUTO=false;window.MATCHLAB_INITIAL_DATA='+data+';').replace('/*JS*/',js)
out=root/'MatchLab_4_2026-11-05.html';out.write_text(s,encoding='utf-8')
print(out,len(s),'bytes')