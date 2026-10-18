/* Repetición 2D de una jugada con la revisión VAR encima, al estilo de la transmisión.

   Repeticion.montar(elemento, datos, opciones)
     datos = {cancha:{largo, ancho}, t_evento, fps,
              cuadros:[{t, j:[[id, equipo, x, y, arquero 0/1, visible 0/1], ...], b:[x, y, z?] | null}],
              nombres:{id: "N. Moreno"}, revision:{tipo, motivo, veredicto, linea_x, direccion, pasador, receptor,
              defensor, margen_m, sigma_m, receptor_xy, defensor_xy, balon, vision, explicacion, avisos, zona,
              adelantados, ocasion, x, y}}
     opciones = {auto: true (se reproduce sola y vuelve a empezar), inicio: 'evento', velocidad: 0.5}

   Metros con el origen en el centro e y hacia arriba. Entre cuadros se interpola, así la animación
   es fluida aunque el tracking venga a 10 cuadros por segundo. La cámara sigue al balón; al llegar al
   instante revisado se congela la imagen como en el VAR de la TV: zoom a la jugada, la línea del
   penúltimo defensor y la del atacante se trazan de banda a banda, el margen cuenta hasta su valor
   y aparece el veredicto. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  let serie = 0;
  const reducido = () => window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function el(tag, at, padre) {
    const e = document.createElementNS(NS, tag);
    for (const k in at) e.setAttribute(k, at[k]);
    if (padre) padre.appendChild(e);
    return e;
  }
  function h(tag, cls, padre, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined) e.textContent = txt;
    if (padre) padre.appendChild(e);
    return e;
  }
  const coma = (v, d = 2) => Math.abs(v).toFixed(d).replace('.', ',');
  const signo = (v, d = 2) => (v > 0 ? '+' : v < 0 ? '−' : '±') + coma(v, d);
  const suave = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
  const fase = (t, a, dur) => suave((t - a) / dur);
  const VER = {
    'fuera de lugar': ['Fuera de lugar', 'rojo'], 'habilitado': ['Habilitado', 'verde'], 'dudoso': ['Dudoso', 'ambar'],
    'sin intervención': ['Adelantado sin intervenir', 'gris'], 'no aplica': ['No aplica', 'gris'], 'no revisable': ['No revisable', 'gris'],
    'sin receptor': ['Sin receptor', 'gris'], 'gol': ['Gol', 'verde'], 'no entró': ['No entró', 'rojo'], 'salió': ['Balón fuera', 'rojo'],
    'dentro del área': ['Dentro del área', 'rojo'], 'fuera del área': ['Fuera del área', 'verde'], 'gol válido': ['Gol válido', 'verde'],
    'revisar': ['Revisar', 'ambar'], 'ocasión manifiesta': ['Ocasión manifiesta', 'rojo'], 'no es ocasión manifiesta': ['Sin ocasión manifiesta', 'verde']
  };
  const TITULO = {
    'fuera de lugar': 'Revisión de fuera de lugar', 'línea de gol': 'Tecnología de línea de gol', 'revisión de gol': 'Revisión del gol',
    'balón fuera': 'Balón fuera de juego', 'falta en el área': 'Falta: ¿dentro del área?', 'ocasión manifiesta': 'Ocasión manifiesta de gol',
    'posible penal': 'Posible penal'
  };

  function cancha(g, L, W) {
    // franjas de corte del césped
    const n = 14, paso = 2 * L / n;
    for (let i = 0; i < n; i++) el('rect', {class: i % 2 ? 'cesped2' : 'cesped1', x: -L + i * paso, y: -W - 4, width: paso + 0.02, height: 2 * W + 8}, g);
    const c = {class: 'c'};
    el('rect', {...c, x: -L, y: -W, width: 2 * L, height: 2 * W}, g);
    el('line', {...c, x1: 0, y1: -W, x2: 0, y2: W}, g);
    el('circle', {...c, cx: 0, cy: 0, r: 9.15}, g);
    el('circle', {class: 'c-p', cx: 0, cy: 0, r: 0.3}, g);
    for (const s of [-1, 1]) {
      const x0 = s * L;
      el('path', {...c, d: `M${x0} ${-20.16}H${x0 - s * 16.5}V20.16H${x0}`}, g);
      el('path', {...c, d: `M${x0} ${-9.16}H${x0 - s * 5.5}V9.16H${x0}`}, g);
      el('circle', {class: 'c-p', cx: x0 - s * 11, cy: 0, r: 0.3}, g);
      const dx = 16.5 - 11, dy = Math.sqrt(9.15 * 9.15 - dx * dx);
      el('path', {...c, d: `M${x0 - s * 16.5} ${-dy}A9.15 9.15 0 0 ${s > 0 ? 0 : 1} ${x0 - s * 16.5} ${dy}`}, g);
      el('path', {class: 'arco', d: `M${x0} -3.66H${x0 + s * 1.8}V3.66H${x0}`}, g);
      for (const sy of [-1, 1]) el('path', {...c, d: `M${x0} ${sy * W - sy * 1}A1 1 0 0 ${s * sy > 0 ? 0 : 1} ${x0 - s * 1} ${sy * W}`}, g);
    }
  }

  function montar(raiz, d, op = {}) {
    if (!d || !d.cuadros || !d.cuadros.length) { raiz.textContent = 'Sin posiciones para esta jugada.'; return null; }
    const id = 'rep' + (++serie);
    const L = (d.cancha && d.cancha.largo || 105) / 2, W = (d.cancha && d.cancha.ancho || 68) / 2, M = 4;
    const rv = d.revision || {};
    const nombres = d.nombres || {};
    const estatico = d.cuadros.length < 2;
    const sinMov = reducido();
    const ASP = 16 / 10;
    raiz.classList.add('rep');
    raiz.innerHTML = '';
    const escena = h('div', 'rep-escena', raiz);
    const plano = h('div', 'rep-plano', escena);
    const svg = el('svg', {class: 'rep-svg', viewBox: `${-L - M} ${-W - M} ${2 * (L + M)} ${2 * (L + M) / ASP}`, preserveAspectRatio: 'xMidYMid slice',
      role: 'img', 'aria-label': rv.explicacion ? `Repetición: ${rv.explicacion}` : 'Repetición de la jugada'}, plano);
    const defs = el('defs', {}, svg);
    const mk = el('marker', {id: id + '-flecha', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto-start-reverse'}, defs);
    el('path', {d: 'M0 0L10 5L0 10z', fill: 'currentColor'}, mk);
    const grad = el('radialGradient', {id: id + '-sombra'}, defs);
    el('stop', {offset: '0', 'stop-color': '#000', 'stop-opacity': '.45'}, grad);
    el('stop', {offset: '1', 'stop-color': '#000', 'stop-opacity': '0'}, grad);
    const P = (x, y) => [x, -y];
    const todo = el('g', {}, svg);
    cancha(el('g', {}, todo), L, W);
    if (rv.zona && rv.zona.length > 2) el('polygon', {class: 'zona', points: rv.zona.map(([x, y]) => P(x, y).join(',')).join(' ')}, todo);
    const gVar = el('g', {class: 'var'}, todo);
    const gEst = el('g', {}, todo);
    const gJug = el('g', {}, todo);
    const gEt = el('g', {}, todo);

    // ------------------------------------------------ superposiciones (HTML)
    const sello = h('div', 'rep-sello', escena);
    const selloTit = h('span', 'rep-sello-var', sello, 'VAR');
    const selloTxt = h('span', 'rep-sello-tit', sello, TITULO[rv.tipo] || 'Revisión');
    const ver = h('div', 'rep-veredicto', escena);
    const verB = h('b', '', ver);
    const verM = h('span', '', ver);
    const flash = h('div', 'rep-flash', escena);
    const cuenta = el('g', {class: 'rep-cuenta'}, null);
    const mapa = el('svg', {class: 'rep-mapa', viewBox: `${-L - 2} ${-W - 2} ${2 * L + 4} ${2 * W + 4}`, 'aria-hidden': 'true'}, escena);
    el('rect', {class: 'm-c', x: -L, y: -W, width: 2 * L, height: 2 * W}, mapa);
    el('line', {class: 'm-c', x1: 0, y1: -W, x2: 0, y2: W}, mapa);
    for (const s of [-1, 1]) el('path', {class: 'm-c', d: `M${s * L} -20.16H${s * (L - 16.5)}V20.16H${s * L}`}, mapa);
    const mVista = el('rect', {class: 'm-vista'}, mapa);
    const mBal = el('circle', {class: 'm-bal', r: 1.6}, mapa);
    const mLinea = el('line', {class: 'm-linea', y1: -W, y2: W}, mapa);

    // ------------------------------------------------ revisión VAR
    const enVar = new Set([rv.pasador, rv.receptor, rv.defensor].filter(v => v !== undefined && v !== null).map(String));
    const dir = rv.direccion || 1;
    const esFuera = rv.tipo === 'fuera de lugar' && rv.linea_x !== undefined && rv.linea_x !== null;
    const xl = esFuera ? rv.linea_x : null, xr = esFuera && rv.receptor_xy ? rv.receptor_xy[0] : null;
    let foco = null, anchoFoco = 30;
    const piezas = {};
    if (esFuera) {
      if (xr !== null) {
        const clase = rv.veredicto === 'fuera de lugar' ? 'banda-mal' : (rv.veredicto === 'dudoso' || rv.veredicto === 'revisar') ? 'banda-duda' : 'banda-ok';
        piezas.banda = el('rect', {class: clase, x: Math.min(xl, xr), y: -W, width: Math.abs(xr - xl), height: 2 * W}, gVar);
        piezas.ata = el('line', {class: 'l-ata', x1: xr, x2: xr, y1: -W, y2: -W}, gVar);
      }
      piezas.def = el('line', {class: 'l-def', x1: xl, x2: xl, y1: -W, y2: -W}, gVar);
      const yr = rv.receptor_xy ? -rv.receptor_xy[1] : (rv.defensor_xy ? -rv.defensor_xy[1] : 0);
      foco = [xr !== null ? (xl + xr) / 2 : xl, yr];
      anchoFoco = Math.max(24, Math.abs((xr ?? xl) - xl) * 3 + 14);
      if (rv.vision) {
        const [b, g] = rv.vision;
        piezas.vision = el('line', {class: 'vision', x1: b[0], y1: -b[1], x2: b[0], y2: -b[1]}, gVar);
        piezas.visionFin = [g[0], -g[1]];
        foco = [(b[0] + g[0] + (xr ?? xl)) / 3, (-b[1] - g[1] + yr) / 3];
        anchoFoco = Math.max(28, Math.abs(b[0] - g[0]) + 16);
      }
      if (xr !== null) {
        piezas.cuentaFondo = el('rect', {class: 'cuenta-fondo', x: -4.2, y: -1.6, width: 8.4, height: 3.2, rx: 0.8}, cuenta);
        piezas.cuentaTxt = el('text', {class: 'cuenta-txt', 'text-anchor': 'middle', y: 0.75}, cuenta);
        gVar.appendChild(cuenta);
      }
      for (const a of rv.adelantados || []) {
        if (String(a.id) === String(rv.receptor)) continue;
        el('circle', {class: 'otro-adel', cx: a.x, cy: -a.y, r: 1.9}, gVar);
      }
    } else if (rv.tipo === 'línea de gol' || rv.tipo === 'balón fuera') {
      if (rv.balon) {
        const [bx, by] = rv.balon;
        if (rv.tipo === 'línea de gol' || rv.linea === 'de meta') {
          const xg = (bx >= 0 ? 1 : -1) * L;
          piezas.linea = el('line', {class: 'l-def', x1: xg, x2: xg, y1: -W, y2: -W}, gVar);
          piezas.lineaV = true;
        } else {
          const yg = (by >= 0 ? -1 : 1) * W;
          piezas.linea = el('line', {class: 'l-def', x1: -L, x2: -L, y1: yg, y2: yg}, gVar);
          piezas.lineaV = false;
        }
        piezas.punto = el('circle', {class: 'punto-bal', cx: bx, cy: -by, r: 0.6}, gVar);
        piezas.anillo = el('circle', {class: 'anillo', cx: bx, cy: -by, r: 1.2}, gVar);
        foco = [bx, -by];
        anchoFoco = 22;
      }
    } else if ((rv.tipo === 'falta en el área' || rv.tipo === 'posible penal' || rv.tipo === 'ocasión manifiesta') && rv.x !== undefined) {
      const lado = rv.direccion || (rv.x >= 0 ? 1 : -1), x0 = lado * L;
      piezas.area = el('path', {class: 'area-falta', d: `M${x0} -20.16H${x0 - lado * 16.5}V20.16H${x0}Z`}, gVar);
      const oc = rv.ocasion;
      if (oc && oc.criterios) {
        // corredor hacia el arco: los defensores que podrían evitar el gol
        const y0 = -rv.y, xs = rv.x;
        const pts = [[xs, y0 - 4], [x0, -4.66], [x0, 4.66], [xs, y0 + 4]];
        piezas.corredor = el('polygon', {class: 'corredor', points: pts.map(p => p.join(',')).join(' ')}, gVar);
      }
      piezas.punto = el('circle', {class: 'punto-bal falta', cx: rv.x, cy: -rv.y, r: 0.8}, gVar);
      piezas.anillo = el('circle', {class: 'anillo', cx: rv.x, cy: -rv.y, r: 1.4}, gVar);
      // encuadre: el punto de la falta y el área completa
      const xs = [rv.x, x0, x0 - lado * 16.5], ys = [-rv.y, -20.16, 20.16];
      const x1 = Math.min(...xs) - 4, x2 = Math.max(...xs) + 4, y1 = Math.min(...ys) - 4, y2 = Math.max(...ys) + 4;
      foco = [(x1 + x2) / 2, (y1 + y2) / 2];
      anchoFoco = Math.max(x2 - x1, (y2 - y1) * ASP);
    }
    const tieneVar = !!rv.veredicto;
    selloTxt.textContent = rv.motivo === 'rebote' ? 'Fuera de lugar en el rebote' : rv.motivo === 'interferencia' ? 'Interferencia con el arquero' : (TITULO[rv.tipo] || 'Revisión');
    if (tieneVar) {
      const [txt, cl] = VER[rv.veredicto] || [rv.veredicto, 'gris'];
      verB.textContent = txt; verB.className = cl;
      if (rv.margen_m === undefined || rv.margen_m === null) verM.remove();
    }

    // ------------------------------------------------ jugadores (un elemento por id, se mueven)
    const cuadros = d.cuadros.map(c => ({t: c.t, b: c.b, m: new Map(c.j.map(r => [String(r[0]), r]))}));
    const ids = new Map();
    for (const c of cuadros) for (const [k, r] of c.m) if (!ids.has(k)) ids.set(k, r);
    const nodos = new Map();
    for (const [k, r] of ids) {
      const g = el('g', {class: 'jug'}, gJug);
      const aro = el('circle', {class: 'aro', r: 1.7}, g);
      el('ellipse', {class: 'sombra-j', rx: 1.15, ry: 0.55, cy: 0.75}, g);
      const c = el('circle', {class: `j j-${r[1]}${r[4] ? ' arq' : ''}`, r: 1.0}, g);
      const tit = el('title', {}, c); tit.textContent = nombres[k] || (r[4] ? 'Arquero' : 'Jugador');
      const est = el('polyline', {class: `est-l est-${r[1]}`}, gEst);
      let et = null, rol = null;
      if (enVar.has(k)) {
        rol = k === String(rv.pasador) ? 'pas' : k === String(rv.receptor) ? 'rec' : 'def';
        aro.classList.add('aro-' + rol);
        et = el('text', {class: 'et', 'text-anchor': 'middle'}, gEt);
        const rolTxt = rol === 'pas' ? (rv.motivo === 'rebote' || rv.motivo === 'interferencia' ? 'remata' : rv.tipo === 'fuera de lugar' ? 'pasa' : 'víctima')
          : rol === 'rec' ? (rv.tipo === 'fuera de lugar' ? (rv.motivo === 'rebote' ? 'rebote' : rv.motivo === 'interferencia' ? 'interfiere' : 'recibe') : 'atacante')
          : (rv.tipo === 'fuera de lugar' ? 'defensor' : 'rival');
        et.textContent = nombres[k] ? `${nombres[k]} · ${rolTxt}` : rolTxt;
      }
      nodos.set(k, {g, c, aro, est, et, rol, hist: [], dy: rol === 'def' ? 3.6 : -2.5});
    }
    const balSombra = el('ellipse', {rx: 0.8, ry: 0.42, fill: `url(#${id}-sombra)`}, gJug);
    const bal = el('circle', {class: 'bal', r: 0.55}, gJug);
    const balEst = el('polyline', {class: 'est-l est-bal'}, gEst);
    const balHist = [];
    const pase = el('line', {class: 'pase', 'marker-end': `url(#${id}-flecha)`}, gVar);

    // ------------------------------------------------ controles
    const t0 = cuadros[0].t, t1 = cuadros[cuadros.length - 1].t, te = d.t_evento !== undefined ? d.t_evento : t0;
    const dtCuadro = estatico ? 0.1 : Math.max(0.02, (t1 - t0) / Math.max(1, cuadros.length - 1));
    const ctl = h('div', 'rep-ctl', raiz);
    const btn = (txt, et, cl = '') => { const b = h('button', cl, ctl, txt); b.type = 'button'; b.setAttribute('aria-label', et); b.title = et; return b; };
    const bPlay = btn('▶', 'Reproducir (espacio)', 'pri');
    const bAtras = btn('⏮', 'Cuadro anterior (←)');
    const bAdel = btn('⏭', 'Cuadro siguiente (→)');
    const linea = h('div', 'rep-linea', ctl);
    const barra = h('input', '', linea); barra.type = 'range'; barra.min = t0; barra.max = t1; barra.step = 0.01; barra.value = t0;
    barra.setAttribute('aria-label', 'Momento de la jugada');
    if (!estatico && t1 > t0) {
      const marca = h('i', 'rep-marca', linea); marca.style.left = `calc(${(100 * (te - t0) / (t1 - t0)).toFixed(2)}% + ${(8 - 16 * (te - t0) / (t1 - t0)).toFixed(1)}px)`;
      marca.title = 'Instante revisado';
    }
    const tTxt = h('span', 'rep-t', ctl);
    const vel = h('select', '', ctl); vel.setAttribute('aria-label', 'Velocidad');
    for (const v of [0.25, 0.5, 1]) { const o = h('option', '', vel, String(v).replace('.', ',') + '×'); o.value = v; }
    vel.value = String(op.velocidad || 0.5);
    const bIr = btn('VAR', 'Ir al instante revisado (V)', 'var-btn');
    const vista = h('select', '', ctl); vista.setAttribute('aria-label', 'Vista');
    [['sigue', 'Sigue la jugada'], ['tv', 'Vista de transmisión'], ['cancha', 'Cancha entera']].forEach(([v, t]) => { const o = h('option', '', vista, t); o.value = v; });
    let pref = null; try { pref = localStorage.getItem('repeticion.vista'); } catch (_) {}
    vista.value = pref && ['sigue', 'tv', 'cancha'].includes(pref) ? pref : 'sigue';
    const lEst = h('label', '', ctl); lEst.innerHTML = '<input type="checkbox" checked> Estelas';
    const cEst = lEst.querySelector('input');
    if (estatico) [bPlay, bAtras, bAdel, linea, vel, lEst].forEach(x => x.hidden = true);

    if (rv.explicacion) h('p', 'rep-txt', raiz, rv.explicacion);
    if (rv.avisos && rv.avisos.length) {
      const ul = h('ul', 'rep-avisos', raiz);
      rv.avisos.forEach(a => h('li', '', ul, a));
    }
    const ley = h('div', 'rep-ley', raiz);
    if (esFuera) {
      ley.innerHTML = '<span><i style="background:var(--rep-def)"></i>' + (rv.linea_por === 'balón' ? 'línea del balón' : 'penúltimo defensor') + '</span>' +
        '<span><i style="background:var(--rep-ata)"></i>' + (rv.motivo === 'interferencia' ? 'atacante que interfiere' : rv.motivo === 'rebote' ? 'juega el rebote' : 'receptor del pase') + '</span>' +
        (rv.vision ? '<span><i style="background:#fff;border:1px dashed #333"></i>visión del arquero</span>' : '') +
        ((rv.adelantados || []).length > 1 ? '<span><i class="anillo-ley"></i>otros adelantados</span>' : '');
    }
    ley.innerHTML += '<span><i class="est-ley"></i>borde punteado: fuera de cámara (estimado)</span>';

    // ------------------------------------------------ estado y dibujo
    let tv = estatico ? te : t0, jugando = false, ultimo = null, enPausaVar = false, tVar = sinMov ? 99 : 0, reanudar = null;
    let cam = null, movida = true, sucio = true, escala = 1;   // cámara: [cx, cy, ancho] en metros (y del SVG)
    function interp(t) {
      let k = 0;
      while (k < cuadros.length - 2 && cuadros[k + 1].t <= t) k++;
      const a = cuadros[k], b = cuadros[Math.min(k + 1, cuadros.length - 1)];
      const f = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 0;
      return {a, b, f};
    }
    function posBalon(a, b, f) {
      const ba = a.b, bb = b.b;
      if (ba && bb) return [ba[0] + (bb[0] - ba[0]) * f, ba[1] + (bb[1] - ba[1]) * f, ((ba[2] || 0) + ((bb[2] || 0) - (ba[2] || 0)) * f)];
      const q = ba || bb; return q ? [q[0], q[1], q[2] || 0] : null;
    }
    function objetivoCamara(pb, enVarAhora) {
      if (vista.value === 'cancha') return [0, 0, 2 * (L + M)];
      if (enVarAhora && foco) return [foco[0], foco[1], anchoFoco];
      if (pb) return [pb[0], -pb[1], 64];
      return [0, 0, 2 * (L + M)];
    }
    function aplicarCamara(objetivo, dt) {
      if (!cam || sinMov) cam = objetivo.slice();
      else {
        const k = 1 - Math.exp(-dt * 3.2);
        for (let i = 0; i < 3; i++) cam[i] += (objetivo[i] - cam[i]) * k;
      }
      let [cx, cy, w] = cam;
      w = Math.min(w, 2 * (L + M)); const hh = w / ASP;
      cx = Math.max(-L - M + w / 2, Math.min(L + M - w / 2, cx));
      cy = Math.max(-W - M + hh / 2, Math.min(W + M - hh / 2, cy));
      svg.setAttribute('viewBox', `${(cx - w / 2).toFixed(2)} ${(cy - hh / 2).toFixed(2)} ${w.toFixed(2)} ${hh.toFixed(2)}`);
      movida = Math.abs(objetivo[0] - cam[0]) + Math.abs(objetivo[1] - cam[1]) + Math.abs(objetivo[2] - cam[2]) > 0.02;
      const zoomed = w < 2 * (L + M) * 0.8;
      mapa.classList.toggle('ver', zoomed);
      mVista.setAttribute('x', cx - w / 2); mVista.setAttribute('y', cy - hh / 2); mVista.setAttribute('width', w); mVista.setAttribute('height', hh);
      // tamaño de textos constante en pantalla; los jugadores se achican menos que el zoom
      const esc = w / 64;
      escena.style.setProperty('--esc', esc.toFixed(3));
      escala = Math.max(0.42, Math.min(1, esc));
      if (piezas.cuentaTxt && foco) {
        const yr = rv.receptor_xy ? -rv.receptor_xy[1] : foco[1];
        const yc = Math.max(cy - hh / 2 + 3 * esc, Math.min(cy + hh / 2 - 7 * esc, yr + 4.5 * escala + 2 * esc));
        cuenta.setAttribute('transform', `translate(${((xl + xr) / 2).toFixed(2)} ${yc.toFixed(2)}) scale(${(esc * 0.9).toFixed(3)})`);
      }
    }
    function dibujarVar(fx) {
      // fx: segundos desde que se congeló la imagen (o 99 = todo dibujado)
      const pDef = fase(fx, 0.5, 0.7), pAta = fase(fx, 1.2, 0.7), pBanda = fase(fx, 1.8, 0.5), pVer = fase(fx, 2.2, 0.4);
      const y = p => (-W + 2 * W * p);
      if (piezas.def) piezas.def.setAttribute('y2', y(pDef));
      if (piezas.ata) piezas.ata.setAttribute('y2', y(pAta));
      if (piezas.banda) piezas.banda.style.opacity = pBanda;
      if (piezas.linea) {
        if (piezas.lineaV) piezas.linea.setAttribute('y2', y(pDef));
        else piezas.linea.setAttribute('x2', -L + 2 * L * pDef);
      }
      if (piezas.vision) {
        const [x1, y1] = [+piezas.vision.getAttribute('x1'), +piezas.vision.getAttribute('y1')];
        const [x2, y2] = piezas.visionFin;
        piezas.vision.setAttribute('x2', x1 + (x2 - x1) * pAta); piezas.vision.setAttribute('y2', y1 + (y2 - y1) * pAta);
      }
      if (piezas.area) { piezas.area.style.opacity = pDef; piezas.area.style.fillOpacity = 0.16 * pBanda; }
      if (piezas.corredor) piezas.corredor.style.opacity = pAta;
      if (piezas.anillo) { const r = 1.2 + 2.5 * (1 - fase(fx, 0.3, 0.6)); piezas.anillo.setAttribute('r', r); piezas.anillo.style.opacity = fase(fx, 0.3, 0.3); }
      ver.style.opacity = pVer; ver.style.transform = `scale(${1.25 - 0.25 * pVer})`;
      if (rv.margen_m !== undefined && rv.margen_m !== null) {
        const v = rv.margen_m * fase(fx, 1.8, 0.6);
        verM.textContent = `${signo(v)} m` + (rv.sigma_m !== undefined && rv.sigma_m !== null ? `  ±${coma(rv.sigma_m)}` : '');
        if (piezas.cuentaTxt) {
          cuenta.style.opacity = fase(fx, 1.8, 0.3);
          piezas.cuentaTxt.textContent = `${signo(v)} m`;
          cuenta.setAttribute('class', 'rep-cuenta ' + (rv.veredicto === 'fuera de lugar' ? 'rojo' : rv.veredicto === 'habilitado' ? 'verde' : 'ambar'));
        }
      }
      sello.style.opacity = fase(fx, 0, 0.35); sello.style.transform = `translateX(${-14 * (1 - fase(fx, 0, 0.35))}px)`;
      flash.style.opacity = fx < 0.25 && !sinMov ? (1 - fx / 0.25) * 0.55 : 0;
    }
    function dibujar(dtReal = 0) {
      const {a, b, f} = interp(tv);
      const congelado = enPausaVar || estatico;
      escena.classList.toggle('congelado', congelado && tieneVar);
      escena.classList.toggle('tv', vista.value === 'tv');
      for (const [k, n] of nodos) {
        const ra = a.m.get(k), rb = b.m.get(k);
        const r = ra && rb ? [ra[2] + (rb[2] - ra[2]) * f, ra[3] + (rb[3] - ra[3]) * f] : ra ? [ra[2], ra[3]] : rb ? [rb[2], rb[3]] : null;
        if (!r) { n.g.style.display = 'none'; n.est.setAttribute('points', ''); if (n.et) n.et.style.display = 'none'; continue; }
        n.g.style.display = '';
        const [x, y] = P(r[0], r[1]);
        n.g.setAttribute('transform', `translate(${x.toFixed(3)} ${y.toFixed(3)}) scale(${escala.toFixed(3)})`);
        const vis = (ra || rb)[5];
        n.c.classList.toggle('est', !vis);
        n.aro.classList.toggle('ver', !!n.rol && congelado && tieneVar);
        n.g.classList.toggle('apagado', congelado && tieneVar && enVar.size > 0 && !n.rol && esFuera);
        if (n.et) { n.et.style.display = congelado || enVar.size === 0 ? '' : 'none'; n.et.setAttribute('x', x); n.et.setAttribute('y', y + n.dy * escala); }
        if (!congelado) {
          n.hist.push([x, y, tv]);
          while (n.hist.length && (n.hist[0][2] < tv - 1.6 || n.hist[0][2] > tv)) n.hist.shift();
        }
        n.est.setAttribute('points', cEst.checked ? n.hist.map(q => q[0].toFixed(2) + ',' + q[1].toFixed(2)).join(' ') : '');
      }
      const pb = posBalon(a, b, f);
      if (pb) {
        const [x, y] = P(pb[0], pb[1]); const z = Math.max(0, pb[2] || 0);
        bal.style.display = ''; balSombra.style.display = '';
        balSombra.setAttribute('cx', x); balSombra.setAttribute('cy', y + 0.15);
        bal.setAttribute('cx', x); bal.setAttribute('cy', y - Math.min(4, z * 0.55) * escala);
        bal.setAttribute('r', (0.55 + Math.min(0.45, z * 0.08)) * Math.max(0.6, escala));
        balSombra.setAttribute('rx', 0.8 * Math.max(0.6, escala)); balSombra.setAttribute('ry', 0.42 * Math.max(0.6, escala));
        mBal.setAttribute('cx', x); mBal.setAttribute('cy', y);
        if (!congelado) {
          balHist.push([x, y, tv]);
          while (balHist.length && (balHist[0][2] < tv - 1.0 || balHist[0][2] > tv)) balHist.shift();
        }
        balEst.setAttribute('points', cEst.checked ? balHist.map(q => q[0].toFixed(2) + ',' + q[1].toFixed(2)).join(' ') : '');
      } else { bal.style.display = 'none'; balSombra.style.display = 'none'; }
      if (esFuera) { mLinea.setAttribute('x1', xl); mLinea.setAttribute('x2', xl); }
      mLinea.style.display = esFuera && congelado ? '' : 'none';
      // flecha del pase o del tiro
      if (esFuera && congelado) {
        const pp = rv.pasador !== undefined ? (a.m.get(String(rv.pasador)) || b.m.get(String(rv.pasador))) : null;
        const org = pp ? P(pp[2], pp[3]) : (rv.balon ? P(rv.balon[0], rv.balon[1]) : null);
        const dst = rv.motivo === 'rebote' || rv.motivo === 'interferencia' ? [dir * L, 0] : rv.receptor_xy ? P(rv.receptor_xy[0], rv.receptor_xy[1]) : null;
        if (org && dst) {
          const pa = fase(tVar, 0.3, 0.6);
          pase.setAttribute('x1', org[0]); pase.setAttribute('y1', org[1]);
          pase.setAttribute('x2', org[0] + (dst[0] - org[0]) * pa); pase.setAttribute('y2', org[1] + (dst[1] - org[1]) * pa);
          pase.style.display = '';
        }
      } else pase.style.display = 'none';
      gVar.classList.toggle('ver', congelado);
      if (congelado && tieneVar) dibujarVar(tVar); else { ver.style.opacity = 0; sello.style.opacity = 0; cuenta.style.opacity = 0; flash.style.opacity = 0; }
      aplicarCamara(objetivoCamara(pb, congelado && tieneVar), dtReal);
      barra.value = tv;
      const rel = tv - te;
      tTxt.textContent = estatico ? 'instante revisado' : `${rel >= 0 ? '+' : '−'}${coma(rel, 2)} s`;
    }
    function bucle(ahora) {
      if (!vivo) return;
      const dtReal = ultimo === null ? 0 : Math.min(0.1, (ahora - ultimo) / 1000);
      ultimo = ahora;
      if (jugando) {
        const antes = tv;
        tv = Math.min(t1, tv + dtReal * (+vel.value));
        if (antes < te && tv >= te && tieneVar) {          // el VAR congela la imagen en el instante revisado
          tv = te; congelar();
          if (op.auto) reanudar = setTimeout(() => { reproducir(); }, sinMov ? 2500 : 5200);
        }
        if (tv >= t1) {
          jugando = false; marcarPlay();
          if (op.auto) reanudar = setTimeout(() => { tv = t0; limpiar(); reproducir(); }, 1600);
        }
      } else if (enPausaVar || estatico) tVar += dtReal;
      if (jugando || sucio || movida || ((enPausaVar || estatico) && tVar < 3.5)) { sucio = false; dibujar(dtReal); }
      requestAnimationFrame(bucle);
    }
    let vivo = true;
    function limpiar() { for (const n of nodos.values()) n.hist.length = 0; balHist.length = 0; }
    function marcarPlay() { bPlay.textContent = jugando ? '❚❚' : '▶'; bPlay.setAttribute('aria-label', jugando ? 'Pausa (espacio)' : 'Reproducir (espacio)'); bPlay.title = bPlay.getAttribute('aria-label'); }
    function congelar() { jugando = false; enPausaVar = true; tVar = sinMov ? 99 : 0; marcarPlay(); }
    function reproducir() {
      clearTimeout(reanudar);
      if (tv >= t1) { tv = t0; limpiar(); }
      enPausaVar = false; jugando = true; marcarPlay();
    }
    function pausar() { clearTimeout(reanudar); jugando = false; marcarPlay(); }
    function ir(t, var_ = false) {
      clearTimeout(reanudar); jugando = false; marcarPlay(); sucio = true;
      if (Math.abs(t - tv) > 0.35) limpiar();
      tv = Math.max(t0, Math.min(t1, t));
      if (var_ && tieneVar) { enPausaVar = true; tVar = sinMov ? 99 : 0; }
      else { enPausaVar = Math.abs(tv - te) < dtCuadro * 0.5 && tieneVar; tVar = 99; }
    }
    bPlay.onclick = () => (jugando ? pausar() : reproducir());
    bAtras.onclick = () => ir(tv - dtCuadro);
    bAdel.onclick = () => ir(tv + dtCuadro);
    barra.oninput = () => ir(+barra.value);
    bIr.onclick = () => ir(te, true);
    vista.onchange = () => { sucio = true; try { localStorage.setItem('repeticion.vista', vista.value); } catch (_) {} };
    cEst.onchange = () => { sucio = true; };
    raiz.tabIndex = raiz.tabIndex >= 0 ? raiz.tabIndex : 0;
    raiz.addEventListener('keydown', e => {
      if (e.target.closest('select,input[type=checkbox]')) return;
      if (e.key === ' ') { e.preventDefault(); bPlay.click(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); bAtras.click(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); bAdel.click(); }
      else if (e.key === 'v' || e.key === 'V') { e.preventDefault(); bIr.click(); }
    });
    // primer cuadro: con movimiento reducido (o una sola foto) se muestra directo el instante revisado
    if (estatico || sinMov || op.inicio === 'evento') { tv = te; if (tieneVar) { enPausaVar = true; tVar = sinMov ? 99 : 0; } }
    dibujar(0);
    requestAnimationFrame(bucle);
    if (op.auto && !sinMov && !estatico && !(op.inicio === 'evento')) {
      const io = 'IntersectionObserver' in window ? new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); reproducir(); } }) : null;
      if (io) io.observe(raiz); else reproducir();
    }
    return {reproducir, pausar, ir, destruir: () => { vivo = false; clearTimeout(reanudar); raiz.innerHTML = ''; }};
  }
  window.Repeticion = {montar};
})();
