/**
 * SimDemo — interactive end-to-end simulation running fully in-browser.
 *
 * Layer 1 (JS port): pool_image → classify (matrix multiply + bias)
 * Layer 2 (JS port): animated discrete-event accumulator replay
 * Layer 3: pre-computed Pymunk detent analysis chart
 */
import { useState, useRef, useEffect, useCallback } from "react";
import MODEL from "./modelData.json";

// ── design tokens (reuse palette from SimPlan) ────────────────────────────────
const C = {
  bg:       "#F5F2EB", panel:    "#FDFBF7", ink:      "#1A1C23",
  rule:     "#CCC8BC", blue:     "#1B4F8A", blueL:    "#2E6EB5",
  blueFill: "#EAF0F9", amber:    "#B85C00", amberFill:"#FDF3E7",
  green:    "#1E6B45", greenFill:"#E8F5EE", grey:     "#6B6860",
  greyFill: "#EEECE8", white:    "#FFFFFF",
};
const DIGITS = ["0","1","2","3","4","5","6","7","8","9"];

// ── Layer 1: JS port ─────────────────────────────────────────────────────────

function poolImage(pixels28x28) {
  // pixels28x28: flat Float32Array or Array of length 784, values 0-255
  const counts = new Array(49).fill(0);
  for (let tr = 0; tr < 7; tr++) {
    for (let tc = 0; tc < 7; tc++) {
      let sum = 0;
      for (let pr = 0; pr < 4; pr++)
        for (let pc = 0; pc < 4; pc++)
          sum += pixels28x28[(tr * 4 + pr) * 28 + (tc * 4 + pc)];
      counts[tr * 7 + tc] = sum;
    }
  }
  return counts;
}

function classify(pooled, weights, bias) {
  // weights: 49×10 array of arrays, bias: length-10 array
  return bias.map((b, d) => b + pooled.reduce((acc, p, f) => acc + weights[f][d] * p, 0));
}

// ── Layer 2: event-step simulator ────────────────────────────────────────────

function buildEvents(pooled, weights, bias) {
  // Returns ordered list of {t, kind, data} similar to SimPy Layer 2
  const events = [];
  let t = 0;
  const T_PULSE = 1, T_ADVANCE = 0.5, T_RESET_STEP = 1, T_RESET_SETTLE = 2;
  const accs = [...bias];

  // Reset phase
  events.push({ t, kind: "RESET_START", accs: [...accs] });
  const resetSchedule = [[0,[0,1]],[1,[2]],[2,[3]],[3,[4]],[4,[5]],[5,[6]],[6,[7]],[7,[8]],[8,[9]]];
  for (const [dt, ids] of resetSchedule) {
    t = dt;
    for (const id of ids) {
      events.push({ t, kind: "CAM_RESET", acc: id, value: bias[id], accs: [...accs] });
    }
  }
  t += T_RESET_SETTLE;
  events.push({ t, kind: "RESET_DONE", accs: [...accs] });

  // Feature cycles
  for (let f = 0; f < 49; f++) {
    const count = pooled[f];
    events.push({ t, kind: "FEATURE_START", f, count, accs: [...accs] });
    for (let p = 0; p < count; p++) {
      t += T_PULSE;
      for (let d = 0; d < 10; d++) accs[d] += weights[f][d];
      events.push({ t, kind: "PULSE", f, p, accs: [...accs] });
    }
    t += T_ADVANCE;
    events.push({ t, kind: "FEATURE_END", f, accs: [...accs] });
  }
  events.push({ t, kind: "DONE", prediction: accs.indexOf(Math.max(...accs)), accs: [...accs] });
  return events;
}

// ── Canvas drawing pad ────────────────────────────────────────────────────────

function DrawPad({ onPixels, size = 280 }) {
  const canvasRef = useRef(null);
  const drawing   = useRef(false);

  const getCtx = () => canvasRef.current?.getContext("2d");

  const clear = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
  }, [size]);

  useEffect(() => { clear(); }, [clear]);

  const toPixels = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return null;
    const raw = ctx.getImageData(0, 0, size, size).data;
    // downsample to 28×28 by averaging each 10×10 block (size=280)
    const scale = size / 28;
    const out   = new Float32Array(784);
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        let sum = 0;
        for (let dr = 0; dr < scale; dr++)
          for (let dc = 0; dc < scale; dc++) {
            const idx = 4 * ((r * scale + dr) * size + (c * scale + dc));
            sum += raw[idx]; // red channel (greyscale)
          }
        out[r * 28 + c] = sum / (scale * scale);
      }
    }
    return out;
  }, [size]);

  const paint = useCallback((e) => {
    if (!drawing.current) return;
    const ctx  = getCtx();
    const rect = canvasRef.current.getBoundingClientRect();
    const x    = (e.clientX ?? e.touches[0].clientX) - rect.left;
    const y    = (e.clientY ?? e.touches[0].clientY) - rect.top;
    ctx.fillStyle = "#FFF";
    ctx.beginPath();
    ctx.arc(x, y, size / 28 * 1.5, 0, Math.PI * 2);
    ctx.fill();
  }, [size]);

  const start = (e) => { drawing.current = true; paint(e); };
  const stop  = () => { drawing.current = false; onPixels(toPixels()); };

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={size} height={size}
        style={{ border: `2px solid ${C.blue}`, borderRadius: 4, cursor: "crosshair", touchAction: "none", display: "block" }}
        onMouseDown={start} onMouseMove={paint} onMouseUp={stop} onMouseLeave={stop}
        onTouchStart={start} onTouchMove={paint} onTouchEnd={stop}
      />
      <button onClick={() => { clear(); onPixels(null); }}
        style={{ marginTop: 6, width: "100%", padding: "5px 0", background: C.greyFill,
                 border: `1px solid ${C.rule}`, borderRadius: 3, cursor: "pointer",
                 fontSize: 12, color: C.grey }}>
        Clear
      </button>
    </div>
  );
}

// ── Score bar chart ───────────────────────────────────────────────────────────

function ScoreBars({ scores, highlight }) {
  if (!scores) return null;
  const mn = Math.min(...scores), mx = Math.max(...scores);
  const range = mx - mn || 1;
  return (
    <div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: C.grey, marginBottom: 6, letterSpacing: 0.5 }}>
        ACCUMULATOR SCORES
      </div>
      {scores.map((s, d) => {
        const pct  = (s - mn) / range * 100;
        const isWin = d === highlight;
        return (
          <div key={d} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontFamily: "monospace", fontSize: 12, color: isWin ? C.green : C.grey, minWidth: 16, fontWeight: isWin ? 800 : 400 }}>
              {d}
            </span>
            <div style={{ flex: 1, background: C.greyFill, borderRadius: 2, height: 14, position: "relative" }}>
              <div style={{ width: `${pct}%`, background: isWin ? C.green : C.blueL, height: "100%", borderRadius: 2, transition: "width 0.3s" }} />
            </div>
            <span style={{ fontFamily: "monospace", fontSize: 11, color: isWin ? C.green : C.grey, minWidth: 40, textAlign: "right", fontWeight: isWin ? 800 : 400 }}>
              {s}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Feature timeline strip ────────────────────────────────────────────────────

function FeatureGrid({ pooled, currentF }) {
  if (!pooled) return null;
  const mx = Math.max(...pooled, 1);
  return (
    <div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: C.grey, marginBottom: 6, letterSpacing: 0.5 }}>
        FEATURE COUNTS (7×7 TILES)
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 2 }}>
        {pooled.map((v, i) => {
          const intensity = Math.round(v / mx * 255);
          const isCurrent = i === currentF;
          return (
            <div key={i} style={{
              aspectRatio: "1", borderRadius: 2,
              background: isCurrent ? C.amber : `rgb(${255 - intensity}, ${255 - intensity}, 255)`,
              border: isCurrent ? `2px solid ${C.amber}` : `1px solid ${C.rule}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 8, fontFamily: "monospace", color: intensity > 127 ? "#FFF" : C.grey,
            }}>{v}</div>
          );
        })}
      </div>
    </div>
  );
}

// ── Mechanical rack view ──────────────────────────────────────────────────────

const LANE_HEX = ['#C84646','#CD7820','#A49A05','#418C3A','#1C807A',
                  '#1C58AC','#4E35AC','#8C2A98','#B62662','#8A842A'];
const W_HEX = {'-2':'#D23208','-1':'#D7821F','0':'#A8A49A','1':'#329A4C','2':'#146E33'};

function _drawMachine(canvas, springs, renderState) {
  const { curWeightRow, curF, isDone, prediction } = renderState;
  const ctx = canvas.getContext('2d');
  const CW = canvas.width, CH = canvas.height;
  const NL = 10, LANE_W = CW / NL;
  const CAM_Y    = 22;
  const GT       = 44;                    // groove top
  const GB       = CH - 36;              // groove bottom
  const GH       = GB - GT;
  const CY       = GT + GH / 2;          // zero line y
  const TRAVEL   = GH / 2 - 6;           // px max travel
  const TOOTH_PX = 9;                     // px between tooth marks
  const RW = 22, RH = 18;

  // background
  ctx.fillStyle = '#F5F2EB';
  ctx.fillRect(0, 0, CW, CH);

  // progress bar (top)
  if (curF >= 0) {
    const pct = (curF + 1) / 49;
    ctx.fillStyle = '#1B4F8A';
    ctx.fillRect(0, 0, CW * pct, 5);
  }

  // cam shaft rail
  ctx.fillStyle = '#B8B5AB';
  const railY = CAM_Y - 5;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(0, railY, CW, 10, 2);
  else ctx.rect(0, railY, CW, 10);
  ctx.fill();

  // lane separators
  ctx.strokeStyle = '#E0DDD5';
  ctx.lineWidth = 1;
  for (let d = 1; d < NL; d++) {
    ctx.beginPath(); ctx.moveTo(LANE_W * d, 34); ctx.lineTo(LANE_W * d, CH - 26); ctx.stroke();
  }

  // feature / done labels
  ctx.font = 'bold 10px monospace';
  ctx.fillStyle = '#1B4F8A';
  ctx.textAlign = 'left';
  if (curF >= 0) ctx.fillText(`Feature ${curF + 1} / 49`, 4, GT - 5);
  if (isDone) {
    ctx.fillStyle = '#1E6B45';
    ctx.textAlign = 'right';
    ctx.fillText(`→  Digit  ${prediction}  wins`, CW - 4, GT - 5);
  }

  for (let d = 0; d < NL; d++) {
    const cx = LANE_W * d + LANE_W / 2;
    const sp = springs?.[d] ?? { pos: 0 };
    // clamp rack to groove bounds
    const rawY = CY - sp.pos;
    const sy   = Math.max(GT + RH / 2 + 2, Math.min(GB - RH / 2 - 2, rawY));

    // groove
    ctx.fillStyle = '#C8C4B4';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - 10, GT, 20, GH, 4);
    else ctx.rect(cx - 10, GT, 20, GH);
    ctx.fill();

    ctx.fillStyle = '#BAB6A6';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - 7, GT + 2, 14, GH - 4, 3);
    else ctx.rect(cx - 7, GT + 2, 14, GH - 4);
    ctx.fill();

    // tooth marks
    ctx.strokeStyle = '#A8A498';
    ctx.lineWidth = 0.8;
    for (let ty = GT + TOOTH_PX * 0.5; ty < GB; ty += TOOTH_PX) {
      ctx.beginPath(); ctx.moveTo(cx - 10, ty); ctx.lineTo(cx + 10, ty); ctx.stroke();
    }

    // zero line
    ctx.strokeStyle = '#888480';
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(cx - 13, CY); ctx.lineTo(cx + 13, CY); ctx.stroke();

    // displacement fill
    const fillTop = Math.min(sy + RH / 2, CY);
    const fillBot = Math.max(sy - RH / 2, CY);
    const fh = Math.abs(fillBot - fillTop);
    if (fh > 1) {
      ctx.fillStyle = sp.pos >= 0 ? '#1E6B4550' : '#B85C0050';
      ctx.fillRect(cx - 5, fillTop, 10, fh);
    }

    // rack body — shadow then body
    ctx.fillStyle = '#00000030';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - RW/2 + 2, sy - RH/2 + 2, RW, RH, 3);
    else ctx.rect(cx - RW/2 + 2, sy - RH/2 + 2, RW, RH);
    ctx.fill();

    ctx.fillStyle = LANE_HEX[d];
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - RW/2, sy - RH/2, RW, RH, 3);
    else ctx.rect(cx - RW/2, sy - RH/2, RW, RH);
    ctx.fill();

    // specular highlight
    ctx.fillStyle = '#FFFFFF35';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - RW/2 + 2, sy - RH/2 + 2, RW - 4, 4, 1);
    else ctx.rect(cx - RW/2 + 2, sy - RH/2 + 2, RW - 4, 4);
    ctx.fill();

    // predicted win star
    if (isDone && d === prediction) {
      ctx.fillStyle = '#C3A014';
      ctx.font = 'bold 15px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('★', cx, sy - RH / 2 - 4);
    }

    // cam lobe indicator on shaft
    const w = curWeightRow ? curWeightRow[d] : 0;
    ctx.fillStyle = W_HEX[String(w)] ?? '#888';
    ctx.beginPath(); ctx.arc(cx, CAM_Y, 11, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#FFFFFF50'; ctx.lineWidth = 1.5; ctx.stroke();

    ctx.fillStyle = '#FFF';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText((w > 0 ? '+' : '') + w, cx, CAM_Y + 3);

    // digit label
    ctx.fillStyle = LANE_HEX[d];
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(String(d), cx, CH - 8);
  }
}

function MechanicalRackView({ accs, bias, finalScores, curWeightRow, curF, isDone, prediction }) {
  const canvasRef  = useRef(null);
  const physRef    = useRef(null);
  const renderRef  = useRef({ curWeightRow, curF, isDone, prediction });
  const frameRef   = useRef(null);
  const TRAVEL_PX  = 110;   // maximum pixel travel (half the groove height)

  // sync render-state ref (so animation loop always sees latest props)
  useEffect(() => {
    renderRef.current = { curWeightRow, curF, isDone, prediction };
  }, [curWeightRow, curF, isDone, prediction]);

  // initialise spring states once
  useEffect(() => {
    physRef.current = bias.map(() => ({ pos: 0, vel: 0, target: 0 }));
  }, [bias]);

  // update spring targets when accs changes
  useEffect(() => {
    if (!physRef.current || !accs) return;
    // normalization: use the final scores range so the animation is stable
    const ref = finalScores
      ? finalScores.map((s, d) => s - bias[d])
      : accs.map((a, d) => a - bias[d]);
    const maxAbs = Math.max(Math.max(...ref.map(Math.abs)), 1);
    const scale  = TRAVEL_PX / maxAbs;
    accs.forEach((a, d) => {
      if (!physRef.current[d]) return;
      const net = a - bias[d];
      physRef.current[d].target = Math.max(-TRAVEL_PX, Math.min(TRAVEL_PX, net * scale));
    });
  }, [accs, bias, finalScores]);

  // animation loop — spring-mass stepping
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const K = 0.13, DAMP = 0.24;
    function tick() {
      if (physRef.current) {
        physRef.current = physRef.current.map(s => {
          const acc = -K * (s.pos - s.target);
          const vel = (s.vel + acc) * (1 - DAMP);
          return { pos: s.pos + vel, vel, target: s.target };
        });
      }
      _drawMachine(canvas, physRef.current, renderRef.current);
      frameRef.current = requestAnimationFrame(tick);
    }
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, []);   // intentionally empty — loop runs for component lifetime

  return (
    <div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: C.grey, marginBottom: 6, letterSpacing: 0.5 }}>
        MECHANICAL VIEW — 10 ACCUMULATOR RACKS  (spring physics)
      </div>
      <canvas ref={canvasRef} width={880} height={310}
        style={{ width: "100%", height: "auto", display: "block",
                 borderRadius: 4, border: `1px solid ${C.rule}` }} />
    </div>
  );
}

// ── Accumulator animation ─────────────────────────────────────────────────────

function AccumulatorBars({ accs, bias, final }) {
  if (!accs) return null;
  const mn = Math.min(...accs), mx = Math.max(...accs);
  const rng = mx - mn || 1;
  const pred = accs.indexOf(Math.max(...accs));
  return (
    <div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: C.grey, marginBottom: 6, letterSpacing: 0.5 }}>
        ACCUMULATOR STATE
      </div>
      <div style={{ display: "flex", gap: 4, height: 100, alignItems: "flex-end" }}>
        {accs.map((v, d) => {
          const h  = Math.max(2, ((v - mn) / rng) * 96);
          const win = d === pred;
          return (
            <div key={d} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <div style={{ width: "100%", height: h, background: win ? C.green : C.blueL, borderRadius: "2px 2px 0 0", transition: "height 0.08s", position: "relative" }}>
                {win && final && (
                  <div style={{ position: "absolute", top: -18, left: "50%", transform: "translateX(-50%)", fontSize: 14 }}>★</div>
                )}
              </div>
              <span style={{ fontFamily: "monospace", fontSize: 10, color: win ? C.green : C.grey, fontWeight: win ? 800 : 400 }}>{d}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main demo component ───────────────────────────────────────────────────────

export default function SimDemo() {
  const [pixels,    setPixels]    = useState(null);
  const [pooled,    setPooled]    = useState(null);
  const [scores,    setScores]    = useState(null);
  const [animating, setAnimating] = useState(false);
  const [eventIdx,  setEventIdx]  = useState(0);
  const [events,    setEvents]    = useState([]);
  const [presetIdx, setPresetIdx] = useState(null);
  const animRef = useRef(null);

  const { weights, bias, presets } = MODEL;

  // ── run classification ──────────────────────────────────────────────────────

  const runClassify = useCallback((pld) => {
    if (!pld) return;
    const sc = classify(pld, weights, bias);
    setScores(sc);
    setPooled(pld);
    const evts = buildEvents(pld, weights, bias);
    setEvents(evts);
    setEventIdx(0);
  }, [weights, bias]);

  const handlePixels = useCallback((px) => {
    setPresetIdx(null);
    if (!px) { setScores(null); setPooled(null); setEvents([]); return; }
    setPixels(px);
    runClassify(poolImage(px));
  }, [runClassify]);

  const loadPreset = (i) => {
    setPresetIdx(i);
    setPixels(null);
    const pld = presets[i].pooled;
    runClassify(pld);
  };

  // ── animation playback ──────────────────────────────────────────────────────

  const startAnim = () => {
    if (!events.length) return;
    setAnimating(true);
    setEventIdx(0);
  };

  useEffect(() => {
    if (!animating) return;
    if (eventIdx >= events.length) { setAnimating(false); return; }
    const delay = events[eventIdx].kind === "PULSE" ? 18 : 60;
    animRef.current = setTimeout(() => setEventIdx(i => i + 1), delay);
    return () => clearTimeout(animRef.current);
  }, [animating, eventIdx, events]);

  const stopAnim  = () => { clearTimeout(animRef.current); setAnimating(false); };
  const resetAnim = () => { stopAnim(); setEventIdx(0); };

  const curEvent   = events[eventIdx] ?? null;
  const curAccs    = curEvent?.accs ?? bias;
  const curF       = curEvent?.kind === "PULSE" || curEvent?.kind === "FEATURE_START"
                     ? curEvent.f : (curEvent?.kind === "FEATURE_END" ? curEvent.f : -1);
  const isDone     = curEvent?.kind === "DONE";
  const prediction = scores ? scores.indexOf(Math.max(...scores)) : null;
  const curWeightRow = curF >= 0 ? weights[curF] : null;

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'DM Sans','Segoe UI',sans-serif", color: C.ink }}>

      {/* Header */}
      <div style={{ background: C.blue, color: C.white, padding: "24px 32px 20px" }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, letterSpacing: 2, opacity: 0.7, marginBottom: 6 }}>
          MECHANICAL MNIST — LIVE SIMULATION
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, letterSpacing: -0.5 }}>
          Interactive 3-Layer Simulation
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, opacity: 0.85, maxWidth: 640 }}>
          Draw a digit, or pick a preset. L1 (NumPy arithmetic) and L2 (event replay) run
          entirely in your browser. L3 Pymunk results are pre-computed.
        </div>
      </div>

      <div style={{ padding: "20px 24px", maxWidth: 1100, margin: "0 auto" }}>

        {/* ── Row 1: input + L1 result ── */}
        <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 20, marginBottom: 20 }}>

          {/* Drawing pad */}
          <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16 }}>
            <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.blue, marginBottom: 10, letterSpacing: 0.5 }}>
              DRAW A DIGIT
            </div>
            <DrawPad onPixels={handlePixels} size={224} />
          </div>

          {/* Preset selector */}
          <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16 }}>
            <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.blue, marginBottom: 10, letterSpacing: 0.5 }}>
              PRESET TEST IMAGES (2 per digit)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 6 }}>
              {presets.map((p, i) => (
                <button key={i} onClick={() => loadPreset(i)}
                  style={{
                    background: presetIdx === i ? C.blue : C.greyFill,
                    color: presetIdx === i ? C.white : C.ink,
                    border: `1px solid ${presetIdx === i ? C.blue : C.rule}`,
                    borderRadius: 4, padding: "8px 4px", cursor: "pointer",
                    fontFamily: "monospace", fontSize: 12, fontWeight: presetIdx === i ? 800 : 400,
                  }}>
                  {p.label}
                </button>
              ))}
            </div>
            {scores && (
              <div style={{ marginTop: 16, padding: "10px 14px", background: prediction !== null ? C.greenFill : C.greyFill,
                            border: `1px solid ${prediction !== null ? C.green : C.rule}`, borderRadius: 4, textAlign: "center" }}>
                <div style={{ fontFamily: "monospace", fontSize: 10, color: C.grey, marginBottom: 4 }}>PREDICTED DIGIT</div>
                <div style={{ fontSize: 40, fontWeight: 900, color: C.green, lineHeight: 1 }}>{prediction}</div>
                <div style={{ fontSize: 11, color: C.grey, marginTop: 4 }}>score = {scores[prediction]}</div>
              </div>
            )}
          </div>

          {/* Feature grid */}
          <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16 }}>
            <FeatureGrid pooled={pooled} currentF={curF} />
            {pooled && (
              <div style={{ marginTop: 12, fontFamily: "monospace", fontSize: 11, color: C.grey }}>
                Total pixel-count: {pooled.reduce((a,b)=>a+b,0)}
                &nbsp;&nbsp;Max tile: {Math.max(...pooled)}
              </div>
            )}
          </div>
        </div>

        {/* ── Row 1b: Mechanical rack simulation ── */}
        {pooled && (
          <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16, marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.amber, letterSpacing: 0.5 }}>
                LAYER 2 — MECHANICAL SIMULATION  (10 accumulator lanes, spring physics)
              </div>
              <div style={{ fontSize: 11, color: C.grey }}>
                Cam lobes → rack advances → detent snaps → score
              </div>
            </div>
            <MechanicalRackView
              accs={curAccs}
              bias={bias}
              finalScores={scores}
              curWeightRow={curWeightRow}
              curF={curF}
              isDone={isDone}
              prediction={prediction}
            />
            <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              {["▶ Play","■ Stop","↺ Reset"].map((label, i) => (
                <button key={i}
                  onClick={[startAnim, stopAnim, resetAnim][i]}
                  disabled={!events.length || (i === 0 && animating)}
                  style={{
                    padding: "5px 14px", border: `1px solid ${C.rule}`, borderRadius: 3,
                    background: !events.length ? C.greyFill : C.blueFill,
                    color: !events.length ? C.grey : C.blue,
                    cursor: "pointer", fontFamily: "monospace", fontSize: 12,
                    opacity: (!events.length || (i === 0 && animating)) ? 0.4 : 1,
                  }}>{label}
                </button>
              ))}
              {events.length > 0 && (
                <div style={{ flex: 1, minWidth: 160 }}>
                  <input type="range" min={0} max={events.length - 1} value={eventIdx}
                    onChange={e => { stopAnim(); setEventIdx(+e.target.value); }}
                    style={{ width: "100%", accentColor: C.amber }} />
                </div>
              )}
              {curEvent && (
                <span style={{ fontFamily: "monospace", fontSize: 10, color: C.grey }}>
                  <span style={{ color: C.blueL, fontWeight: 700 }}>{curEvent.kind}</span>
                  {curEvent.kind === "FEATURE_START" && ` f=${curEvent.f} count=${curEvent.count}`}
                  {curEvent.kind === "PULSE" && ` f=${curEvent.f} p=${curEvent.p}`}
                  {curEvent.kind === "DONE" && ` → digit ${curEvent.prediction} wins`}
                </span>
              )}
            </div>
          </div>
        )}

        {/* ── Row 2: L1 scores + L2 animation ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>

          {/* L1 score bars */}
          <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16 }}>
            <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.green, marginBottom: 12, letterSpacing: 0.5 }}>
              LAYER 1 — NumPy Reference Scores
            </div>
            {scores
              ? <ScoreBars scores={scores} highlight={prediction} />
              : <div style={{ color: C.grey, fontSize: 13 }}>Draw or select a digit to see scores.</div>}
          </div>

          {/* L2 animation */}
          <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.blue, letterSpacing: 0.5 }}>
                LAYER 2 — Discrete-Event Replay
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                {["▶ Play","■ Stop","↺ Reset"].map((label, i) => (
                  <button key={i}
                    onClick={[startAnim, stopAnim, resetAnim][i]}
                    disabled={!events.length || (i === 0 && animating)}
                    style={{
                      padding: "4px 10px", border: `1px solid ${C.rule}`, borderRadius: 3,
                      background: !events.length ? C.greyFill : C.blueFill,
                      color: !events.length ? C.grey : C.blue,
                      cursor: "pointer", fontFamily: "monospace", fontSize: 11,
                      opacity: (!events.length || (i === 0 && animating)) ? 0.4 : 1,
                    }}>{label}
                  </button>
                ))}
              </div>
            </div>

            <AccumulatorBars accs={curAccs} bias={bias} final={isDone} />

            {curEvent && (
              <div style={{ marginTop: 12, fontFamily: "monospace", fontSize: 11, color: C.grey,
                            background: C.greyFill, padding: "6px 10px", borderRadius: 3 }}>
                <span style={{ color: C.blueL, fontWeight: 700 }}>{curEvent.kind}</span>
                {curEvent.kind === "FEATURE_START" && ` f=${curEvent.f} count=${curEvent.count}`}
                {curEvent.kind === "PULSE" && ` f=${curEvent.f} pulse=${curEvent.p}`}
                {curEvent.kind === "CAM_RESET" && ` acc=${curEvent.acc} → ${curEvent.value}`}
                {curEvent.kind === "DONE" && ` → digit ${curEvent.prediction} wins`}
                &nbsp;&nbsp;
                <span style={{ color: C.rule }}>({eventIdx}/{events.length} events)</span>
              </div>
            )}

            {/* Timeline progress bar */}
            {events.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <input type="range" min={0} max={events.length - 1} value={eventIdx}
                  onChange={e => { stopAnim(); setEventIdx(+e.target.value); }}
                  style={{ width: "100%", accentColor: C.blue }} />
              </div>
            )}
          </div>
        </div>

        {/* ── Row 3: L3 pre-computed chart ── */}
        <div style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 6, padding: 16, marginBottom: 20 }}>
          <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.amber, marginBottom: 12, letterSpacing: 0.5 }}>
            LAYER 3 — Pymunk Detent Spring Analysis (pre-computed)
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 20, alignItems: "start" }}>
            <div>
              <img src="l3_detent_analysis.png" alt="Detent spring sweep"
                style={{ width: "100%", borderRadius: 4, border: `1px solid ${C.rule}` }} />
            </div>
            <div style={{ minWidth: 200 }}>
              {[
                ["Min detent k", "139 N/m"],
                ["Rack drift @ 50 steps", "0.002 mm"],
                ["Threshold", "1.0 mm (half tooth)"],
                ["Jitter @ 2× speed", "No double-advance"],
                ["Tooth pitch", "2.0 mm"],
                ["Cam lobe heights", "0,1,2,3,4 mm"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 12,
                                      padding: "5px 0", borderBottom: `1px solid ${C.rule}`,
                                      fontSize: 12 }}>
                  <span style={{ color: C.grey }}>{k}</span>
                  <span style={{ fontFamily: "monospace", color: C.amber, fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Layer badge row ── */}
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            [C.green,  "L1 NumPy",  "Runs in browser (JS port)"],
            [C.blue,   "L2 SimPy",  "Runs in browser (JS port)"],
            [C.amber,  "L3 Pymunk", "Pre-computed Python/Pymunk"],
          ].map(([color, label, note]) => (
            <div key={label} style={{ background: C.panel, border: `1px solid ${color}44`,
                                      borderLeft: `4px solid ${color}`, borderRadius: 4,
                                      padding: "8px 14px", flex: 1, minWidth: 160 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color }}>{label}</div>
              <div style={{ fontSize: 11, color: C.grey, marginTop: 2 }}>{note}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
