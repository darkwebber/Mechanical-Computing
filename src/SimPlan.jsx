import { useState } from "react";

// ─── Design tokens ──────────────────────────────────────────────────────────
const C = {
  bg:       "#F5F2EB",
  panel:    "#FDFBF7",
  ink:      "#1A1C23",
  rule:     "#CCC8BC",
  blue:     "#1B4F8A",
  blueL:    "#2E6EB5",
  blueFill: "#EAF0F9",
  amber:    "#B85C00",
  amberFill:"#FDF3E7",
  green:    "#1E6B45",
  greenFill:"#E8F5EE",
  grey:     "#6B6860",
  greyFill: "#EEECE8",
  white:    "#FFFFFF",
};

function Tag({ color = C.blue, bg = C.blueFill, children }) {
  return (
    <span style={{
      display: "inline-block", padding: "1px 8px", borderRadius: 3,
      background: bg, color, fontFamily: "monospace", fontSize: 11,
      fontWeight: 700, letterSpacing: 0.5, border: `1px solid ${color}22`,
    }}>{children}</span>
  );
}

function Rule() {
  return <div style={{ borderBottom: `1px solid ${C.rule}`, margin: "20px 0" }} />;
}

function SectionLabel({ n, title }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginBottom: 16 }}>
      <span style={{
        fontFamily: "monospace", fontSize: 11, fontWeight: 700,
        background: C.blue, color: C.white, padding: "2px 7px", borderRadius: 2,
        letterSpacing: 1, flexShrink: 0,
      }}>L{n}</span>
      <h2 style={{
        fontFamily: "'DM Sans', 'Segoe UI', sans-serif", fontSize: 18,
        fontWeight: 700, color: C.ink, margin: 0, letterSpacing: -0.3,
      }}>{title}</h2>
    </div>
  );
}

function Card({ children, accent = C.blue, style = {} }) {
  return (
    <div style={{
      background: C.panel, border: `1px solid ${accent}33`,
      borderLeft: `4px solid ${accent}`, borderRadius: 4,
      padding: "16px 20px", marginBottom: 16, ...style,
    }}>{children}</div>
  );
}

function InfoBox({ label, text, color = C.blue, bg = C.blueFill }) {
  return (
    <div style={{
      background: bg, border: `1px solid ${color}44`,
      borderRadius: 4, padding: "10px 16px", marginBottom: 12,
      display: "flex", gap: 10, alignItems: "flex-start",
    }}>
      <span style={{ fontWeight: 800, color, fontFamily: "monospace", fontSize: 12, marginTop: 1 }}>
        {label}
      </span>
      <span style={{ color: C.ink, fontSize: 13, lineHeight: 1.55 }}>{text}</span>
    </div>
  );
}

function Th({ children }) {
  return (
    <th style={{
      background: C.blue, color: C.white, padding: "7px 12px",
      fontFamily: "monospace", fontSize: 11, fontWeight: 700,
      letterSpacing: 0.5, textAlign: "left", whiteSpace: "nowrap",
    }}>{children}</th>
  );
}

function Td({ children, mono, accent }) {
  return (
    <td style={{
      padding: "7px 12px", borderBottom: `1px solid ${C.rule}`,
      fontFamily: mono ? "monospace" : "'DM Sans','Segoe UI',sans-serif",
      fontSize: 12, color: accent || C.ink, verticalAlign: "top",
    }}>{children}</td>
  );
}

function Table({ headers, rows }) {
  return (
    <div style={{ overflowX: "auto", marginBottom: 16 }}>
      <table style={{ borderCollapse: "collapse", width: "100%", background: C.panel }}>
        <thead><tr>{headers.map((h, i) => <Th key={i}>{h}</Th>)}</tr></thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 === 1 ? C.greyFill : C.panel }}>
              {row.map((cell, ci) => (
                <Td key={ci} mono={ci === 0} accent={ci === 0 ? C.blueL : null}>
                  {cell}
                </Td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LayerCard({ n, name, tool, type, scope, why, why_not, apis, tasks, color, expanded, onToggle }) {
  return (
    <div style={{
      border: `1px solid ${color}55`, borderTop: `4px solid ${color}`,
      borderRadius: 6, marginBottom: 20, background: C.panel, overflow: "hidden",
    }}>
      <div
        style={{
          display: "flex", alignItems: "center", gap: 12, padding: "14px 20px",
          cursor: "pointer", userSelect: "none",
          background: expanded ? `${color}11` : C.panel,
        }}
        onClick={onToggle}
      >
        <span style={{
          width: 28, height: 28, borderRadius: "50%", background: color,
          color: C.white, fontWeight: 800, fontFamily: "monospace", fontSize: 13,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>{n}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, color: C.ink, fontSize: 15, fontFamily: "'DM Sans','Segoe UI',sans-serif" }}>
            {name}
          </div>
          <div style={{ fontSize: 12, color: C.grey, marginTop: 2 }}>
            <span style={{ fontFamily: "monospace", color }}>{tool}</span>
            <span style={{ margin: "0 8px", color: C.rule }}>|</span>
            {type}
          </div>
        </div>
        <span style={{ color: C.grey, fontSize: 18 }}>{expanded ? "▲" : "▼"}</span>
      </div>

      {expanded && (
        <div style={{ padding: "0 20px 20px" }}>
          <Rule />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
            <InfoBox label="SCOPE" text={scope} color={color} bg={`${color}11`} />
            <InfoBox label="WHY" text={why} color={C.green} bg={C.greenFill} />
          </div>
          <InfoBox label="LIMITATION" text={why_not} color={C.amber} bg={C.amberFill} />

          <div style={{ marginBottom: 10 }}>
            <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color, marginBottom: 6, letterSpacing: 0.5 }}>
              KEY APIs / PRIMITIVES USED
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {apis.map((a, i) => <Tag key={i} color={color} bg={`${color}15`}>{a}</Tag>)}
            </div>
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.ink, marginBottom: 8, letterSpacing: 0.5 }}>
              SIMULATION TASKS FOR THIS LAYER
            </div>
            {tasks.map((t, i) => (
              <div key={i} style={{
                display: "flex", gap: 10, padding: "5px 0",
                borderBottom: i < tasks.length - 1 ? `1px solid ${C.rule}` : "none",
              }}>
                <span style={{ fontFamily: "monospace", fontSize: 11, color, fontWeight: 700, minWidth: 24 }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span style={{ fontSize: 12, color: C.ink, lineHeight: 1.5 }}>{t}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Pipeline() {
  const stages = [
    { label: "MNIST Draw", sub: "Canvas / ESP32", color: C.grey },
    { label: "Pool", sub: "28×28 → 49 counts", color: C.grey },
    { label: "L1 Logic", sub: "Pure Python\nNumPy", color: C.green },
    { label: "L2 Event", sub: "SimPy DES\nmachine cycles", color: C.blue },
    { label: "L3 Kinematic", sub: "Pymunk 2D\ncam/ratchet", color: C.amber },
    { label: "Visualise", sub: "Matplotlib / Pygame\nreal-time output", color: C.blueL },
  ];

  return (
    <div style={{ overflowX: "auto", marginBottom: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 0, minWidth: 600 }}>
        {stages.map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center" }}>
            <div style={{
              background: s.color, color: C.white, borderRadius: 6,
              padding: "10px 14px", textAlign: "center", minWidth: 90,
            }}>
              <div style={{ fontWeight: 700, fontSize: 12, fontFamily: "'DM Sans','Segoe UI',sans-serif" }}>
                {s.label}
              </div>
              <div style={{ fontSize: 10, opacity: 0.85, fontFamily: "monospace", marginTop: 3, whiteSpace: "pre" }}>
                {s.sub}
              </div>
            </div>
            {i < stages.length - 1 && (
              <div style={{
                width: 28, height: 2, background: C.rule, position: "relative",
                display: "flex", alignItems: "center", justifyContent: "flex-end",
              }}>
                <span style={{ color: C.rule, fontSize: 14 }}>▶</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SimPlan() {
  const [expanded, setExpanded] = useState({ 0: true, 1: false, 2: false });
  const [activeTab, setActiveTab] = useState("layers");

  const toggle = (i) => setExpanded(e => ({ ...e, [i]: !e[i] }));

  const layers = [
    {
      n: 1,
      name: "Logical / Arithmetic Simulation",
      tool: "NumPy + Python",
      type: "Deterministic reference implementation",
      color: C.green,
      scope: "Implements score[d] = bias[d] + weights @ pooled exactly. Tests all 49 × 10 weight cells. Runs 10,000 MNIST images to benchmark accuracy.",
      why: "Ground truth. Zero ambiguity. Runs in milliseconds. Any physics layer result must match this exactly. Also generates bias_steps and worst-case score ranges needed for accumulator sizing.",
      why_not: "Does not model timing, mechanical failure, jitter, reset ordering, or any physical behaviour. It is the oracle, not the machine.",
      apis: ["numpy.dot", "numpy.load (csv)", "pandas.read_csv", "sklearn.datasets (MNIST)", "matplotlib.pyplot (score histograms)"],
      tasks: [
        "Load quantized_weights_49x10.csv and bias_steps_10.csv.",
        "Implement pool_image(img_28x28) → counts[49] using 4×4 tile sums.",
        "Implement classify(counts) → scores[10] using NumPy matrix multiply + bias.",
        "Run over full MNIST 10k test set → record accuracy, confusion matrix, per-digit score histograms.",
        "Compute min/max score per digit to verify accumulator range (±50 assumed safe).",
        "Export reference scores for 20 canonical test digits — used to validate Layers 2 and 3.",
      ],
    },
    {
      n: 2,
      name: "Discrete-Event Machine Simulation",
      tool: "SimPy + Python",
      color: C.blue,
      type: "Process-based event simulation with timing and ordering",
      scope: "Models the machine as a sequence of timed events: reset phases, 49 feature cycles, pulse emission, accumulator updates. Each mechanical sub-system is a SimPy Process or Resource. Catches sequencing bugs and timing violations.",
      why: "Verifies the reset cam sequence ordering (Section 6 of blueprint) is correct. Detects race conditions (e.g. detent re-engaged before rack reaches bias stop). Measures total cycle time. Validates ESP32 control logic without any physics.",
      why_not: "Does not model force, contact, or kinematic constraints. A pawl skipping a tooth under load is invisible here. This is a logical/timing model, not a physics model.",
      apis: [
        "simpy.Environment", "simpy.Process", "simpy.Resource",
        "simpy.Timeout", "simpy.Event", "simpy.AnyOf / AllOf",
        "simpy.Store (pulse bus queue)",
      ],
      tasks: [
        "Model ResetCamShaft as a SimPy process: 9 cam events fired in angular-time order (Section 6 timing table).",
        "Model PulseReplayUnit: timeout(N pulses × pulse_period) → emit N events on pulse bus store.",
        "Model FeatureSequencer: waits for advance event, increments row counter, asserts row ≤ 48.",
        "Model 10 WeightSelectorLane processes: each reads current row weight, applies ±1/±2/0 to accumulator integer.",
        "Model 10 AccumulatorModule as SimPy Containers: level starts at bias[d], incremented/decremented per lane events.",
        "Assert final Container levels match Layer 1 reference scores for 20 canonical test images.",
        "Instrument timing: record per-feature cycle time, total run time, reset duration. Plot Gantt chart of events.",
        "Inject fault scenarios: double-advance on sequencer, SOL-A misfire (N+1 count) → observe propagated error.",
      ],
    },
    {
      n: 3,
      name: "Kinematic / Contact Physics Simulation",
      tool: "Pymunk (Chipmunk2D) + Pygame",
      color: C.amber,
      type: "2D rigid-body physics with joint constraints and collision",
      scope: "Physically simulates one accumulator lane end-to-end: cam lobe → follower lever → lane selector fork → enable dog → ratchet pawl → rack-and-pinion → accumulator rack. Verifies that mechanism geometry produces correct integer steps and that detents hold against gravity.",
      why: "Only layer that catches real mechanical failure modes: pawl bounce causing double-advance, detent force insufficient to hold rack, follower losing contact with cam at speed, enable dog partial engagement. Essential before committing to fabrication dimensions.",
      why_not: "Simulating all 10 lanes × 49 rows in full physics would be very slow. Simulate ONE representative lane with a parameterised test sequence. 3D effects (shaft twist, alignment) are out of scope — use Layer 1/2 for those concerns.",
      apis: [
        "pymunk.Space", "pymunk.Body", "pymunk.Shape",
        "pymunk.constraints.PivotJoint", "pymunk.constraints.GearJoint",
        "pymunk.constraints.RatchetJoint", "pymunk.constraints.GrooveJoint",
        "pymunk.constraints.SimpleMotor", "pymunk.constraints.DampedRotarySpring",
        "pymunk.constraints.RotaryLimitJoint",
        "pymunk.pygame_util.DrawOptions",
      ],
      tasks: [
        "Implement cam lobe body: 5 discrete heights (0–4 mm), rotating at clock speed. Spring-loaded follower lever as pivot body.",
        "Verify follower arm tip travels exactly 0/1/2/3/4 mm for each cam height (H0–H4). Check with pymunk body.position readings.",
        "Implement lane selector fork: GrooveJoint sliding body, 5-position RotaryLimitJoint detent at each notch.",
        "Implement enable dog clutch: two bodies, PivotJoint with RotaryLimitJoint to simulate engagement / disengagement.",
        "Implement direction gear path: ADD = direct GearJoint (ratio 1:1), SUB = idler GearJoint (ratio -1:1), selector fork toggles.",
        "Implement bilobed cam for ×2: second cam body with 2 lobes per revolution on same SimpleMotor shaft as ×1 cam.",
        "Implement accumulator rack: kinematic linear body, RatchetJoint against pawl body, DampedRotarySpring as detent.",
        "Run test sequence: 16 pulses with weight +2 → rack must advance exactly +32. 16 pulses with weight -2 → rack must retreat exactly -32.",
        "Run jitter test: advance at 2× normal speed → check pawl bounce does not cause double-advance.",
        "Measure detent spring force needed to resist gravity on a 50-step rack displacement. Compare to designed spring preload.",
        "Render real-time Pygame visualisation: colour-coded cam height, lane state (ADD2/ADD1/IGN/SUB1/SUB2), accumulator rack height bar.",
      ],
    },
  ];

  const fileplan = [
    ["sim/l1_reference.py", "Layer 1", "NumPy classify; pool_image; MNIST accuracy benchmark; exports ref_scores_20.npy"],
    ["sim/l2_machine.py", "Layer 2", "SimPy machine model: reset + 49 cycles + assertion vs ref_scores_20.npy"],
    ["sim/l2_timing.py", "Layer 2", "Gantt chart of SimPy events; cycle time analysis; fault injection harness"],
    ["sim/l3_lane.py", "Layer 3", "Pymunk single-lane full physics; cam → follower → fork → pawl → rack"],
    ["sim/l3_visualise.py", "Layer 3", "Pygame real-time render of l3_lane; interactive speed slider"],
    ["sim/l3_detent_analysis.py", "Layer 3", "Spring force sweep; finds minimum detent preload for rack stability"],
    ["sim/run_all.py", "All layers", "Master runner: L1 → L2 assert → L3 spot-check; prints PASS/FAIL report"],
    ["sim/data/weights.npy", "Assets", "Loaded from quantized_weights_49x10.csv"],
    ["sim/data/bias.npy", "Assets", "Loaded from bias_steps_10.csv"],
    ["sim/data/ref_scores_20.npy", "Assets", "Generated by L1; consumed by L2 and L3 assertions"],
  ];

  const milestones = [
    ["M-S1", "Layer 1 complete", "1–2 days", "Reference classify verified on full MNIST test set. Accuracy printed. Score ranges exported.", C.green],
    ["M-S2", "Layer 2 reset verified", "2–3 days", "SimPy reset sequence runs; 9 cam events in correct order; all sub-system states correct after reset.", C.blue],
    ["M-S3", "Layer 2 full cycle", "3–4 days", "All 49 feature cycles complete; accumulator levels match L1 ref_scores for 20 test digits.", C.blue],
    ["M-S4", "Layer 2 fault injection", "1–2 days", "Double-advance and SOL-A misfire faults produce detectable divergence from ref_scores. Alarms logged.", C.blue],
    ["M-S5", "Layer 3 cam + follower", "2–3 days", "Pymunk cam lobe → follower verified: 5 height levels produce 0/1/2/3/4 mm tip travel.", C.amber],
    ["M-S6", "Layer 3 full lane", "4–5 days", "Full single-lane simulation: all 5 weight states produce correct accumulator rack delta.", C.amber],
    ["M-S7", "Layer 3 detent analysis", "1–2 days", "Minimum detent spring preload determined. Bilobed cam verified no double-advance at 2× speed.", C.amber],
    ["M-S8", "All layers integrated", "1 day", "run_all.py produces PASS on all assertions. Simulation package ready to hand off to CAD phase.", C.green],
  ];

  const tabs = [
    { id: "layers", label: "Simulation Layers" },
    { id: "files", label: "File Plan" },
    { id: "milestones", label: "Milestones" },
    { id: "tools", label: "Tool Comparison" },
  ];

  const toolComparison = [
    ["Tool", "Category", "Ratchet/Detent?", "Cam follower?", "Timing/seq?", "10k image batch?", "Install", "Verdict"],
    ["NumPy", "Math", "—", "—", "—", "✓ fast", "Standard", "Layer 1 oracle"],
    ["SimPy", "DES", "As events", "As timeout", "✓ native", "✓ fast", "pip", "Layer 2 sequencer"],
    ["Pymunk", "2D physics", "RatchetJoint ✓", "Cam body ✓", "Partial", "Slow", "pip", "Layer 3 kinematic"],
    ["PyDy", "Multibody dynamics", "Limited", "Via ODE", "—", "No", "pip", "Over-complex; skip"],
    ["MBDyn", "FEM multibody", "Yes", "Yes", "Yes", "No", "Compile", "C++ heavy; skip for now"],
    ["Pygame only", "Visualisation", "Manual", "Manual", "No", "No", "pip", "Renderer only; use with Pymunk"],
    ["Pysics", "2D holonomic", "No", "Limited", "No", "No", "GitHub", "Academic; incomplete; skip"],
    ["Box2D/PyBox2D", "2D physics", "No native", "No native", "No", "No", "pip", "No ratchet primitive; skip"],
  ];

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", fontFamily: "'DM Sans','Segoe UI',sans-serif",
      color: C.ink, padding: 0,
    }}>
      {/* Header */}
      <div style={{ background: C.blue, color: C.white, padding: "28px 32px 24px" }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, letterSpacing: 2, opacity: 0.7, marginBottom: 8 }}>
          MECHANICAL MNIST CLASSIFIER — END-TO-END SIMULATION PLAN
        </div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, letterSpacing: -0.5, lineHeight: 1.2 }}>
          3-Layer Digital Simulation Strategy
        </h1>
        <div style={{ marginTop: 10, fontSize: 13, opacity: 0.85, maxWidth: 680 }}>
          A layered approach: logical oracle → discrete-event machine model → kinematic contact physics.
          Each layer validates the previous. All three must pass before CAD fabrication begins.
        </div>
      </div>

      <div style={{ padding: "24px 28px", maxWidth: 1000, margin: "0 auto" }}>

        <InfoBox
          label="CORE MANDATE"
          text="Simulation must be end-to-end: from a raw digit drawing through 49 mechanical feature cycles to a predicted rack height — matching the physical machine behaviour including sequencing order, reset correctness, pulse counting accuracy, and accumulator direction/magnitude selection."
          color={C.blue}
          bg={C.blueFill}
        />

        <div style={{ marginBottom: 20 }}>
          <div style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: C.grey, marginBottom: 10, letterSpacing: 0.5 }}>
            SIMULATION PIPELINE
          </div>
          <Pipeline />
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: `2px solid ${C.rule}` }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
              background: activeTab === t.id ? C.blue : "transparent",
              color: activeTab === t.id ? C.white : C.grey,
              border: "none", borderRadius: "4px 4px 0 0", cursor: "pointer",
              padding: "8px 18px", fontWeight: 700, fontSize: 13,
              fontFamily: "'DM Sans','Segoe UI',sans-serif",
              borderBottom: activeTab === t.id ? `2px solid ${C.blue}` : "none",
              marginBottom: -2,
            }}>{t.label}</button>
          ))}
        </div>

        {activeTab === "layers" && (
          <div>
            <SectionLabel n="1" title="Simulation Layer Details" />
            <p style={{ fontSize: 13, color: C.grey, marginBottom: 20, lineHeight: 1.6 }}>
              Three layers, each independently runnable and assertable. Later layers consume outputs (reference scores, timing data)
              from earlier ones. Click each layer to expand.
            </p>
            {layers.map((l, i) => (
              <LayerCard key={i} {...l} expanded={!!expanded[i]} onToggle={() => toggle(i)} />
            ))}

            <Card accent={C.amber}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8, color: C.amber }}>
                ⚠ What NOT to simulate at this stage
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[
                  ["3D shaft twist / alignment", "FreeCAD FEM or Abaqus; out of scope now"],
                  ["Thermal expansion / material wear", "Irrelevant at prototype scale"],
                  ["Electromagnetic solenoid dynamics", "Datasheet spec is sufficient"],
                  ["Full 10-lane Pymunk physics", "Too slow; one lane is the representative proof"],
                  ["Fluid / lubrication effects", "Not applicable to dry ratchet mechanism"],
                  ["Monte Carlo noise in cam heights", "Tolerance analysis is a CAD/GD&T task"],
                ].map(([what, why], i) => (
                  <div key={i} style={{ fontSize: 12, display: "flex", gap: 8 }}>
                    <span style={{ color: C.amber, fontWeight: 700 }}>✕</span>
                    <span><strong>{what}</strong> — {why}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {activeTab === "files" && (
          <div>
            <SectionLabel n="2" title="Simulation File Plan" />
            <InfoBox
              label="STRUCTURE"
              text="All simulation code lives in sim/. The project root already has the model export folder. sim/ is self-contained; it only reads from ../mechanical_mnist_count_pooled_export_5level/ and writes to sim/data/."
              color={C.blue}
              bg={C.blueFill}
            />
            <Table
              headers={["File", "Layer", "Purpose"]}
              rows={fileplan}
            />
            <Card accent={C.green}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 10, color: C.green }}>
                Dependency graph
              </div>
              <pre style={{
                fontFamily: "monospace", fontSize: 12, margin: 0,
                color: C.ink, lineHeight: 1.7, background: "transparent",
              }}>{`weights.npy  bias.npy (from model export CSVs)
       │
  l1_reference.py
       │ exports
  ref_scores_20.npy
       │
  l2_machine.py ──── l2_timing.py
       │ asserts
  ref_scores_20.npy  (same oracle)
       │
  l3_lane.py ──── l3_visualise.py ──── l3_detent_analysis.py
       │
  run_all.py  (imports and asserts all three layers)`}
              </pre>
            </Card>

            <SectionLabel n="3" title="Key Library Dependencies" />
            <Table
              headers={["Package", "Version pin", "Purpose", "Install"]}
              rows={[
                ["numpy", "≥ 1.26", "All matrix arithmetic; L1 oracle; data I/O", "already installed"],
                ["pandas", "≥ 2.0", "Load CSV model exports", "pip install pandas"],
                ["simpy", "≥ 4.1", "Process-based DES for L2", "pip install simpy"],
                ["pymunk", "≥ 7.0", "Chipmunk2D wrapper; RatchetJoint, GearJoint etc.", "pip install pymunk"],
                ["pygame", "≥ 2.5", "Real-time 2D render for L3 kinematic", "pip install pygame"],
                ["matplotlib", "≥ 3.8", "Score histograms, Gantt chart, analysis plots", "already installed"],
                ["scikit-learn", "≥ 1.4", "MNIST dataset loader (fetch_openml)", "pip install scikit-learn"],
                ["pytest", "≥ 8.0", "Assertion framework for run_all.py", "pip install pytest"],
              ]}
            />
          </div>
        )}

        {activeTab === "milestones" && (
          <div>
            <SectionLabel n="4" title="Simulation Milestones" />
            <InfoBox
              label="TOTAL ESTIMATE"
              text="~16–22 developer-days for a solo developer familiar with Python and basic mechanics. Layers 1 and 2 can proceed immediately. Layer 3 requires Pymunk familiarisation (1 day) but can begin in parallel with L2."
              color={C.green}
              bg={C.greenFill}
            />
            <Table
              headers={["ID", "Milestone", "Effort", "Pass criterion", ""]}
              rows={milestones.map(([id, name, effort, crit]) => [id, name, effort, crit, ""])}
            />
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
              {[
                [C.green, "Layer 1 — Logic"],
                [C.blue, "Layer 2 — Events"],
                [C.amber, "Layer 3 — Physics"],
              ].map(([color, label]) => (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                  <div style={{ width: 12, height: 12, borderRadius: 2, background: color }} />
                  <span>{label}</span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel n="5" title="Critical Path" />
              <div style={{ fontFamily: "monospace", fontSize: 12, lineHeight: 2, color: C.ink }}>
                {[
                  ["M-S1", "L1 reference", "→ unlocks L2 (ref_scores oracle needed)", C.green],
                  ["M-S2", "L2 reset", "→ validates Section 6 cam ordering", C.blue],
                  ["M-S3", "L2 full cycle", "→ confirms score correctness end-to-end", C.blue],
                  ["M-S5", "L3 cam+follower", "→ validates cam height tolerances for fabrication", C.amber],
                  ["M-S6", "L3 full lane", "→ confirms mechanism geometry; unlocks CAD phase", C.amber],
                  ["M-S8", "All integrated", "→ SIMULATION COMPLETE; hand off to CAD + build", C.green],
                ].map(([id, name, note, color]) => (
                  <div key={id} style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
                    <span style={{ color, fontWeight: 800, minWidth: 48 }}>{id}</span>
                    <span style={{ minWidth: 140 }}>{name}</span>
                    <span style={{ color: C.grey, fontSize: 11 }}>{note}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === "tools" && (
          <div>
            <SectionLabel n="6" title="Tool Comparison & Rationale" />
            <InfoBox
              label="DECISION"
              text="NumPy (L1) + SimPy (L2) + Pymunk/Pygame (L3) is the optimal stack. It is entirely pip-installable, Python-native, open-source, and maps cleanly to the three conceptually distinct simulation concerns: arithmetic correctness, event sequencing, and contact physics."
              color={C.blue}
              bg={C.blueFill}
            />
            <Table headers={toolComparison[0]} rows={toolComparison.slice(1)} />

            <Card accent={C.green} style={{ marginTop: 20 }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 12, color: C.green }}>
                Why Pymunk specifically for Layer 3
              </div>
              {[
                ["RatchetJoint built-in", "The RatchetJoint constraint in Pymunk directly models a rotary ratchet — exactly the accumulator mechanism. No custom contact detection needed."],
                ["GearJoint for direction reversal", "GearJoint(a, b, phase, ratio=-1) exactly models the idler-gear SUB path. ratio=+1 for ADD. Toggle is a constraint add/remove."],
                ["SimpleMotor for cam shaft", "SimpleMotor drives the main cam shaft at controlled rate_of_rotation — one full revolution per clock pulse is trivially implemented."],
                ["DampedRotarySpring for detent", "The ball-detent spring behaviour is well approximated by DampedRotarySpring at each indexed position. Spring constant maps directly to physical spring preload."],
                ["GrooveJoint for selector fork", "The 5-position selector fork slides in a groove — exactly GrooveJoint + RotaryLimitJoint at 5 stop positions."],
                ["Pygame integration", "pymunk.pygame_util.DrawOptions gives real-time 2D render of the entire lane with zero additional code."],
              ].map(([title, desc], i) => (
                <div key={i} style={{
                  display: "flex", gap: 12, padding: "8px 0",
                  borderBottom: i < 5 ? `1px solid ${C.rule}` : "none",
                }}>
                  <span style={{ fontFamily: "monospace", color: C.green, fontWeight: 700, minWidth: 28, fontSize: 13 }}>✓</span>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12 }}>{title}</div>
                    <div style={{ fontSize: 12, color: C.grey, lineHeight: 1.5 }}>{desc}</div>
                  </div>
                </div>
              ))}
            </Card>

            <Card accent={C.amber}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 12, color: C.amber }}>
                Why NOT a full physics engine (MBDyn, Simbody, etc.) for this project
              </div>
              <p style={{ fontSize: 13, lineHeight: 1.6, margin: 0 }}>
                MBDyn and Simbody are powerful but require compiling from source, writing model files in domain-specific languages,
                and have steep learning curves. For this project's mechanism scale (discrete-indexed ratchets, 5-height cams, linear racks),
                Pymunk's constraint primitives are <em>exactly sufficient</em> and take days not weeks to set up.
                The machine does not have continuous flexible bodies, fluid coupling, or complex 3D kinematics — the 2D approximation
                of Pymunk covers 95% of the mechanical risk surface. If later simulation reveals 3D shaft effects (twist, axial load),
                that can be addressed with a targeted FreeCAD FEM analysis, not a whole-machine re-simulation.
              </p>
            </Card>
          </div>
        )}

        <Rule />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: C.grey }}>
          <span style={{ fontFamily: "monospace" }}>
            MECHANICAL MNIST DIGIT CLASSIFIER — SIM PLAN v1.0
          </span>
          <span>
            L1: NumPy &nbsp;|&nbsp; L2: SimPy &nbsp;|&nbsp; L3: Pymunk + Pygame
          </span>
        </div>
      </div>
    </div>
  );
}
