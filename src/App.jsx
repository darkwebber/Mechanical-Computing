import { useState } from "react";
import SimPlan from "./SimPlan";
import SimDemo from "./SimDemo";

const C = { blue: "#1B4F8A", white: "#FFFFFF", bg: "#F5F2EB", grey: "#6B6860" };

export default function App() {
  const [tab, setTab] = useState("demo");

  return (
    <div style={{ background: C.bg, minHeight: "100vh" }}>
      {/* Top nav */}
      <div style={{ background: C.blue, display: "flex", gap: 0, padding: "0 24px" }}>
        {[
          { id: "demo", label: "▶  Live Simulation" },
          { id: "plan", label: "   Simulation Plan" },
        ].map(({ id, label }) => (
          <button key={id} onClick={() => setTab(id)}
            style={{
              background: tab === id ? "rgba(255,255,255,0.15)" : "transparent",
              color: C.white,
              border: "none",
              borderBottom: tab === id ? "3px solid #FFF" : "3px solid transparent",
              padding: "12px 22px",
              cursor: "pointer",
              fontWeight: tab === id ? 700 : 400,
              fontSize: 14,
              fontFamily: "'DM Sans','Segoe UI',sans-serif",
            }}>
            {label}
          </button>
        ))}
      </div>

      {tab === "demo" ? <SimDemo /> : <SimPlan />}
    </div>
  );
}
