import { useEffect, useRef, useState } from 'react'

// Ambient neon field + film grain + cursor spotlight
export function Backdrop() {
  const dot = useRef(null)
  useEffect(() => {
    const move = (e) => {
      if (dot.current) {
        dot.current.style.left = e.clientX + 'px'
        dot.current.style.top = e.clientY + 'px'
      }
    }
    window.addEventListener('pointermove', move)
    return () => window.removeEventListener('pointermove', move)
  }, [])
  return (
    <>
      <div className="aurora"><div className="aurora-3" /></div>
      <div className="grain" />
      <div ref={dot} className="spotlight" />
    </>
  )
}

// Circular conic match gauge
export function MatchRing({ value, size = 46, stroke = 4, color = '#8B5CF6' }) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const off = c * (1 - value / 100)
  return (
    <svg width={size} height={size} className="-rotate-90" style={{ filter: `drop-shadow(0 0 5px ${color}88)` }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
        strokeLinecap="round" strokeDasharray={c} strokeDashoffset={off}
        style={{ transition: 'stroke-dashoffset .8s cubic-bezier(.2,.7,.2,1)' }}
      />
      <text x="50%" y="50%" dominantBaseline="central" textAnchor="middle"
        className="rotate-90" style={{ fontSize: 11, fontWeight: 700, fill: '#EDEFF4', transformOrigin: 'center' }}>
        {value}
      </text>
    </svg>
  )
}

export function Stat({ icon, label, value, accent }) {
  return (
    <div className="glass flex items-center gap-3 rounded-2xl px-4 py-3">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl text-lg"
        style={{ background: `${accent}1a`, color: accent, boxShadow: `0 0 18px -6px ${accent}` }}>
        <i className={`ti ${icon}`} />
      </span>
      <div>
        <div className="text-lg font-semibold leading-none">{value}</div>
        <div className="mt-1 text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-mute)' }}>{label}</div>
      </div>
    </div>
  )
}

export function Skeletons({ n = 6 }) {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="overflow-hidden rounded-3xl" style={{ border: '1px solid var(--glass-border)' }}>
          <div className="skeleton aspect-[4/3]" />
          <div className="space-y-2 p-4">
            <div className="skeleton h-3 w-2/3 rounded-full" />
            <div className="skeleton h-3 w-1/2 rounded-full" />
          </div>
        </div>
      ))}
    </div>
  )
}

// Animated count-up number
export function CountUp({ to = 0, dur = 900 }) {
  const [n, setN] = useState(0)
  useEffect(() => {
    let raf, start
    const step = (t) => {
      if (!start) start = t
      const p = Math.min(1, (t - start) / dur)
      setN(Math.round((1 - Math.pow(1 - p, 3)) * to))
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [to, dur])
  return <>{n.toLocaleString()}</>
}
