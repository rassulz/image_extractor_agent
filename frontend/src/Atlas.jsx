import { useEffect, useState } from 'react'

// Neon palette override so points glow on the dark canvas
const NEON = ['#8B5CF6', '#22D3EE', '#FF2E97', '#B6FF3B', '#FBBF24', '#34D399', '#F472B6', '#38BDF8']

export default function Atlas() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [hover, setHover] = useState(null)
  const [active, setActive] = useState(null)

  useEffect(() => {
    fetch('/api/atlas?k=6')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('atlas not ready'))))
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  if (error)
    return <p className="py-20 text-center text-sm" style={{ color: 'var(--text-mute)' }}>Atlas unavailable: {error}</p>
  if (!data)
    return (
      <div className="flex flex-col items-center gap-3 py-24">
        <i className="ti ti-loader-2 animate-spin text-2xl" style={{ color: 'var(--violet)' }} />
        <p className="text-sm" style={{ color: 'var(--text-dim)' }}>Projecting 512-D CLIP space…</p>
      </div>
    )

  const colorOf = (c) => NEON[c % NEON.length]
  const shown = active == null ? data.points : data.points.filter((p) => p.cluster === active)

  return (
    <div>
      <div className="mb-3 text-center">
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          The <span className="neon-text">semantic map</span>
        </h2>
        <p className="mx-auto mt-2 max-w-lg text-sm" style={{ color: 'var(--text-dim)' }}>
          Every photo projected from 512-D CLIP space to 2D (PCA). Clusters & labels are
          discovered automatically (KMeans + CLIP). Hover any point to preview.
        </p>
      </div>

      <div className="mb-4 flex flex-wrap items-center justify-center gap-2">
        <button onClick={() => setActive(null)}
          className="glass rounded-full px-3 py-1.5 text-xs font-medium transition"
          style={active == null ? { color: '#fff', boxShadow: '0 0 0 1px var(--glass-border)' } : { color: 'var(--text-mute)' }}>
          all · {data.total}
        </button>
        {data.clusters.map((cl) => {
          const col = colorOf(cl.id)
          const on = active === cl.id
          return (
            <button key={cl.id} onClick={() => setActive(on ? null : cl.id)}
              className="glass flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition hover:scale-105"
              style={{ color: on ? '#fff' : 'var(--text-dim)', boxShadow: on ? `0 0 0 1px ${col}, 0 0 20px -6px ${col}` : 'none' }}>
              <span className="h-2 w-2 rounded-full" style={{ background: col, boxShadow: `0 0 8px ${col}` }} />
              {cl.label} · {cl.count}
            </button>
          )
        })}
      </div>

      <div className="glass relative w-full overflow-hidden rounded-3xl"
        style={{ aspectRatio: '16 / 9', background: 'radial-gradient(circle at 50% 40%, rgba(139,92,246,0.06), transparent 70%), rgba(5,6,9,0.5)' }}>
        {/* cluster labels */}
        {data.clusters.map((cl) => (
          <span key={cl.id} className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 text-xs font-semibold uppercase tracking-wider"
            style={{ left: `${cl.cx * 100}%`, top: `${(1 - cl.cy) * 100}%`, color: colorOf(cl.id), textShadow: `0 0 12px ${colorOf(cl.id)}` }}>
            {cl.label}
          </span>
        ))}
        {/* points */}
        {shown.map((p) => {
          const col = colorOf(p.cluster)
          return (
            <span key={p.image_id}
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover((h) => (h?.image_id === p.image_id ? null : h))}
              className="absolute cursor-pointer rounded-full transition-transform hover:scale-[3]"
              style={{ left: `${p.x * 100}%`, top: `${(1 - p.y) * 100}%`, width: 5, height: 5, marginLeft: -2.5, marginTop: -2.5,
                background: col, boxShadow: `0 0 6px ${col}`, opacity: 0.85 }} />
          )
        })}
        {/* hover preview */}
        {hover && (
          <div className="pointer-events-none absolute z-10 overflow-hidden rounded-xl"
            style={{ left: `min(${hover.x * 100}%, calc(100% - 140px))`, top: `max(${(1 - hover.y) * 100}%, 8px)`,
              border: `2px solid ${colorOf(hover.cluster)}`, boxShadow: `0 0 24px -4px ${colorOf(hover.cluster)}` }}>
            <img src={hover.url} alt="" className="h-32 w-32 object-cover" />
          </div>
        )}
      </div>
    </div>
  )
}
