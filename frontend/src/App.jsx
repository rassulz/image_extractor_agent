import { useEffect, useRef, useState } from 'react'
import Atlas from './Atlas.jsx'
import { Backdrop, MatchRing, Stat, Skeletons, CountUp } from './ui.jsx'

const CHIPS = [
  { text: 'dog running on grass', icon: 'ti-dog', c: '#B6FF3B' },
  { text: 'people standing near water', icon: 'ti-ripple', c: '#22D3EE' },
  { text: 'a child playing outdoors', icon: 'ti-mood-kid', c: '#FF2E97' },
  { text: 'a person riding a bicycle', icon: 'ti-bike', c: '#8B5CF6' },
  { text: 'city street with people', icon: 'ti-building-skyscraper', c: '#FBBF24' },
  { text: 'a boat on the water', icon: 'ti-sailboat', c: '#34D399' },
]

const toPct = (s) => Math.round(Math.max(2, Math.min(99, ((s - 0.14) / 0.18) * 100)))
const ringColor = (p) => (p >= 80 ? '#22D3EE' : p >= 55 ? '#8B5CF6' : '#FF2E97')

function ResultCard({ r, i }) {
  const pct = toPct(r.similarity_score)
  const col = ringColor(pct)
  const hasCaption = r.generated_caption && r.generated_caption !== 'No caption generated.'
  return (
    <div
      className="group glass rise relative flex flex-col overflow-hidden rounded-3xl transition duration-300 hover:-translate-y-1"
      style={{ animationDelay: `${i * 55}ms` }}
      onMouseEnter={(e) => (e.currentTarget.style.boxShadow = `0 0 0 1px ${col}55, 0 20px 60px -20px ${col}`)}
      onMouseLeave={(e) => (e.currentTarget.style.boxShadow = 'none')}
    >
      <div className="relative aspect-[4/3] overflow-hidden">
        <img src={r.url} alt={r.generated_caption || r.file_name} loading="lazy"
          className="h-full w-full object-cover transition duration-700 group-hover:scale-110" />
        <div className="pointer-events-none absolute inset-0"
          style={{ background: 'linear-gradient(180deg, rgba(5,6,9,0) 45%, rgba(5,6,9,0.85))' }} />
        <span className="absolute left-3 top-3 flex h-7 items-center gap-1 rounded-full px-2.5 text-xs font-bold text-white"
          style={{ background: 'rgba(5,6,9,0.6)', border: '1px solid var(--glass-border)' }}>
          #{r.rank}
        </span>
        <div className="absolute right-2.5 top-2.5"><MatchRing value={pct} color={col} /></div>
        <div className="absolute inset-x-3 bottom-3 flex items-center gap-2 text-[11px]">
          <span className="flex items-center gap-1 rounded-full px-2 py-1 font-medium text-white"
            style={{ background: 'rgba(255,255,255,0.12)', backdropFilter: 'blur(6px)' }}>
            <i className="ti ti-calendar" style={{ color: col }} /> {r.date}
          </span>
        </div>
      </div>
      <div className="flex flex-1 flex-col gap-2.5 p-4">
        {hasCaption && <p className="text-sm font-medium leading-snug">{r.generated_caption}</p>}
        {r.tags?.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {r.tags.slice(0, 6).map((t, k) => (
              <span key={k} className="rounded-full px-2 py-0.5 text-[11px]"
                style={{ background: 'rgba(139,92,246,0.12)', color: '#C4B5FD' }}>{t}</span>
            ))}
          </div>
        )}
        <div className="mt-auto flex items-center justify-between pt-1 text-[11px]" style={{ color: 'var(--text-mute)' }}>
          <span className="flex items-center gap-1"><i className="ti ti-dimensions" /> {r.width}×{r.height}</span>
          <span className="flex items-center gap-1 font-mono" style={{ color: col }}>
            <i className="ti ti-vector-triangle" /> {r.similarity_score.toFixed(3)}
          </span>
        </div>
        <p className="rounded-2xl px-3 py-2 text-[11px] leading-relaxed"
          style={{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-dim)', border: '1px solid var(--glass-border)' }}>
          {r.explanation}
        </p>
      </div>
    </div>
  )
}

function SearchView({ health }) {
  const [query, setQuery] = useState('dog running on grass')
  const [topK, setTopK] = useState(6)
  const [loading, setLoading] = useState(false)
  const [answer, setAnswer] = useState(null)
  const [samples, setSamples] = useState([])
  const [error, setError] = useState(null)
  const [ms, setMs] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => { fetch('/api/samples?n=12').then((r) => r.json()).then(setSamples).catch(() => {}) }, [])

  async function runSearch(q = query) {
    if (!q.trim()) return
    setLoading(true); setError(null)
    const t0 = performance.now()
    try {
      const res = await fetch('/api/search', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: topK }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Search failed')
      setAnswer(await res.json()); setMs(Math.round(performance.now() - t0))
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <div>
      {/* hero */}
      <div className="mx-auto max-w-3xl pt-10 pb-7 text-center">
        <span className="glass mb-5 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs" style={{ color: 'var(--text-dim)' }}>
          <span className="h-1.5 w-1.5 rounded-full pulse-dot" style={{ background: 'var(--cyan)' }} />
          CLIP ViT-B/32 · fully local retrieval
        </span>
        <h1 className="text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
          Search photos in <span className="neon-text">plain language</span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-sm sm:text-base" style={{ color: 'var(--text-dim)' }}>
          Describe what you remember. The agent embeds every image with CLIP and finds
          the closest matches in a local vector index — no captions, no cloud.
        </p>
      </div>

      {/* search bar */}
      <div className="grad-border glass mx-auto max-w-3xl rounded-2xl">
        <div className="flex items-center gap-2 p-2.5">
          <i className="ti ti-search pl-2 text-lg" style={{ color: 'var(--text-mute)' }} />
          <input
            ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runSearch()}
            placeholder="Describe the photo you're looking for…"
            autoFocus spellCheck={false} autoComplete="off"
            className="relative z-10 flex-1 bg-transparent text-sm outline-none placeholder:text-[color:var(--text-mute)]"
            style={{ caretColor: 'var(--violet)' }}
          />
          <button onClick={() => runSearch()} disabled={loading}
            className="flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold text-white transition active:scale-95 disabled:opacity-60"
            style={{ background: 'linear-gradient(120deg, var(--violet), var(--indigo))', boxShadow: '0 8px 30px -8px var(--violet)' }}>
            {loading ? <i className="ti ti-loader-2 animate-spin" /> : <i className="ti ti-sparkles" />} Search
          </button>
        </div>
      </div>

      {/* top-k + chips */}
      <div className="mx-auto mt-4 flex max-w-3xl flex-wrap items-center justify-center gap-2">
        <span className="text-xs" style={{ color: 'var(--text-mute)' }}>show</span>
        <input type="range" min="3" max="12" value={topK} onChange={(e) => setTopK(Number(e.target.value))}
          className="w-28" style={{ accentColor: 'var(--violet)' }} />
        <span className="w-5 text-center text-xs font-semibold">{topK}</span>
      </div>
      <div className="mx-auto mt-3 flex max-w-3xl flex-wrap justify-center gap-2">
        {CHIPS.map((c) => (
          <button key={c.text} onClick={() => { setQuery(c.text); runSearch(c.text) }}
            className="glass flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition hover:scale-105"
            style={{ color: 'var(--text-dim)' }}>
            <i className={`ti ${c.icon}`} style={{ color: c.c }} /> {c.text}
          </button>
        ))}
      </div>

      {/* stats */}
      {health?.status === 'ready' && (
        <div className="mx-auto mt-8 grid max-w-3xl grid-cols-3 gap-3">
          <Stat icon="ti-photo" label="Images indexed" value={<CountUp to={health.num_images} />} accent="#8B5CF6" />
          <Stat icon="ti-vector-bezier" label="Vector dim" value="512" accent="#22D3EE" />
          <Stat icon="ti-bolt" label="Query latency" value={ms != null ? `${ms} ms` : '~40 ms'} accent="#FF2E97" />
        </div>
      )}

      {error && (
        <div className="mx-auto mt-6 max-w-3xl rounded-2xl px-4 py-3 text-sm"
          style={{ background: 'rgba(255,46,151,0.1)', border: '1px solid rgba(255,46,151,0.3)', color: '#FCA5C5' }}>{error}</div>
      )}

      {/* results */}
      <div className="mt-10">
        {loading && <Skeletons n={topK} />}
        {!loading && answer && (
          <>
            <div className="glass mb-5 flex flex-wrap items-center justify-between gap-2 rounded-2xl px-4 py-3 text-xs">
              <span style={{ color: 'var(--text-dim)' }}>
                <i className="ti ti-bulb mr-1.5" style={{ color: 'var(--violet)' }} />{answer.interpreted_intent}
              </span>
              <span className="flex items-center gap-3 font-mono" style={{ color: 'var(--text-mute)' }}>
                {answer.filters.date && (
                  <span style={{ color: 'var(--cyan)' }}>filter: date {answer.filters.date}</span>
                )}
                {ms != null && <span><i className="ti ti-bolt" /> {ms} ms</span>}
              </span>
            </div>
            {answer.results?.length === 0 ? (
              <p className="py-16 text-center text-sm" style={{ color: 'var(--text-mute)' }}>{answer.message || 'No matches.'}</p>
            ) : (
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
                {answer.results.map((r, i) => <ResultCard key={r.image_id} r={r} i={i} />)}
              </div>
            )}
          </>
        )}
        {!loading && !answer && samples.length > 0 && (
          <div>
            <p className="mb-3 text-xs uppercase tracking-widest" style={{ color: 'var(--text-mute)' }}>Inside the index</p>
            <div className="columns-2 gap-3 sm:columns-3 lg:columns-4 [&>*]:mb-3">
              {samples.map((s) => (
                <div key={s.image_id} className="glass overflow-hidden rounded-2xl">
                  <img src={s.url} alt="" loading="lazy" className="w-full object-cover" />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const TABS = [
  { id: 'search', label: 'Search', icon: 'ti-search' },
  { id: 'atlas', label: 'Atlas', icon: 'ti-atom-2' },
]

export default function App() {
  const [tab, setTab] = useState('search')
  const [health, setHealth] = useState(null)
  useEffect(() => { fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => {}) }, [])

  return (
    <div className="relative min-h-full">
      <Backdrop />
      <div className="relative z-10 pb-28">
        {/* nav */}
        <header className="sticky top-0 z-30">
          <div className="glass mx-auto mt-4 flex max-w-6xl items-center justify-between gap-3 rounded-2xl px-4 py-2.5"
            style={{ background: 'rgba(10,12,18,0.6)' }}>
            <div className="flex items-center gap-2.5">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl text-white"
                style={{ background: 'linear-gradient(135deg, var(--violet), var(--cyan))', boxShadow: '0 0 24px -6px var(--violet)' }}>
                <i className="ti ti-photo-search text-lg" />
              </span>
              <div>
                <p className="text-sm font-semibold leading-tight">Image Extractor Agent</p>
                <p className="text-[10px]" style={{ color: 'var(--text-mute)' }}>semantic vision search</p>
              </div>
            </div>

            {/* segmented tabs */}
            <div className="glass flex items-center gap-1 rounded-xl p-1" style={{ background: 'rgba(255,255,255,0.03)' }}>
              {TABS.map((t) => {
                const on = tab === t.id
                return (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition"
                    style={on
                      ? { background: 'linear-gradient(120deg, var(--violet), var(--indigo))', color: '#fff', boxShadow: '0 6px 20px -8px var(--violet)' }
                      : { color: 'var(--text-dim)' }}>
                    <i className={`ti ${t.icon}`} /> {t.label}
                  </button>
                )
              })}
            </div>

            {health?.status === 'ready' && (
              <span className="hidden items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium sm:flex"
                style={{ background: 'rgba(34,211,238,0.1)', color: '#7DD3FC', border: '1px solid rgba(34,211,238,0.25)' }}>
                <span className="h-1.5 w-1.5 rounded-full pulse-dot" style={{ background: 'var(--cyan)' }} />
                {health.num_images.toLocaleString()} vectors
              </span>
            )}
          </div>
        </header>

        <main className="mx-auto max-w-6xl px-5">
          {tab === 'search' ? <SearchView health={health} /> : <div className="pt-8"><Atlas /></div>}
        </main>

        <footer className="mx-auto mt-16 max-w-6xl px-5 text-center text-[11px]" style={{ color: 'var(--text-mute)' }}>
          100% local CLIP image↔text retrieval · captions.txt never used for indexing · dates are synthetic · YDL 2026 · Day 5
        </footer>
      </div>
    </div>
  )
}
