import { useState, useEffect, useRef, useCallback } from 'react'
import './DrugForm.css'

function useDrugList() {
  const [drugs, setDrugs] = useState([])
  useEffect(() => {
    fetch('/drugs.json')
      .then((r) => r.json())
      .then(setDrugs)
      .catch(() => {})
  }, [])
  return drugs
}

function DrugInput({ id, label, placeholder, value, onChange, disabled, drugs }) {
  const [suggestions, setSuggestions] = useState([])
  const [open, setOpen] = useState(false)
  const [highlighted, setHighlighted] = useState(-1)
  const wrapRef = useRef(null)

  const getSuggestions = useCallback(
    (text) => {
      if (!text || text.length < 2) return []
      const lower = text.toLowerCase()
      const starts = []
      const contains = []
      for (const d of drugs) {
        const dl = d.toLowerCase()
        if (dl.startsWith(lower)) starts.push(d)
        else if (dl.includes(lower)) contains.push(d)
        if (starts.length >= 8) break
      }
      return [...starts, ...contains].slice(0, 8)
    },
    [drugs]
  )

  function handleChange(e) {
    const v = e.target.value
    onChange(v)
    const s = getSuggestions(v)
    setSuggestions(s)
    setOpen(s.length > 0)
    setHighlighted(-1)
  }

  function pick(name) {
    onChange(name)
    setOpen(false)
    setSuggestions([])
  }

  function handleKeyDown(e) {
    if (!open) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlighted((h) => Math.min(h + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlighted((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter' && highlighted >= 0) {
      e.preventDefault()
      pick(suggestions[highlighted])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  useEffect(() => {
    function onDoc(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  return (
    <div className="input-group" ref={wrapRef}>
      <label htmlFor={id} className="input-label">{label}</label>
      <div className="input-wrap">
        <input
          id={id}
          type="text"
          className="drug-input"
          placeholder={placeholder}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            const s = getSuggestions(value)
            if (s.length) { setSuggestions(s); setOpen(true) }
          }}
          disabled={disabled}
          autoComplete="off"
          spellCheck="false"
        />
        {open && (
          <ul className="suggestions">
            {suggestions.map((s, i) => (
              <li
                key={s}
                className={`suggestion-item ${i === highlighted ? 'highlighted' : ''}`}
                onMouseDown={() => pick(s)}
                onMouseEnter={() => setHighlighted(i)}
              >
                {s}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

export default function DrugForm({ onSearch, loading }) {
  const [drug1, setDrug1] = useState('')
  const [drug2, setDrug2] = useState('')
  const drugs = useDrugList()

  function handleSubmit(e) {
    e.preventDefault()
    const d1 = drug1.trim()
    const d2 = drug2.trim()
    if (!d1 || !d2) return
    onSearch(d1, d2)
  }

  return (
    <form className="drug-form" onSubmit={handleSubmit}>
      <h1 className="form-title">Drug Interaction Checker</h1>
      <p className="form-subtitle">
        Enter two medications to find out what happens when they are taken together
        and what alternatives exist.
      </p>

      <div className="inputs-row">
        <DrugInput
          id="drug1"
          label="First Drug"
          placeholder="e.g. Warfarin"
          value={drug1}
          onChange={setDrug1}
          disabled={loading}
          drugs={drugs}
        />

        <div className="plus-divider">+</div>

        <DrugInput
          id="drug2"
          label="Second Drug"
          placeholder="e.g. Aspirin"
          value={drug2}
          onChange={setDrug2}
          disabled={loading}
          drugs={drugs}
        />
      </div>

      <button
        type="submit"
        className={`check-btn ${loading ? 'loading' : ''}`}
        disabled={loading || !drug1.trim() || !drug2.trim()}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Analysing…
          </>
        ) : (
          <>
            <span>🔍</span>
            Check Interaction
          </>
        )}
      </button>
    </form>
  )
}
