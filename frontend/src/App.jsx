import { useState } from 'react'
import DrugForm from './components/DrugForm'
import ResultCard from './components/ResultCard'
import './App.css'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSearch(drug1, drug2) {
    setLoading(true)
    setError(null)
    setResult(null)

    const query = `What happens if I take ${drug1} and ${drug2} together?`

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `Server error ${res.status}`)
      }

      const data = await res.json()
      setResult({ ...data, drug1, drug2 })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⚕</span>
            <span className="logo-text">DrugCheck<span className="logo-accent">AI</span></span>
          </div>
          <p className="tagline">Powered by DrugBank · FDA · PubMed knowledge base</p>
        </div>
      </header>

      <main className="main">
        <DrugForm onSearch={handleSearch} loading={loading} />

        {error && (
          <div className="error-box">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {result && <ResultCard result={result} />}
      </main>

      <footer className="footer">
        <p>For informational purposes only. Always consult a licensed healthcare professional.</p>
      </footer>
    </div>
  )
}
