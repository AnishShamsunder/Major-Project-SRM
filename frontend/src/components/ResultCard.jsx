import './ResultCard.css'

function parseAnswer(answer) {
  // Split into the two named sections
  const interactionMatch = answer.match(
    /\*\*What happens when both are taken together:\*\*\s*([\s\S]*?)(?=\*\*What can be done instead:|$)/i
  )
  const alternativesMatch = answer.match(
    /\*\*What can be done instead:\*\*\s*([\s\S]*?)(?=⚠|$)/i
  )
  const disclaimer = answer.includes('⚠') ? '⚠️ Consult a licensed healthcare professional before making any medication changes.' : null

  const bullets = (block) =>
    block
      ? block
          .split('\n')
          .map((l) => l.replace(/^[-•*]\s*/, '').trim())
          .filter(Boolean)
      : []

  return {
    interaction: bullets(interactionMatch?.[1]),
    alternatives: bullets(alternativesMatch?.[1]),
    disclaimer,
    raw: answer,
  }
}

export default function ResultCard({ result }) {
  const { drug1, drug2, answer, model, sources } = result
  const parsed = parseAnswer(answer)

  const hasSections = parsed.interaction.length > 0 || parsed.body.length > 0 || parsed.alternatives.length > 0

  return (
    <div className="result-card">
      {/* Title */}
      <div className="result-header">
        <div className="drug-pair">
          <span className="drug-tag">{drug1}</span>
          <span className="pair-plus">+</span>
          <span className="drug-tag">{drug2}</span>
        </div>
        <span className="model-badge">{model}</span>
      </div>

      {hasSections ? (
        <>
          {/* Section 1 */}
          <Section
            icon="⚡"
            title="What happens when both are taken together"
            items={parsed.interaction}
            accent="danger"
          />

          {/* Section 2 */}
          <Section
            icon="🪧"
            title="What happens in the body"
            items={parsed.body}
            accent="body"
          />

          {/* Section 3 */}
          <Section
            icon="✅"
            title="What can be done instead"
            items={parsed.alternatives}
            accent="safe"
          />

          {/* Disclaimer */}
          {parsed.disclaimer && (
            <div className="disclaimer">{parsed.disclaimer}</div>
          )}
        </>
      ) : (
        <div className="raw-answer">{answer}</div>
      )}

      {/* Sources */}
      {sources?.length > 0 && (
        <details className="sources-details">
          <summary className="sources-summary">
            📄 {sources.length} source chunk{sources.length !== 1 ? 's' : ''} retrieved
          </summary>
          <div className="sources-list">
            {sources.map((s, i) => (
              <div key={s.id} className="source-item">
                <div className="source-meta">
                  <span className="source-num">#{i + 1}</span>
                  <span className="source-file">{s.source.split('\\').pop()}</span>
                  <span className="source-score">score {s.score.toFixed(4)}</span>
                </div>
                <p className="source-text">{s.text.slice(0, 260)}{s.text.length > 260 ? '…' : ''}</p>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

function Section({ icon, title, items, accent }) {
  const isKbMiss =
    items.length === 1 &&
    items[0].toLowerCase().includes('knowledge base does not')

  return (
    <div className={`section section-${accent} ${isKbMiss ? 'section-muted' : ''}`}>
      <h3 className="section-title">
        <span className="section-icon">{icon}</span>
        {title}
      </h3>
      <ul className="section-list">
        {items.map((item, i) => (
          <li key={i} className="section-item">
            <span className="bullet" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}
