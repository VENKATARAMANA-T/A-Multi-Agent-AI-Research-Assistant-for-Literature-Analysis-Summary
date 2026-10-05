import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

// Each card holds the stage for three seconds before the next one takes over.
const REEL_INTERVAL_MS = 3000;

const REEL = [
  {
    key: 'qa',
    head: 'Question Answering Agent',
    body: (
      <>
        <p>
          SparseSum reaches 44.1 ROUGE-L on arXiv <span className="hero-cite">[S1]</span>,
          outperforming the dense baseline <span className="hero-cite">[S2]</span>.
        </p>
        <div className="hero-chip">high confidence</div>
      </>
    ),
  },
  {
    key: 'verify',
    head: 'Verification',
    body: (
      <>
        <div className="hero-verdicts">
          <span className="hero-verdict is-ok">Supported</span>
          <span className="hero-verdict is-ok">Supported</span>
          <span className="hero-verdict is-bad">Unsupported</span>
        </div>
        <div className="hero-score">75% grounded · 4 claims checked</div>
      </>
    ),
  },
  {
    key: 'graph',
    head: 'Knowledge graph',
    body: (
      <>
        <svg viewBox="0 0 220 110" className="hero-graph">
          <line x1="40" y1="30" x2="110" y2="60" />
          <line x1="110" y1="60" x2="180" y2="28" />
          <line x1="110" y1="60" x2="95" y2="98" />
          <line x1="40" y1="30" x2="180" y2="28" />
          <circle cx="40" cy="30" r="9" className="n-method" />
          <circle cx="110" cy="60" r="11" className="n-dataset" />
          <circle cx="180" cy="28" r="9" className="n-method" />
          <circle cx="95" cy="98" r="7" className="n-metric" />
        </svg>
        <div className="hero-score">84 entities · 141 relationships</div>
      </>
    ),
  },
  {
    key: 'ocr',
    head: 'OCR · scanned pages',
    body: (
      <>
        <p className="hero-ocr">
          <span className="hero-strike">the acceleration spectrum of the</span> vibrational field
          is related to the pressure spectrum
        </p>
        <div className="hero-meter">
          <span style={{ width: '82%' }} />
        </div>
        <div className="hero-score">16 of 28 pages recovered · 0.82 confidence</div>
      </>
    ),
  },
  {
    key: 'figures',
    head: 'Figures read by vision',
    body: (
      <>
        <svg viewBox="0 0 220 96" className="hero-chart">
          <line x1="22" y1="86" x2="212" y2="86" />
          <line x1="22" y1="86" x2="22" y2="8" />
          <rect x="44" y="52" width="34" height="34" className="b1" />
          <rect x="94" y="34" width="34" height="52" className="b2" />
          <rect x="144" y="16" width="34" height="70" className="b3" />
        </svg>
        <div className="hero-score">wav2vec reaches 0.93 — highest of the three</div>
      </>
    ),
  },
];

/**
 * A reel of five cards: one on stage at a time, each sliding down and away as
 * the next rises into its place.
 *
 * Three cards fanned out statically looked like a screenshot of clutter. One at
 * a time means each gets read, and five fit where three overlapped.
 */
function HeroReel() {
  const [index, setIndex] = useState(0);
  const [leaving, setLeaving] = useState(null);
  const [paused, setPaused] = useState(false);
  const timer = useRef(null);

  const advance = (next) => {
    setLeaving((current) => (current === null ? index : current));
    setIndex(next);
  };

  useEffect(() => {
    if (paused) return undefined;
    timer.current = setInterval(() => {
      setLeaving(null);
      setIndex((current) => {
        setLeaving(current);
        return (current + 1) % REEL.length;
      });
    }, REEL_INTERVAL_MS);
    return () => clearInterval(timer.current);
  }, [paused]);

  // The outgoing card is only kept mounted long enough to animate out.
  useEffect(() => {
    if (leaving === null) return undefined;
    const id = setTimeout(() => setLeaving(null), 650);
    return () => clearTimeout(id);
  }, [leaving]);

  return (
    <div
      className="hero-visual"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <div className="hero-reel">
        {REEL.map((card, i) => (
          <article
            key={card.key}
            className={`hero-card ${
              i === index ? 'is-active' : i === leaving ? 'is-leaving' : ''
            }`}
            aria-hidden={i !== index}
          >
            <div className="hero-card-head">{card.head}</div>
            {card.body}
          </article>
        ))}
      </div>

      <div className="reel-dots" role="tablist" aria-label="Feature preview">
        {REEL.map((card, i) => (
          <button
            key={card.key}
            type="button"
            role="tab"
            aria-selected={i === index}
            aria-label={card.head}
            className={`reel-dot ${i === index ? 'is-active' : ''}`}
            onClick={() => advance(i)}
          />
        ))}
      </div>
    </div>
  );
}

const PILLARS = [
  {
    icon: '?',
    title: 'Ask your corpus',
    body: 'Retrieval-augmented answers with citations back to the exact page and passage. Click any marker to land on the sentence in the PDF.',
  },
  {
    icon: '⁂',
    title: 'See the relationships',
    body: 'A knowledge graph built from your papers answers questions no single passage contains — which methods share a dataset, what connects two fields.',
  },
  {
    icon: '▩',
    title: 'Read the figures',
    body: 'Charts and tables are lifted out of the PDF and read by a vision model, so a result that only exists in a plot becomes searchable.',
  },
  {
    icon: '⌾',
    title: 'Recover scanned pages',
    body: 'Photocopies and archive scans go through OCR page by page, with confidence scores and provenance on every chunk.',
  },
  {
    icon: '⌁',
    title: 'Find the unstated',
    body: 'Literature-based discovery surfaces connections your papers imply but none of them states — the Swanson ABC method, automated.',
  },
  {
    icon: '✓',
    title: 'Check the answers',
    body: 'A verification agent re-retrieves evidence for every claim independently, so a green badge means the corpus backs it — not that the model agreed with itself.',
  },
];

const STEPS = [
  ['Upload', 'Drop in PDFs. Text, metadata, sections, figures and tables come out automatically.'],
  ['Index', 'Chunked, embedded and stored in a vector database — searchable without any API key.'],
  ['Analyse', 'Eight specialised agents summarise, extract, compare, map and fact-check.'],
  ['Report', 'Export a literature review as Markdown or PDF, with every claim traceable.'],
];

export default function Home() {
  const { isAuthenticated, user } = useAuth();

  return (
    <div className="home">
      <header className="home-nav">
        <div className="brand">
          <span className="brand-mark">RC</span>
          <div>
            <div className="brand-name">ResearchCompass</div>
            <div className="brand-tag">Multi-agent literature analysis</div>
          </div>
        </div>

        <div className="home-nav-actions">
          {isAuthenticated ? (
            <>
              <span className="muted">Signed in as {user.username}</span>
              <Link className="btn btn-primary" to="/app">
                Open workspace →
              </Link>
            </>
          ) : (
            <>
              <Link className="btn btn-ghost" to="/signin">
                Sign in
              </Link>
              <Link className="btn btn-primary" to="/signup">
                Sign up
              </Link>
            </>
          )}
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="hero-badge">Eight agents · one corpus</span>
          <h1>
            Read a hundred papers
            <br />
            <span className="hero-accent">like you read one.</span>
          </h1>
          <p className="hero-sub">
            Upload your literature and ResearchCompass extracts it, indexes it, maps the
            relationships between papers, and answers questions with citations you can click
            through to the source. Every claim is traceable — and a verification agent checks
            the others' work against evidence it gathers itself.
          </p>

          <div className="hero-actions">
            {isAuthenticated ? (
              <Link className="btn btn-primary btn-lg" to="/app">
                Open your workspace →
              </Link>
            ) : (
              <>
                <Link className="btn btn-primary btn-lg" to="/signup">
                  Create an account
                </Link>
                <Link className="btn btn-ghost btn-lg" to="/signin">
                  I already have one
                </Link>
              </>
            )}
          </div>

          <p className="hero-note">
            Runs on your machine. Your papers are never sent anywhere except the model you
            configure.
          </p>
        </div>

        <HeroReel />
      </section>

      <section className="home-section">
        <h2>What it does</h2>
        <div className="pillar-grid">
          {PILLARS.map((pillar) => (
            <article key={pillar.title} className="pillar">
              <span className="pillar-icon" aria-hidden="true">
                {pillar.icon}
              </span>
              <h3>{pillar.title}</h3>
              <p>{pillar.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="home-section">
        <h2>How it works</h2>
        <ol className="step-row">
          {STEPS.map(([title, body], index) => (
            <li key={title}>
              <span className="step-number">{index + 1}</span>
              <h3>{title}</h3>
              <p>{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="home-cta">
        <h2>Start with one paper.</h2>
        <p>
          Upload a single PDF and ask it a question. Everything else — the graph, the figures,
          the gap analysis — builds on the same corpus as it grows.
        </p>
        {isAuthenticated ? (
          <Link className="btn btn-primary btn-lg" to="/app/upload">
            Upload a paper →
          </Link>
        ) : (
          <Link className="btn btn-primary btn-lg" to="/signup">
            Create an account →
          </Link>
        )}
      </section>

      <footer className="home-footer">
        <span>ResearchCompass — multi-agent literature analysis</span>
        <span className="muted">FastAPI · LangGraph · ChromaDB · Neo4j · Gemini</span>
      </footer>
    </div>
  );
}
