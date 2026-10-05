import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

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

// How long each card holds the stage before the next one drops in.
const REEL_INTERVAL_MS = 3000;
// Must match the CSS transition, so the outgoing card is unmounted from the
// stage only once it has finished falling.
const DROP_MS = 700;

const REEL = [
  {
    key: 'qa',
    head: 'Ask',
    title: 'Answers you can trace',
    body: (
      <p className="reel-quote">
        SparseSum reaches <strong>44.1 ROUGE-L</strong> on arXiv{' '}
        <span className="hero-cite">[S1]</span>, outperforming the dense baseline by 2.3
        points <span className="hero-cite">[S2]</span>.
      </p>
    ),
    foot: (
      <>
        <span className="hero-chip">high confidence</span>
        <span className="reel-foot-note">every marker opens the page it came from</span>
      </>
    ),
  },
  {
    key: 'verify',
    head: 'Verify',
    title: 'It checks its own work',
    body: (
      <>
        <div className="hero-verdicts">
          <span className="hero-verdict is-ok">Supported</span>
          <span className="hero-verdict is-ok">Supported</span>
          <span className="hero-verdict is-ok">Supported</span>
          <span className="hero-verdict is-bad">Unsupported</span>
        </div>
        <div className="hero-meter">
          <span style={{ width: '75%' }} />
        </div>
      </>
    ),
    foot: (
      <>
        <strong className="reel-stat">75% grounded</strong>
        <span className="reel-foot-note">evidence re-retrieved for each claim</span>
      </>
    ),
  },
  {
    key: 'graph',
    head: 'Connect',
    title: 'Relationships, not just passages',
    body: (
      <svg viewBox="0 0 240 92" className="hero-graph" role="img" aria-label="Knowledge graph">
        <line x1="36" y1="24" x2="118" y2="52" />
        <line x1="118" y1="52" x2="204" y2="22" />
        <line x1="118" y1="52" x2="96" y2="82" />
        <line x1="36" y1="24" x2="204" y2="22" />
        <line x1="204" y1="22" x2="96" y2="82" />
        <circle cx="36" cy="24" r="9" className="n-method" />
        <circle cx="118" cy="52" r="12" className="n-dataset" />
        <circle cx="204" cy="22" r="9" className="n-method" />
        <circle cx="96" cy="82" r="7" className="n-metric" />
      </svg>
    ),
    foot: (
      <>
        <strong className="reel-stat">84 entities · 141 links</strong>
        <span className="reel-foot-note">answers no single passage contains</span>
      </>
    ),
  },
  {
    key: 'ocr',
    head: 'Recover',
    title: 'Scans become searchable',
    body: (
      <>
        <p className="reel-quote hero-ocr">
          <span className="hero-strike">▚▚▚ ▚▚▚▚▚▚▚▚ ▚▚▚▚▚▚</span> the acceleration spectrum
          of the vibrational field is related to the pressure spectrum
        </p>
        <div className="hero-meter">
          <span style={{ width: '82%' }} />
        </div>
      </>
    ),
    foot: (
      <>
        <strong className="reel-stat">16 of 28 pages</strong>
        <span className="reel-foot-note">recovered from a 1962 scan · 0.82 confidence</span>
      </>
    ),
  },
  {
    key: 'figures',
    head: 'Read figures',
    title: 'Charts the text throws away',
    body: (
      <svg viewBox="0 0 240 92" className="hero-chart" role="img" aria-label="Accuracy by feature set">
        <line x1="26" y1="82" x2="232" y2="82" />
        <line x1="26" y1="82" x2="26" y2="8" />
        <rect x="52" y="52" width="38" height="30" className="b1" />
        <rect x="112" y="36" width="38" height="46" className="b2" />
        <rect x="172" y="16" width="38" height="66" className="b3" />
        <text x="60" y="94" className="hero-chart-label">MFCC</text>
        <text x="113" y="94" className="hero-chart-label">spectro</text>
        <text x="174" y="94" className="hero-chart-label">wav2vec</text>
      </svg>
    ),
    foot: (
      <>
        <strong className="reel-stat">0.93 accuracy</strong>
        <span className="reel-foot-note">read off the plot, then made searchable</span>
      </>
    ),
  },
];

/**
 * A reel of cards that drop through a window: the one on stage falls away
 * downwards while the next lands in its place from above.
 *
 * Strictly vertical. A sideways carousel reads as "there is more to the right"
 * and invites swiping; dropping reads as one thing replacing another, which is
 * what this is. Every card is the same size so the page does not twitch as the
 * content changes beneath it.
 */
function HeroReel() {
  const [index, setIndex] = useState(0);
  const [leaving, setLeaving] = useState(null);
  const [paused, setPaused] = useState(false);
  const clear = useRef(null);

  useEffect(() => {
    if (paused) return undefined;
    const timer = setInterval(() => {
      setIndex((current) => {
        setLeaving(current);
        return (current + 1) % REEL.length;
      });
    }, REEL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [paused]);

  // The outgoing card stays mounted on the stage only while it is falling;
  // afterwards it snaps back above the window with no transition, ready to drop
  // again on its next turn.
  useEffect(() => {
    if (leaving === null) return undefined;
    clear.current = setTimeout(() => setLeaving(null), DROP_MS);
    return () => clearTimeout(clear.current);
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
              i === index ? 'is-active' : i === leaving ? 'is-leaving' : 'is-waiting'
            }`}
            aria-hidden={i !== index}
          >
            <div className="hero-card-head">{card.head}</div>
            <h3 className="reel-title">{card.title}</h3>
            <div className="reel-body">{card.body}</div>
            <div className="reel-foot">{card.foot}</div>
          </article>
        ))}
      </div>

    </div>
  );
}


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
