import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import api from '../api/client';
import AgentTrace from '../components/AgentTrace';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const NODE_COLORS = {
  Paper: '#2563eb',
  Method: '#7c3aed',
  Dataset: '#0891b2',
  Metric: '#ca8a04',
  Task: '#dc2626',
  Concept: '#059669',
  Author: '#db2777',
  Tool: '#ea580c',
  Application: '#4f46e5',
};

const DEFAULT_COLOR = '#64748b';

export default function GraphView() {
  const { effectiveIds, indexedPapers, llmReady, titleFor } = useCorpus();
  const containerRef = useRef(null);
  const graphRef = useRef(null);

  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState(null);
  const [buildResult, setBuildResult] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hiddenTypes, setHiddenTypes] = useState(() => new Set());
  const [minDegree, setMinDegree] = useState(0);
  const [size, setSize] = useState({ width: 800, height: 560 });

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setGraph(await api.getGraph({ limit: 3000 }));
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(320, entry.contentRect.width),
        height: Math.max(420, Math.round(window.innerHeight * 0.66)),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [graph]);

  const build = async () => {
    setBuilding(true);
    setError(null);
    try {
      const result = await api.buildGraph({ paper_ids: effectiveIds });
      setBuildResult(result);
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBuilding(false);
    }
  };

  const clear = async () => {
    if (!window.confirm('Delete every node and relation in the knowledge graph?')) return;
    try {
      await api.clearGraph();
      setSelectedNode(null);
      await load();
    } catch (err) {
      setError(err.message);
    }
  };

  const data = useMemo(() => {
    if (!graph) return { nodes: [], links: [] };

    const degree = {};
    graph.edges.forEach((edge) => {
      degree[edge.source] = (degree[edge.source] || 0) + 1;
      degree[edge.target] = (degree[edge.target] || 0) + 1;
    });

    const nodes = graph.nodes
      .filter((node) => !hiddenTypes.has(node.type))
      .filter((node) => (degree[node.id] || 0) >= minDegree)
      .map((node) => ({ ...node, degree: degree[node.id] || 0 }));

    const ids = new Set(nodes.map((node) => node.id));
    const links = graph.edges
      .filter((edge) => ids.has(edge.source) && ids.has(edge.target))
      .map((edge) => ({ ...edge }));

    return { nodes, links };
  }, [graph, hiddenTypes, minDegree]);

  const types = useMemo(() => {
    const counts = {};
    (graph?.nodes || []).forEach((node) => {
      counts[node.type] = (counts[node.type] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [graph]);

  const toggleType = (type) => {
    setHiddenTypes((current) => {
      const next = new Set(current);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const drawNode = useCallback((node, ctx, globalScale) => {
    const radius = node.type === 'Paper' ? 7 : 4 + Math.min(4, node.degree * 0.35);
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = NODE_COLORS[node.type] || DEFAULT_COLOR;
    ctx.fill();
    ctx.lineWidth = 1 / globalScale;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.stroke();

    if (globalScale > 1.1 || node.type === 'Paper') {
      const label = String(node.name || node.id);
      const fontSize = Math.max(9, 11 / globalScale);
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = '#0f172a';
      ctx.fillText(label.length > 34 ? `${label.slice(0, 33)}…` : label, node.x, node.y + radius + 2);
    }
  }, []);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Knowledge graph</h1>
          <p className="page-sub">
            Entities and typed relations extracted from the corpus
            {graph?.backend ? ` · backend: ${graph.backend}` : ''}
          </p>
        </div>
        <div className="row-actions">
          <button type="button" className="btn btn-ghost" onClick={load} disabled={loading}>
            Refresh
          </button>
          <button type="button" className="btn btn-danger" onClick={clear}>
            Clear graph
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={build}
            disabled={building || indexedPapers.length === 0}
          >
            {building ? 'Building…' : 'Build from selection'}
          </button>
        </div>
      </header>

      {!llmReady && (
        <WarningBanner>
          Gemini is not configured, so new graph fragments cannot be extracted. Any previously
          built graph is still viewable.
        </WarningBanner>
      )}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {graph?.backend === 'in-memory' && (
        <WarningBanner>
          Neo4j is unreachable, so the graph is stored in a local JSON file instead. Start the Neo4j
          container (<code>docker compose up neo4j</code>) and restart the API to use the real graph
          database.
        </WarningBanner>
      )}

      <div className="graph-layout">
        <div className="graph-canvas" ref={containerRef}>
          {loading ? (
            <Spinner label="Loading graph…" />
          ) : data.nodes.length === 0 ? (
            <EmptyState
              title="The graph is empty"
              description="Select papers on the right and click “Build from selection” to extract entities and relations."
            />
          ) : (
            <ForceGraph2D
              ref={graphRef}
              graphData={data}
              width={size.width}
              height={size.height}
              backgroundColor="#ffffff"
              nodeCanvasObject={drawNode}
              nodePointerAreaPaint={(node, color, ctx) => {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
                ctx.fill();
              }}
              nodeLabel={(node) =>
                `${node.name} (${node.type})${node.description ? `\n${node.description}` : ''}`
              }
              linkLabel={(link) => `${link.type}${link.evidence ? `\n${link.evidence}` : ''}`}
              linkColor={() => 'rgba(100,116,139,0.35)'}
              linkDirectionalArrowLength={3.5}
              linkDirectionalArrowRelPos={1}
              linkWidth={0.8}
              cooldownTicks={120}
              onNodeClick={(node) => {
                setSelectedNode(node);
                graphRef.current?.centerAt(node.x, node.y, 600);
                graphRef.current?.zoom(2.4, 600);
              }}
            />
          )}
        </div>

        <aside className="graph-side">
          <Card title="Filters">
            <div className="legend">
              {types.map(([type, count]) => (
                <button
                  key={type}
                  type="button"
                  className={`legend-item ${hiddenTypes.has(type) ? 'is-off' : ''}`}
                  onClick={() => toggleType(type)}
                >
                  <span
                    className="legend-swatch"
                    style={{ background: NODE_COLORS[type] || DEFAULT_COLOR }}
                  />
                  {type}
                  <span className="legend-count">{count}</span>
                </button>
              ))}
            </div>

            <label className="field">
              Minimum connections: <strong>{minDegree}</strong>
              <input
                type="range"
                min={0}
                max={8}
                value={minDegree}
                onChange={(event) => setMinDegree(Number(event.target.value))}
              />
            </label>

            <p className="muted">
              Showing {data.nodes.length} nodes · {data.links.length} relations
            </p>
          </Card>

          {selectedNode && (
            <Card
              title={selectedNode.name}
              subtitle={selectedNode.type}
              actions={
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => setSelectedNode(null)}
                >
                  Close
                </button>
              }
            >
              {selectedNode.description && <p>{selectedNode.description}</p>}
              <p className="muted">{selectedNode.degree} connection(s)</p>
              {selectedNode.papers?.length > 0 && (
                <>
                  <h4>Appears in</h4>
                  <div className="chip-row">
                    {selectedNode.papers.map((paperId) => (
                      <span key={paperId} className="chip chip-muted">
                        {titleFor(paperId)}
                      </span>
                    ))}
                  </div>
                </>
              )}
              {selectedNode.authors?.length > 0 && (
                <p className="muted">{selectedNode.authors.join(', ')}</p>
              )}
            </Card>
          )}

          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>

          {buildResult && (
            <Card title="Last build">
              <div className="chip-row">
                <Badge tone="info">{buildResult.graph?.nodes?.length ?? 0} nodes</Badge>
                <Badge tone="info">{buildResult.graph?.edges?.length ?? 0} relations</Badge>
              </div>
              <AgentTrace
                trace={buildResult.trace}
                errors={buildResult.errors}
                llmCalls={buildResult.llm_calls}
                durationMs={buildResult.duration_ms}
              />
            </Card>
          )}
        </aside>
      </div>
    </div>
  );
}
