import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import api from '../api/client';

const CorpusContext = createContext(null);

const STORAGE_KEY = 'researchcompass.scopes';

/**
 * Which page's selection we are looking at.
 *
 * Selection used to be one list for the whole app, which meant ticking a paper
 * in the Library silently changed what Figures, Ask and the review generator
 * would act on. Each page now keeps its own, because "the papers I am reading"
 * and "the papers I want a review written from" are different questions that
 * happen to share a widget.
 */
function scopeFor(pathname) {
  const segment = pathname.replace(/^\/app\/?/, '').split('/')[0];
  return segment || 'dashboard';
}

function loadScopes() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {};
  } catch {
    return {};
  }
}

export function CorpusProvider({ children }) {
  const [papers, setPapers] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // { [page]: string[] } — one selection per page, persisted together.
  const [scopes, setScopes] = useState(loadScopes);

  const { pathname } = useLocation();
  const scope = scopeFor(pathname);
  const selectedIds = useMemo(() => scopes[scope] || [], [scopes, scope]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [paperList, healthPayload] = await Promise.all([
        api.listPapers(),
        api.health().catch(() => null),
      ]);
      setPapers(paperList);
      setHealth(healthPayload);
      setError(null);

      // Drop selections that point at papers which no longer exist — in every
      // page's list, not just the one currently on screen.
      const valid = new Set(paperList.map((p) => p.id));
      setScopes((current) => {
        const next = {};
        let changed = false;
        for (const [key, ids] of Object.entries(current)) {
          const kept = ids.filter((id) => valid.has(id));
          if (kept.length !== ids.length) changed = true;
          if (kept.length) next[key] = kept;
          else if (ids.length) changed = true;
        }
        return changed ? next : current;
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(scopes));
    } catch {
      // Storage can be unavailable; the selection then lasts for the session.
    }
  }, [scopes]);

  const indexedPapers = useMemo(() => papers.filter((p) => p.status === 'indexed'), [papers]);

  /** Replace this page's selection. Accepts a value or an updater, like setState. */
  const setSelectedIds = useCallback(
    (update) => {
      setScopes((current) => {
        const existing = current[scope] || [];
        const next = typeof update === 'function' ? update(existing) : update;
        return { ...current, [scope]: next };
      });
    },
    [scope],
  );

  const toggleSelected = useCallback(
    (id) => {
      setSelectedIds((current) =>
        current.includes(id) ? current.filter((x) => x !== id) : [...current, id],
      );
    },
    [setSelectedIds],
  );

  const selectAll = useCallback(() => {
    setSelectedIds(indexedPapers.map((p) => p.id));
  }, [setSelectedIds, indexedPapers]);

  const clearSelection = useCallback(() => setSelectedIds([]), [setSelectedIds]);

  // An empty selection means "the whole indexed corpus" on the backend.
  const effectiveIds = useMemo(
    () => (selectedIds.length ? selectedIds : indexedPapers.map((p) => p.id)),
    [selectedIds, indexedPapers],
  );

  const value = useMemo(
    () => ({
      papers,
      indexedPapers,
      health,
      loading,
      error,
      refresh,
      scope,
      selectedIds,
      effectiveIds,
      setSelectedIds,
      toggleSelected,
      selectAll,
      clearSelection,
      llmReady: Boolean(health?.llm?.configured),
      titleFor: (id) => papers.find((p) => p.id === id)?.title || id,
    }),
    [
      papers,
      indexedPapers,
      health,
      loading,
      error,
      refresh,
      scope,
      selectedIds,
      effectiveIds,
      setSelectedIds,
      toggleSelected,
      selectAll,
      clearSelection,
    ],
  );

  return <CorpusContext.Provider value={value}>{children}</CorpusContext.Provider>;
}

export function useCorpus() {
  const context = useContext(CorpusContext);
  if (!context) throw new Error('useCorpus must be used inside a CorpusProvider');
  return context;
}
