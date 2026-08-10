import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import api from '../api/client';

const CorpusContext = createContext(null);

const STORAGE_KEY = 'researchcompass.selectedPapers';

export function CorpusProvider({ children }) {
  const [papers, setPapers] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedIds, setSelectedIds] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
      return [];
    }
  });

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

      // Drop selections that point at papers which no longer exist.
      const validIds = new Set(paperList.map((p) => p.id));
      setSelectedIds((current) => current.filter((id) => validIds.has(id)));
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
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedIds));
  }, [selectedIds]);

  const indexedPapers = useMemo(() => papers.filter((p) => p.status === 'indexed'), [papers]);

  const toggleSelected = useCallback((id) => {
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id],
    );
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(indexedPapers.map((p) => p.id));
  }, [indexedPapers]);

  const clearSelection = useCallback(() => setSelectedIds([]), []);

  // An empty selection means "the whole indexed corpus" on the backend.
  const effectiveIds = selectedIds.length ? selectedIds : indexedPapers.map((p) => p.id);

  const value = useMemo(
    () => ({
      papers,
      indexedPapers,
      health,
      loading,
      error,
      refresh,
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
      selectedIds,
      effectiveIds,
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
