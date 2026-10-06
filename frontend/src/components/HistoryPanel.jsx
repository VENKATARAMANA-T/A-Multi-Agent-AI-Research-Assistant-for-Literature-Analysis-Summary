import { useCallback, useEffect, useState } from 'react';
import api from '../api/client';
import { Badge, Card, Spinner } from './common';
import { useFlash } from '../context/FlashContext';

function when(iso) {
  const date = new Date(iso);
  const minutes = Math.round((Date.now() - date.getTime()) / 60000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`;
  if (minutes < 60 * 24 * 7) return `${Math.round(minutes / (60 * 24))} d ago`;
  return date.toLocaleDateString();
}

/**
 * Past results for a page, and a way to reopen one.
 *
 * Every agent run was already being written down as an audit trail; this makes
 * that record usable. Reopening costs nothing — the run holds its own output —
 * which matters on a tier that allows twenty model requests a day, where
 * re-running something just to look at it again is a real cost.
 *
 * `intents` is a list because one page can produce more than one kind of run:
 * Summaries writes `summarize` or `multi_summarize` depending on the mode.
 */
export default function HistoryPanel({
  intents,
  onOpen,
  refreshKey = 0,
  title = 'History',
  empty = 'Nothing saved yet. Results appear here once you run this agent.',
}) {
  const flash = useFlash();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openingId, setOpeningId] = useState(null);

  const key = intents.join(',');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setItems(await api.listHistory({ intent: intents }));
    } catch {
      // A failed history load must not take the page with it.
      setItems([]);
    } finally {
      setLoading(false);
    }
    // `key` stands in for `intents`, which is a new array on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  const open = async (id) => {
    setOpeningId(id);
    try {
      onOpen(await api.getHistory(id));
    } catch (err) {
      flash.error(err.message);
    } finally {
      setOpeningId(null);
    }
  };

  const forget = async (id, event) => {
    event.stopPropagation();
    try {
      await api.deleteHistory(id);
      setItems((current) => current.filter((item) => item.id !== id));
    } catch (err) {
      flash.error(err.message);
    }
  };

  const clearAll = async () => {
    try {
      await api.clearHistory({ intent: intents });
      setItems([]);
      flash.info('History cleared.');
    } catch (err) {
      flash.error(err.message);
    }
  };

  return (
    <Card
      title={title}
      actions={
        items.length > 0 && (
          <button type="button" className="btn btn-ghost btn-sm" onClick={clearAll}>
            Clear
          </button>
        )
      }
    >
      {loading ? (
        <Spinner label="Loading…" />
      ) : items.length === 0 ? (
        <p className="muted">{empty}</p>
      ) : (
        <ul className="history-list">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                className="history-open"
                onClick={() => open(item.id)}
                disabled={openingId === item.id}
              >
                <span className="history-label">{item.label}</span>
                <span className="history-meta">
                  {when(item.created_at)} · {item.paper_count} paper
                  {item.paper_count === 1 ? '' : 's'}
                  {item.status !== 'completed' && (
                    <>
                      {' · '}
                      <Badge tone={item.status === 'failed' ? 'danger' : 'warning'}>
                        {item.status}
                      </Badge>
                    </>
                  )}
                </span>
              </button>
              <button
                type="button"
                className="history-forget"
                onClick={(event) => forget(item.id, event)}
                title="Remove from history"
                aria-label="Remove from history"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
