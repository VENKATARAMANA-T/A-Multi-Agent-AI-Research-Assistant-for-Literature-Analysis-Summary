import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import { Badge, Card, ErrorBanner, Spinner, statusTone } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const STAGE_LABELS = {
  storing: 'Storing file',
  extracting: 'Extracting text',
  chunking: 'Chunking',
  embedding: 'Embedding',
  indexing: 'Indexing',
};

const TERMINAL = ['completed', 'partial', 'failed'];

export default function Upload() {
  const { refresh } = useCorpus();
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const streamRef = useRef(null);

  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const [job, setJob] = useState(null);

  const busy = Boolean(job && !TERMINAL.includes(job.status));

  // Close any open stream when the component unmounts.
  useEffect(() => () => streamRef.current?.close(), []);

  const follow = useCallback(
    (jobId) => {
      streamRef.current?.close();
      const source = api.streamJob(jobId);
      streamRef.current = source;

      const onUpdate = (event) => {
        try {
          setJob(JSON.parse(event.data));
        } catch {
          /* ignore malformed frames */
        }
      };

      source.addEventListener('progress', onUpdate);
      source.addEventListener('done', (event) => {
        onUpdate(event);
        source.close();
        refresh();
      });
      // The browser retries automatically on a dropped connection; fall back to
      // a single poll so a closed stream never leaves the UI stuck mid-progress.
      source.onerror = () => {
        source.close();
        api
          .getJob(jobId)
          .then((payload) => {
            setJob(payload);
            if (TERMINAL.includes(payload.status)) refresh();
          })
          .catch(() => setError('Lost contact with the indexing job.'));
      };
    },
    [refresh],
  );

  const handleFiles = async (fileList) => {
    const files = Array.from(fileList || []).filter((file) =>
      file.name.toLowerCase().endsWith('.pdf'),
    );
    if (files.length === 0) {
      setError('Please choose at least one PDF file.');
      return;
    }

    setError(null);
    setJob(null);
    try {
      const started = await api.uploadPapers(files);
      setJob(started);
      follow(started.id);
    } catch (err) {
      setError(err.message);
    } finally {
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const percent = Math.round((job?.progress ?? 0) * 100);
  const indexed = job?.items?.filter((item) => item.state === 'indexed').length ?? 0;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Upload papers</h1>
          <p className="page-sub">
            Indexing runs in the background — you can navigate away and come back.
          </p>
        </div>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div
        className={`dropzone ${dragging ? 'is-dragging' : ''} ${busy ? 'is-busy' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!busy) handleFiles(event.dataTransfer.files);
        }}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if ((event.key === 'Enter' || event.key === ' ') && !busy) inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(event) => handleFiles(event.target.files)}
        />
        {busy ? (
          <Spinner label={`Indexing ${job.completed + job.failed} of ${job.total}…`} />
        ) : (
          <>
            <div className="dropzone-icon" aria-hidden="true">
              ⇪
            </div>
            <p className="dropzone-title">Drop PDFs here, or click to browse</p>
            <p className="dropzone-hint">
              Up to 20 files per upload · 50 MB each · text-layer PDFs only (scans need OCR first)
            </p>
          </>
        )}
      </div>

      {job && (
        <Card
          title={busy ? 'Indexing in progress' : 'Upload complete'}
          subtitle={
            busy
              ? `${job.completed + job.failed} of ${job.total} processed`
              : `${indexed} indexed · ${job.failed} failed`
          }
          actions={
            !busy &&
            indexed > 0 && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() => navigate('/library')}
              >
                Go to library
              </button>
            )
          }
        >
          <div
            className="progress"
            role="progressbar"
            aria-valuenow={percent}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className={`progress-fill ${job.status === 'failed' ? 'is-failed' : ''}`}
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="progress-label">{percent}%</p>

          <table className="table">
            <thead>
              <tr>
                <th>File</th>
                <th>Status</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {(job.items || []).map((item, index) => (
                <tr key={index}>
                  <td className="mono">{item.name}</td>
                  <td>
                    {item.state === 'running' ? (
                      <Badge tone="info">{STAGE_LABELS[item.stage] || 'Working'}</Badge>
                    ) : item.state === 'queued' ? (
                      <Badge tone="neutral">Queued</Badge>
                    ) : (
                      <Badge tone={statusTone(item.state)}>{item.state}</Badge>
                    )}
                  </td>
                  <td>{item.detail || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
