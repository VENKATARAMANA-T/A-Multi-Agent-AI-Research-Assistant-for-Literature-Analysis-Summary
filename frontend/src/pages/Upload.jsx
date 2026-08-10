import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import { Badge, Card, ErrorBanner, Spinner, statusTone } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

export default function Upload() {
  const { refresh } = useCorpus();
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFiles = async (fileList) => {
    const files = Array.from(fileList || []).filter((file) =>
      file.name.toLowerCase().endsWith('.pdf'),
    );
    if (files.length === 0) {
      setError('Please choose at least one PDF file.');
      return;
    }

    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const payload = await api.uploadPapers(files);
      setResult(payload);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Upload papers</h1>
          <p className="page-sub">
            PDFs are extracted, chunked, embedded and indexed before the response returns.
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
          <Spinner label="Extracting, chunking, embedding and indexing…" />
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

      {result && (
        <Card
          title="Upload results"
          subtitle={`${result.indexed} indexed · ${result.duplicates} duplicates · ${result.failed} failed`}
          actions={
            result.indexed > 0 && (
              <button type="button" className="btn btn-primary btn-sm" onClick={() => navigate('/library')}>
                Go to library
              </button>
            )
          }
        >
          <table className="table">
            <thead>
              <tr>
                <th>File</th>
                <th>Status</th>
                <th>Detail</th>
                <th>Title detected</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((item, index) => (
                <tr key={index}>
                  <td className="mono">{item.filename}</td>
                  <td>
                    <Badge tone={statusTone(item.status)}>{item.status}</Badge>
                  </td>
                  <td>{item.detail}</td>
                  <td>{item.paper?.title || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
