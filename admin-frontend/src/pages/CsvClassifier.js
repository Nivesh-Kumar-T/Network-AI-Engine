import React, { useState, useRef } from 'react';
import { api } from '../utils/api';

export default function CsvClassifier() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState('');
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);
  const inputRef = useRef();

  const onFile = (f) => {
    if (f && f.name.endsWith('.csv')) {
      setFile(f);
      setDownloadUrl('');
      setError('');
      setDone(false);
    } else {
      setError('Please select a valid .csv file.');
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    onFile(e.dataTransfer.files[0]);
  };

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setDone(false);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await api.post('/classify-csv', fd, {
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const url = URL.createObjectURL(new Blob([res.data], { type: 'text/csv' }));
      setDownloadUrl(url);
      setDone(true);
    } catch (e) {
      setError('Classification failed. Check backend logs.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">CSV Bulk Classifier</h1>
        <p className="page-subtitle">// Upload a batch of network logs for mass threat analysis</p>
      </div>

      <div className="card">
        <div className="card-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
          Upload CSV File
        </div>

        <div
          className={`upload-zone ${dragging ? 'active' : ''} ${file ? 'active' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current.click()}
        >
          <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          <input ref={inputRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={e => onFile(e.target.files[0])} />
          {file ? (
            <>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 14, color: 'var(--green)', fontWeight: 600 }}>
                {file.name}
              </div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>
                {(file.size / 1024).toFixed(1)} KB — Click to change
              </div>
            </>
          ) : (
            <>
              <div style={{ fontFamily: 'Syne', fontSize: 15, fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>
                Drop your CSV here or click to browse
              </div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)' }}>
                Required columns: timestamp, src_ip, dst_ip, src_port, dst_port,<br />
                protocol, bytes_sent, bytes_received, flags, duration
              </div>
            </>
          )}
        </div>

        <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
          <button className="btn btn-primary" onClick={onSubmit} disabled={!file || loading}>
            {loading ? <div className="spinner" /> : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
              </svg>
            )}
            {loading ? 'Processing...' : 'Classify All Logs'}
          </button>
          {downloadUrl && (
            <a
              href={downloadUrl}
              download="classified_logs.csv"
              className="btn btn-ghost"
              style={{ textDecoration: 'none' }}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
              </svg>
              Download Results
            </a>
          )}
        </div>

        {done && !error && (
          <div className="msg msg-success" style={{ marginTop: 16 }}>
            ✓ Classification complete — download your results above.
          </div>
        )}
        {error && <div className="msg msg-error" style={{ marginTop: 16 }}>{error}</div>}
      </div>

      <div className="card">
        <div className="card-title">Expected CSV Format</div>
        <div style={{ overflowX: 'auto' }}>
          <table className="feature-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Example</th>
                <th>Required</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['timestamp', 'string', '06-06-2025 14:15', 'No (default applied)'],
                ['src_ip', 'string', '192.168.1.10', 'No'],
                ['dst_ip', 'string', '10.0.0.5', 'No'],
                ['src_port', 'integer', '54321', 'No (default: 0)'],
                ['dst_port', 'integer', '443', 'No (default: 0)'],
                ['protocol', 'integer', '6 (TCP)', 'No (default: 0)'],
                ['bytes_sent', 'integer', '1024', 'No (default: 0)'],
                ['bytes_received', 'integer', '8192', 'No (default: 0)'],
                ['flags', 'integer', '18 (SYN-ACK)', 'No (default: 0)'],
                ['duration', 'float', '2.5', 'No (default: 0.0)'],
              ].map(([col, type, ex, req]) => (
                <tr key={col}>
                  <td className="mono" style={{ color: 'var(--green)' }}>{col}</td>
                  <td className="mono" style={{ color: 'var(--blue)' }}>{type}</td>
                  <td className="mono">{ex}</td>
                  <td style={{ color: req.startsWith('No') ? 'var(--text-dim)' : 'var(--amber)' }}>{req}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}