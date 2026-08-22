import React, { useState } from 'react';
import { classifyLog } from '../utils/api';

const DEFAULT_FORM = {
  timestamp: '06-06-2025 14:15',
  src_ip: '192.168.1.10',
  dst_ip: '10.0.0.5',
  src_port: '54321',
  dst_port: '80',
  protocol: '6',
  bytes_sent: '500',
  bytes_received: '15000',
  flags: '18',
  duration: '2.5',
};

const FIELD_META = [
  { key: 'timestamp', label: 'Timestamp', placeholder: 'DD-MM-YYYY HH:MM' },
  { key: 'src_ip', label: 'Source IP', placeholder: '192.168.x.x' },
  { key: 'dst_ip', label: 'Destination IP', placeholder: '10.0.x.x' },
  { key: 'src_port', label: 'Src Port', placeholder: '0–65535' },
  { key: 'dst_port', label: 'Dst Port', placeholder: '80, 443...' },
  { key: 'protocol', label: 'Protocol', placeholder: '6=TCP 17=UDP' },
  { key: 'bytes_sent', label: 'Bytes Sent', placeholder: '0' },
  { key: 'bytes_received', label: 'Bytes Received', placeholder: '0' },
  { key: 'flags', label: 'TCP Flags', placeholder: '0–64' },
  { key: 'duration', label: 'Duration (s)', placeholder: '0.0' },
];

function formatDecision(result) {
  const exp = result.explanation;
  const decision = exp?.decision || result.decision;
  const confidence = exp?.confidence;
  const summary = exp?.summary;
  const keyFactors = exp?.key_factors || [];
  const allFeatures = exp?.all_features || [];
  const reason = result.reason;

  return { decision, confidence, summary, keyFactors, allFeatures, reason };
}

function FeatureRow({ f, maxImportance }) {
  const pct = maxImportance > 0 ? (f.importance / maxImportance) * 100 : 0;
  return (
    <tr>
      <td style={{ color: 'var(--text-dim)' }}>{f.feature}</td>
      <td className="mono">{typeof f.value === 'number' ? f.value : '-'}</td>
      <td style={{ color: 'var(--text-dim)', fontSize: 11 }}>{f.description || '—'}</td>
      <td>
        <div className="importance-bar">
          <div className="importance-track" style={{ minWidth: 80 }}>
            <div className="importance-fill" style={{ width: `${pct}%` }} />
          </div>
          <span className="mono" style={{ fontSize: 11, color: 'var(--text-dim)', minWidth: 40 }}>
            {f.importance?.toFixed(4)}
          </span>
        </div>
      </td>
    </tr>
  );
}

export default function ClassifyLog() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onChange = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const onSubmit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const payload = {
        ...form,
        src_port: Number(form.src_port),
        dst_port: Number(form.dst_port),
        protocol: Number(form.protocol),
        bytes_sent: Number(form.bytes_sent),
        bytes_received: Number(form.bytes_received),
        flags: Number(form.flags),
        duration: parseFloat(form.duration),
      };
      const data = await classifyLog(payload);
      setResult(data);
    } catch (e) {
      setError(e?.response?.data?.detail || 'Failed to connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  const parsed = result ? formatDecision(result) : null;
  const maxImportance = parsed?.allFeatures?.length
    ? Math.max(...parsed.allFeatures.map(f => f.importance || 0))
    : 1;

  const isAllow = parsed?.decision === 'ALLOW';
  const isReject = parsed?.decision === 'REJECT';

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Classify Network Log</h1>
        <p className="page-subtitle">// Submit a log entry to the AI engine for threat analysis</p>
      </div>

      <div className="card">
        <div className="card-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
          </svg>
          Log Entry Fields
        </div>
        <div className="field-grid">
          {FIELD_META.map(f => (
            <div className="field" key={f.key}>
              <label>{f.label}</label>
              <input
                value={form[f.key]}
                onChange={e => onChange(f.key, e.target.value)}
                placeholder={f.placeholder}
              />
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn btn-primary" onClick={onSubmit} disabled={loading}>
            {loading ? <div className="spinner" /> : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
              </svg>
            )}
            {loading ? 'Analyzing...' : 'Run Classification'}
          </button>
          <button className="btn btn-ghost" onClick={() => setForm(DEFAULT_FORM)}>Reset</button>
        </div>
        {error && <div className="msg msg-error">{error}</div>}
      </div>

      {parsed && (
        <div className="card">
          <div className="card-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            Engine Decision
          </div>

          {/* Top decision row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginBottom: 20, padding: '16px 0', borderBottom: '1px solid var(--border)' }}>
            <div>
              <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>Verdict</div>
              <span className={`badge ${isAllow ? 'badge-allow' : 'badge-reject'}`} style={{ fontSize: 16, padding: '8px 20px' }}>
                {isAllow ? '✓' : '✗'} {parsed.decision}
              </span>
            </div>

            {parsed.confidence !== undefined && (
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
                  Confidence — {(parsed.confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-bar" style={{ height: 8, borderRadius: 4 }}>
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${(parsed.confidence * 100).toFixed(1)}%`,
                      background: isAllow ? 'var(--green)' : 'var(--red)',
                    }}
                  />
                </div>
              </div>
            )}

            <div>
              <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>Source</div>
              <span className="mono" style={{ fontSize: 13, color: 'var(--text)' }}>{form.src_ip}</span>
            </div>
          </div>

          {/* Reason */}
          {parsed.reason && (
            <div style={{ marginBottom: 16 }}>
              <div className="card-title" style={{ marginBottom: 8 }}>Reason</div>
              <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px', fontFamily: 'JetBrains Mono', fontSize: 13, color: 'var(--text)' }}>
                {parsed.reason}
              </div>
            </div>
          )}

          {/* Summary */}
          {parsed.summary && (
            <div style={{ marginBottom: 20 }}>
              <div className="card-title" style={{ marginBottom: 8 }}>AI Summary</div>
              <div style={{ background: isAllow ? 'var(--green-dim)' : 'var(--red-dim)', border: `1px solid ${isAllow ? '#00e5a030' : '#ff4b6e30'}`, borderRadius: 8, padding: '12px 16px', fontSize: 13, color: isAllow ? 'var(--green)' : 'var(--red)', lineHeight: 1.6 }}>
                {parsed.summary}
              </div>
            </div>
          )}

          {/* Top factors */}
          {parsed.keyFactors?.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <div className="card-title" style={{ marginBottom: 10 }}>Top Decision Factors</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
                {parsed.keyFactors.map((f, i) => (
                  <div key={i} style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: 14 }}>
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
                      {f.feature}
                    </div>
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: 14, color: 'var(--text)', fontWeight: 600, marginBottom: 4 }}>
                      {f.value}
                    </div>
                    {f.description && (
                      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
                        {f.description}
                      </div>
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div className="importance-track" style={{ minWidth: 0, flex: 1 }}>
                        <div
                          className="importance-fill"
                          style={{ width: `${(f.importance / maxImportance) * 100}%` }}
                        />
                      </div>
                      <span className="mono" style={{ fontSize: 10, color: 'var(--green)' }}>
                        {f.importance?.toFixed(4)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* All features table */}
          {parsed.allFeatures?.length > 0 && (
            <div>
              <div className="card-title" style={{ marginBottom: 10 }}>All Feature Analysis</div>
              <div style={{ overflowX: 'auto' }}>
                <table className="feature-table">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Value</th>
                      <th>Description</th>
                      <th>Importance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {parsed.allFeatures.map((f, i) => (
                      <FeatureRow key={i} f={f} maxImportance={maxImportance} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}