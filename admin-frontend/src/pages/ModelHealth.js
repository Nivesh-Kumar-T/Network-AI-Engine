import React, { useState, useEffect } from 'react';
import { getStatus } from '../utils/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

function MetricCard({ label, value, color, suffix = '%' }) {
  const num = typeof value === 'number' ? value : parseFloat(value);
  const display = isNaN(num) ? '—' : `${(num * 100).toFixed(1)}${suffix}`;
  return (
    <div className={`stat-card ${color}`}>
      <div className="stat-value">{display}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px' }}>
      <div className="mono" style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="mono" style={{ fontSize: 12, color: p.color }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </div>
      ))}
    </div>
  );
};

export default function ModelHealth() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getStatus();
      setStatus(data);
    } catch (e) {
      setError('Cannot reach backend. Is it running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const history = status?.model_info?.retraining_history || [];
  const initialInfo = status?.model_info?.initial_training || {};
  const latestRetrain = history[history.length - 1];

  const chartData = history.map((h, i) => ({
    name: `R${i + 1}`,
    accuracy: h.metrics?.accuracy,
    precision: h.metrics?.precision,
    recall: h.metrics?.recall,
    f1: h.metrics?.f1_score,
  }));

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300, flexDirection: 'column', gap: 16 }}>
      <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      <div className="mono" style={{ color: 'var(--text-dim)', fontSize: 13 }}>Fetching engine status...</div>
    </div>
  );

  if (error) return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Health</h1>
      </div>
      <div className="msg msg-error">{error}</div>
      <button className="btn btn-ghost" onClick={load} style={{ marginTop: 12 }}>Retry</button>
    </div>
  );

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <h1 className="page-title">Model Health</h1>
          <p className="page-subtitle">// Real-time engine status, metrics, and retraining history</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={load}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
          </svg>
          Refresh
        </button>
      </div>

      {/* Quick stats */}
      <div className="stat-grid">
        <div className="stat-card blue">
          <div className="stat-value">{status?.feedback_count ?? 0}</div>
          <div className="stat-label">Feedback Logs</div>
        </div>
        <div className="stat-card green">
          <div className="stat-value">{status?.whitelist_stats?.total_ips ?? 0}</div>
          <div className="stat-label">Whitelisted IPs</div>
        </div>
        <div className="stat-card green">
          <div className="stat-value">{status?.whitelist_stats?.hits ?? 0}</div>
          <div className="stat-label">Whitelist Hits</div>
        </div>
        <div className="stat-card red">
          <div className="stat-value">{status?.blacklist_stats?.total_ips ?? 0}</div>
          <div className="stat-label">Blacklisted IPs</div>
        </div>
        <div className="stat-card red">
          <div className="stat-value">{status?.blacklist_stats?.hits ?? 0}</div>
          <div className="stat-label">Blacklist Hits</div>
        </div>
        <div className="stat-card amber">
          <div className="stat-value">{history.length}</div>
          <div className="stat-label">Retrain Cycles</div>
        </div>
      </div>

      {/* Initial training info */}
      {Object.keys(initialInfo).length > 0 && (
        <div className="card">
          <div className="card-title">
            <div className="pulse-dot" />
            Initial Training Info
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 16 }}>
            {initialInfo.timestamp && (
              <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: 14, border: '1px solid var(--border)' }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>TIMESTAMP</div>
                <div className="mono" style={{ fontSize: 12 }}>{new Date(initialInfo.timestamp).toLocaleString()}</div>
              </div>
            )}
            {initialInfo.samples && (
              <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: 14, border: '1px solid var(--border)' }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>TRAINING SAMPLES</div>
                <div className="mono" style={{ fontSize: 20, fontWeight: 700, color: 'var(--blue)' }}>{initialInfo.samples?.toLocaleString()}</div>
              </div>
            )}
            {initialInfo.epochs && (
              <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: 14, border: '1px solid var(--border)' }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>EPOCHS</div>
                <div className="mono" style={{ fontSize: 20, fontWeight: 700, color: 'var(--blue)' }}>{initialInfo.epochs}</div>
              </div>
            )}
          </div>

          {initialInfo.metrics && (
            <div className="stat-grid">
              <MetricCard label="Accuracy" value={initialInfo.metrics.accuracy} color="green" />
              <MetricCard label="Precision" value={initialInfo.metrics.precision} color="blue" />
              <MetricCard label="Recall" value={initialInfo.metrics.recall} color="amber" />
              <MetricCard label="F1 Score" value={initialInfo.metrics.f1_score} color="green" />
              <MetricCard label="Loss" value={initialInfo.metrics.loss} color="red" suffix="" />
            </div>
          )}
        </div>
      )}

      {/* Latest retrain */}
      {latestRetrain && (
        <div className="card">
          <div className="card-title">Latest Retrain Cycle</div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
            <div className="tag">
              <span style={{ color: 'var(--text-dim)' }}>when: </span>
              <span>{new Date(latestRetrain.timestamp).toLocaleString()}</span>
            </div>
            <div className="tag">
              <span style={{ color: 'var(--text-dim)' }}>samples: </span>
              <span style={{ color: 'var(--blue)' }}>{latestRetrain.feedback_samples}</span>
            </div>
            <div className="tag">
              <span style={{ color: 'var(--text-dim)' }}>epochs: </span>
              <span>{latestRetrain.epochs}</span>
            </div>
            <div className="tag">
              <span style={{ color: 'var(--text-dim)' }}>loss: </span>
              <span style={{ color: 'var(--amber)' }}>{latestRetrain.metrics?.loss}</span>
            </div>
          </div>
          <div className="stat-grid">
            <MetricCard label="Accuracy" value={latestRetrain.metrics?.accuracy} color="green" />
            <MetricCard label="Precision" value={latestRetrain.metrics?.precision} color="blue" />
            <MetricCard label="Recall" value={latestRetrain.metrics?.recall} color="amber" />
            <MetricCard label="F1 Score" value={latestRetrain.metrics?.f1_score} color="green" />
          </div>
        </div>
      )}

      {/* Chart */}
      {chartData.length > 1 && (
        <div className="card">
          <div className="card-title">Retrain History — Metrics Over Time</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--text-dim)' }} />
              <YAxis domain={[0, 1]} tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--text-dim)' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
              <Line type="monotone" dataKey="accuracy" stroke="var(--green)" strokeWidth={2} dot={{ r: 3, fill: 'var(--green)' }} name="Accuracy" />
              <Line type="monotone" dataKey="precision" stroke="var(--blue)" strokeWidth={2} dot={{ r: 3, fill: 'var(--blue)' }} name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="var(--amber)" strokeWidth={2} dot={{ r: 3, fill: 'var(--amber)' }} name="Recall" />
              <Line type="monotone" dataKey="f1" stroke="var(--red)" strokeWidth={2} dot={{ r: 3, fill: 'var(--red)' }} name="F1" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* All retrain history */}
      {history.length > 0 && (
        <div className="card">
          <div className="card-title">Full Retrain History ({history.length} cycles)</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="feature-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Timestamp</th>
                  <th>Samples</th>
                  <th>Loss</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((h, i) => (
                  <tr key={i}>
                    <td className="mono" style={{ color: 'var(--text-dim)' }}>{history.length - i}</td>
                    <td className="mono" style={{ fontSize: 11 }}>{new Date(h.timestamp).toLocaleString()}</td>
                    <td className="mono" style={{ color: 'var(--blue)' }}>{h.feedback_samples}</td>
                    <td className="mono" style={{ color: 'var(--red)' }}>{h.metrics?.loss}</td>
                    <td className="mono" style={{ color: 'var(--green)' }}>{h.metrics?.accuracy !== undefined ? `${(h.metrics.accuracy * 100).toFixed(1)}%` : '—'}</td>
                    <td className="mono">{h.metrics?.precision !== undefined ? `${(h.metrics.precision * 100).toFixed(1)}%` : '—'}</td>
                    <td className="mono">{h.metrics?.recall !== undefined ? `${(h.metrics.recall * 100).toFixed(1)}%` : '—'}</td>
                    <td className="mono" style={{ color: 'var(--green)' }}>{h.metrics?.f1_score !== undefined ? `${(h.metrics.f1_score * 100).toFixed(1)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}