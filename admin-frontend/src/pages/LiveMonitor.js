import React, { useState, useEffect, useRef } from 'react';
import { getLogs, clearLogs } from '../utils/api';

const PROTOCOL_MAP = { 6: 'TCP', 17: 'UDP', 1: 'ICMP' };
const PORT_MAP = { 80: 'HTTP', 443: 'HTTPS', 22: 'SSH', 23: 'Telnet', 3389: 'RDP', 3001: 'Demo Site', 8000: 'API' };

function timeSince(isoString) {
  const diff = Math.floor((Date.now() - new Date(isoString)) / 1000);
  if (diff < 5) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function formatBytes(b) {
  if (b < 1024) return `${b}B`;
  return `${(b / 1024).toFixed(1)}KB`;
}

export default function LiveMonitor() {
  const [logs, setLogs] = useState([]);
  const [paused, setPaused] = useState(false);
  const [filter, setFilter] = useState('ALL'); // ALL, ALLOW, REJECT
  const [selected, setSelected] = useState(null);
  const [clearing, setClearing] = useState(false);
  const intervalRef = useRef(null);
  const prevCountRef = useRef(0);
  const [newCount, setNewCount] = useState(0);

  const fetchLogs = async () => {
    if (paused) return;
    try {
      const data = await getLogs();
      const incoming = data.logs || [];
      if (incoming.length > prevCountRef.current) {
        setNewCount(incoming.length - prevCountRef.current);
        setTimeout(() => setNewCount(0), 2000);
      }
      prevCountRef.current = incoming.length;
      setLogs(incoming);
    } catch (e) {
      // backend not reachable
    }
  };

  useEffect(() => {
    fetchLogs();
    intervalRef.current = setInterval(fetchLogs, 2000);
    return () => clearInterval(intervalRef.current);
  // eslint-disable-next-line
  }, [paused]);

  const handleClear = async () => {
    setClearing(true);
    try {
      await clearLogs();
      setLogs([]);
      prevCountRef.current = 0;
      setSelected(null);
    } catch (e) {}
    setClearing(false);
  };

  const filtered = filter === 'ALL' ? logs : logs.filter(l => l.decision === filter);
  const allowCount = logs.filter(l => l.decision === 'ALLOW').length;
  const rejectCount = logs.filter(l => l.decision === 'REJECT').length;

  return (
    <div style={{ display: 'flex', gap: 20, height: 'calc(100vh - 80px)' }}>

      {/* Left — log feed */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>

        <div className="page-header" style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                Live Traffic Monitor
                {!paused && <div className="pulse-dot" />}
              </h1>
              <p className="page-subtitle">// Auto-refreshes every 2 seconds — click any row to inspect</p>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className={`btn btn-sm ${paused ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setPaused(p => !p)}
              >
                {paused ? (
                  <><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="13" height="13"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume</>
                ) : (
                  <><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="13" height="13"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause</>
                )}
              </button>
              <button className="btn btn-danger btn-sm" onClick={handleClear} disabled={clearing}>
                {clearing ? <div className="spinner" style={{ width: 12, height: 12 }} /> : 'Clear'}
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div style={{ display: 'flex', gap: 12, marginTop: 14, flexWrap: 'wrap' }}>
            {[
              { label: 'Total', val: logs.length, color: 'var(--text)', filter: 'ALL' },
              { label: 'Allowed', val: allowCount, color: 'var(--green)', filter: 'ALLOW' },
              { label: 'Rejected', val: rejectCount, color: 'var(--red)', filter: 'REJECT' },
            ].map(s => (
              <button
                key={s.filter}
                onClick={() => setFilter(s.filter)}
                style={{
                  background: filter === s.filter ? 'var(--surface2)' : 'transparent',
                  border: `1px solid ${filter === s.filter ? 'var(--border-bright)' : 'var(--border)'}`,
                  borderRadius: 8,
                  padding: '8px 16px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  transition: 'all 0.15s',
                }}
              >
                <span style={{ fontFamily: 'Syne', fontSize: 20, fontWeight: 800, color: s.color }}>{s.val}</span>
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{s.label}</span>
              </button>
            ))}

            {newCount > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px', background: 'var(--green-dim)', border: '1px solid #00e5a030', borderRadius: 8 }}>
                <div className="pulse-dot" />
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--green)' }}>
                  +{newCount} new
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Log rows */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {filtered.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono', fontSize: 13 }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📡</div>
              {logs.length === 0
                ? 'Waiting for traffic... Open the demo site on your phone'
                : `No ${filter} entries`}
            </div>
          ) : filtered.map((log, i) => {
            const isAllow = log.decision === 'ALLOW';
            const isSelected = selected === i;
            return (
              <div
                key={i}
                onClick={() => setSelected(isSelected ? null : i)}
                style={{
                  background: isSelected ? 'var(--surface2)' : 'var(--surface)',
                  border: `1px solid ${isSelected ? (isAllow ? '#00e5a040' : '#ff4b6e40') : 'var(--border)'}`,
                  borderLeft: `3px solid ${isAllow ? 'var(--green)' : 'var(--red)'}`,
                  borderRadius: 8,
                  padding: '12px 16px',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 14,
                  flexWrap: 'wrap',
                }}
              >
                {/* Decision */}
                <span className={`badge ${isAllow ? 'badge-allow' : 'badge-reject'}`} style={{ minWidth: 72, justifyContent: 'center', fontSize: 11 }}>
                  {isAllow ? '✓' : '✗'} {log.decision}
                </span>

                {/* IP */}
                <span className="mono" style={{ fontSize: 13, minWidth: 110, color: 'var(--text)' }}>
                  {log.src_ip}
                </span>

                {/* Arrow */}
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>→</span>

                {/* Port / protocol */}
                <span className="mono" style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                  :{log.dst_port}
                  {PORT_MAP[log.dst_port] ? ` (${PORT_MAP[log.dst_port]})` : ''}
                </span>

                <span className="mono" style={{ fontSize: 12, color: 'var(--blue)' }}>
                  {PROTOCOL_MAP[log.protocol] || `Proto:${log.protocol}`}
                </span>

                {/* Bytes */}
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                  ↑{formatBytes(log.bytes_sent)} ↓{formatBytes(log.bytes_received)}
                </span>

                {/* Confidence */}
                {log.confidence !== null && log.confidence !== undefined && (
                  <span className="mono" style={{ fontSize: 11, color: isAllow ? 'var(--green)' : 'var(--red)' }}>
                    {(log.confidence * 100).toFixed(1)}%
                  </span>
                )}

                {/* Time */}
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
                  {timeSince(log.timestamp)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right — detail panel */}
      <div style={{ width: 300, flexShrink: 0, overflowY: 'auto' }}>
        {selected !== null && filtered[selected] ? (
          <DetailPanel log={filtered[selected]} />
        ) : (
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: 24, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="40" height="40">
              <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12 }}>Click any log row to inspect details</div>
          </div>
        )}
      </div>
    </div>
  );
}

function DetailPanel({ log }) {
  const isAllow = log.decision === 'ALLOW';
  const exp = log.explanation;

  const PROTOCOL_MAP = { 6: 'TCP', 17: 'UDP', 1: 'ICMP' };
  const PORT_MAP = {
    80: 'HTTP', 443: 'HTTPS', 22: 'SSH', 23: 'Telnet',
    3389: 'RDP', 3001: 'Demo Site', 8000: 'FastAPI', 5900: 'VNC'
  };

  const flagNames = (flags) => {
    const combos = { 18: 'SYN-ACK', 24: 'PSH-ACK', 16: 'ACK', 4: 'RST', 2: 'SYN', 1: 'FIN', 20: 'RST-ACK' };
    if (combos[flags]) return combos[flags];
    const map = { 1: 'FIN', 2: 'SYN', 4: 'RST', 8: 'PSH', 16: 'ACK', 32: 'URG' };
    return Object.entries(map).filter(([bit]) => flags & bit).map(([, n]) => n).join('-') || 'None';
  };

  const keyFactors = exp?.key_factors || [];
  const allFeatures = exp?.all_features || [];
  const maxImp = Math.max(...allFeatures.map(f => f.importance || 0), 0.0001);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* ── Verdict header ── */}
      <div style={{
        background: 'var(--surface)',
        border: `1px solid ${isAllow ? '#00e5a025' : '#ff4b6e25'}`,
        borderRadius: 12, padding: 20, textAlign: 'center'
      }}>
        <div style={{ fontSize: 36, marginBottom: 8 }}>{isAllow ? '✅' : '🚫'}</div>
        <div style={{
          fontFamily: 'Syne', fontSize: 20, fontWeight: 800,
          color: isAllow ? 'var(--green)' : 'var(--red)', marginBottom: 4
        }}>
          {log.decision}
        </div>
        {log.confidence != null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center', marginTop: 8 }}>
            <div style={{ flex: 1, maxWidth: 160, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 3,
                width: `${(log.confidence * 100).toFixed(1)}%`,
                background: isAllow ? 'var(--green)' : 'var(--red)',
                transition: 'width 0.5s ease'
              }} />
            </div>
            <span className="mono" style={{ fontSize: 12, color: isAllow ? 'var(--green)' : 'var(--red)' }}>
              {(log.confidence * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      {/* ── Reason ── */}
      <div style={{
        background: isAllow ? 'var(--green-dim)' : 'var(--red-dim)',
        border: `1px solid ${isAllow ? '#00e5a025' : '#ff4b6e25'}`,
        borderRadius: 10, padding: '12px 14px',
        fontFamily: 'JetBrains Mono', fontSize: 12,
        color: isAllow ? 'var(--green)' : 'var(--red)', lineHeight: 1.6
      }}>
        {log.reason}
      </div>

      {/* ── AI Summary ── */}
      {exp?.summary && (
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 8 }}>AI Summary</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.6 }}>
            {exp.summary}
          </div>
        </div>
      )}

      {/* ── Top 3 key factors ── */}
      {keyFactors.length > 0 && (
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Top Decision Factors</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {keyFactors.map((f, i) => (
              <div key={i} style={{
                background: 'var(--surface2)', border: '1px solid var(--border)',
                borderRadius: 8, padding: 12
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    {f.feature}
                  </span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--green)' }}>
                    {f.importance?.toFixed(4)}
                  </span>
                </div>
                <div className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                  {f.value}
                </div>
                {f.description && (
                  <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
                    {f.description}
                  </div>
                )}
                <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', background: isAllow ? 'var(--green)' : 'var(--red)',
                    borderRadius: 2,
                    width: `${((f.importance || 0) / maxImp * 100).toFixed(0)}%`
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── All features table ── */}
      {allFeatures.length > 0 && (
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>All Feature Analysis</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {allFeatures.map((f, i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '1fr auto',
                alignItems: 'center', gap: 8,
                padding: '9px 0',
                borderBottom: i < allFeatures.length - 1 ? '1px solid var(--border)' : 'none'
              }}>
                <div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>
                    {f.feature}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="mono" style={{ fontSize: 12, color: 'var(--text)' }}>{f.value}</span>
                    {f.description && (
                      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>— {f.description}</span>
                    )}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 80 }}>
                  <div style={{ width: 50, height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', background: 'var(--green)', borderRadius: 2,
                      width: `${((f.importance || 0) / maxImp * 100).toFixed(0)}%`
                    }} />
                  </div>
                  <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 36 }}>
                    {f.importance?.toFixed(4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Connection details ── */}
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: 16 }}>
        <div className="card-title" style={{ marginBottom: 10 }}>Connection Details</div>
        {[
          ['Time', new Date(log.timestamp).toLocaleString()],
          ['Source IP', log.src_ip],
          ['Source Port', log.src_port],
          ['Destination IP', log.dst_ip],
          ['Destination Port', `${log.dst_port}${PORT_MAP[log.dst_port] ? ' (' + PORT_MAP[log.dst_port] + ')' : ''}`],
          ['Protocol', `${log.protocol} — ${PROTOCOL_MAP[log.protocol] || 'Unknown'}`],
          ['TCP Flags', `${log.flags} (${flagNames(log.flags)})`],
          ['Bytes Sent', `${log.bytes_sent} B`],
          ['Bytes Received', `${log.bytes_received} B`],
          ['Duration', `${log.duration}s`],
        ].map(([label, val]) => (
          <div key={label} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
            padding: '7px 0', borderBottom: '1px solid var(--border)'
          }}>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', flexShrink: 0 }}>
              {label}
            </span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text)', textAlign: 'right', marginLeft: 12 }}>
              {val}
            </span>
          </div>
        ))}
      </div>

    </div>
  );
}