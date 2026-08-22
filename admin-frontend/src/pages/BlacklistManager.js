import React, { useState } from 'react';
import { addBlacklistIp, removeBlacklistIp } from '../utils/api';

export default function BlacklistManager() {
  const [input, setInput] = useState('');
  const [items, setItems] = useState([]);
  const [msg, setMsg] = useState('');
  const [isError, setIsError] = useState(false);
  const [loading, setLoading] = useState(false);

  const showMsg = (text, err = false) => {
    setMsg(text);
    setIsError(err);
    setTimeout(() => setMsg(''), 3000);
  };

  const handleAdd = async () => {
    const val = input.trim();
    if (!val) return;
    setLoading(true);
    try {
      const res = await addBlacklistIp(val);
      if (!items.includes(val)) setItems(p => [...p, val]);
      setInput('');
      showMsg(res.message || `Blocked: ${val}`);
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Failed to add.', true);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (item) => {
    try {
      const res = await removeBlacklistIp(item);
      setItems(p => p.filter(x => x !== item));
      showMsg(res.message || `Removed: ${item}`);
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Failed to remove.', true);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Blacklist Manager</h1>
        <p className="page-subtitle">// Instantly block known malicious IPs — no AI needed</p>
      </div>

      <div style={{ background: 'var(--red-dim)', border: '1px solid #ff4b6e30', borderRadius: 10, padding: '14px 18px', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ color: 'var(--red)', fontSize: 20 }}>⚠</div>
        <div>
          <div style={{ fontFamily: 'Syne', fontSize: 14, fontWeight: 700, color: 'var(--red)', marginBottom: 2 }}>Instant REJECT</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.5 }}>
            Blacklisted IPs are immediately rejected at the first check — before whitelist, before AI. Use for known attackers, abuse IPs, or threat intel feeds.
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
          </svg>
          Block an IP Address
        </div>

        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', marginBottom: 14, padding: '8px 12px', background: 'var(--surface2)', borderRadius: 6, border: '1px solid var(--border)' }}>
          IPv4 format only — e.g. 203.0.113.5 — Exact match against src_ip in every log
        </div>

        <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
          <div className="field" style={{ flex: 1 }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="e.g. 203.0.113.5"
              onKeyDown={e => e.key === 'Enter' && handleAdd()}
              style={{ borderColor: input ? '#ff4b6e' : undefined }}
            />
          </div>
          <button
            className="btn"
            onClick={handleAdd}
            disabled={loading || !input.trim()}
            style={{ background: 'var(--red)', color: '#fff' }}
          >
            {loading ? <div className="spinner" style={{ borderTopColor: '#fff' }} /> : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
              </svg>
            )}
            Block IP
          </button>
        </div>

        {msg && (
          <div className={`msg ${isError ? 'msg-error' : 'msg-success'}`} style={{ marginBottom: 12 }}>
            {msg}
          </div>
        )}

        {items.length > 0 ? (
          <div>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
              Blocked this session ({items.length})
            </div>
            {items.map((item, i) => (
              <div className="list-item" key={i} style={{ borderColor: '#ff4b6e20' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--red)' }} />
                  <span className="list-item-text mono">{item}</span>
                  <span className="badge badge-reject" style={{ fontSize: 10, padding: '2px 8px' }}>BLOCKED</span>
                </div>
                <button className="btn btn-ghost btn-sm" onClick={() => handleRemove(item)}>
                  Unblock
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono', fontSize: 12 }}>
            No IPs blocked this session
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">How the Blacklist Works</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
          {[
            { step: '01', label: 'Log Arrives', desc: 'Network log hits /classify endpoint' },
            { step: '02', label: 'Blacklist Check', desc: 'src_ip checked against ip_list.txt instantly' },
            { step: '03', label: 'Instant REJECT', desc: 'No ML inference — immediate block response' },
            { step: '04', label: 'Counter Updated', desc: 'Blacklist hit counter incremented in metadata' },
          ].map(s => (
            <div key={s.step} style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: 14 }}>
              <div className="mono" style={{ fontSize: 20, fontWeight: 700, color: 'var(--red)', marginBottom: 6 }}>{s.step}</div>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{s.label}</div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.5 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}