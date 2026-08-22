import React, { useState } from 'react';
import { addWhitelistIp, removeWhitelistIp, addWhitelistCountry, removeWhitelistCountry } from '../utils/api';

function ListManager({ title, icon, placeholder, onAdd, onRemove, color = 'green', hint }) {
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
      const res = await onAdd(val);
      if (!items.includes(val)) setItems(p => [...p, val]);
      setInput('');
      showMsg(res.message || `Added: ${val}`);
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Failed to add.', true);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (item) => {
    try {
      const res = await onRemove(item);
      setItems(p => p.filter(x => x !== item));
      showMsg(res.message || `Removed: ${item}`);
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Failed to remove.', true);
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span style={{ fontSize: 14 }}>{icon}</span>
        {title}
      </div>

      {hint && (
        <div className="mono" style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 14, padding: '8px 12px', background: 'var(--surface2)', borderRadius: 6, border: '1px solid var(--border)' }}>
          {hint}
        </div>
      )}

      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div className="field" style={{ flex: 1 }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={placeholder}
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
          />
        </div>
        <button className="btn btn-primary" onClick={handleAdd} disabled={loading || !input.trim()}>
          {loading ? <div className="spinner" /> : '+ Add'}
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
            Added this session ({items.length})
          </div>
          {items.map((item, i) => (
            <div className="list-item" key={i}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: color === 'green' ? 'var(--green)' : 'var(--red)' }} />
                <span className="list-item-text">{item}</span>
              </div>
              <button className="btn btn-danger btn-sm" onClick={() => handleRemove(item)}>
                Remove
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '20px 0', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono', fontSize: 12 }}>
          No entries added this session
        </div>
      )}
    </div>
  );
}

export default function WhitelistManager() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Whitelist Manager</h1>
        <p className="page-subtitle">// Trusted IPs and countries bypass the AI engine entirely</p>
      </div>

      <div style={{ background: 'var(--green-dim)', border: '1px solid #00e5a030', borderRadius: 10, padding: '14px 18px', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ color: 'var(--green)', fontSize: 20 }}>ℹ</div>
        <div>
          <div style={{ fontFamily: 'Syne', fontSize: 14, fontWeight: 700, color: 'var(--green)', marginBottom: 2 }}>Instant ALLOW</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.5 }}>
            Whitelisted sources are immediately allowed without invoking the ML model or blacklist checks. Use for trusted internal networks.
          </div>
        </div>
      </div>

      <ListManager
        title="IP Whitelist"
        icon="🌐"
        placeholder="e.g. 192.168.1.100"
        onAdd={addWhitelistIp}
        onRemove={removeWhitelistIp}
        color="green"
        hint="IPv4 format: e.g. 192.168.1.10 — Exact match against src_ip"
      />

      <ListManager
        title="Country Whitelist"
        icon="🗺"
        placeholder="e.g. IN, US, GB"
        onAdd={addWhitelistCountry}
        onRemove={removeWhitelistCountry}
        color="green"
        hint="ISO Alpha-2 country codes. GeoIP lookup maps src_ip → country code."
      />
    </div>
  );
}