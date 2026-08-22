import React, { useState } from 'react';
import { submitFeedback, retrain } from '../utils/api';

const DEFAULT_LOG = {
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

const FIELDS = [
  ['timestamp', 'Timestamp'],
  ['src_ip', 'Source IP'],
  ['dst_ip', 'Destination IP'],
  ['src_port', 'Src Port'],
  ['dst_port', 'Dst Port'],
  ['protocol', 'Protocol'],
  ['bytes_sent', 'Bytes Sent'],
  ['bytes_received', 'Bytes Received'],
  ['flags', 'TCP Flags'],
  ['duration', 'Duration (s)'],
];

export default function FeedbackPanel() {
  const [form, setForm] = useState(DEFAULT_LOG);
  const [action, setAction] = useState(null); // 0 = allow, 1 = flagged
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [msg, setMsg] = useState('');
  const [isError, setIsError] = useState(false);

  const showMsg = (text, err = false) => {
    setMsg(text);
    setIsError(err);
    setTimeout(() => setMsg(''), 4000);
  };

  const onChange = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const onSubmit = async () => {
    if (action === null) {
      showMsg('Please select the correct action (Allow or Reject).', true);
      return;
    }
    setLoading(true);
    try {
      const log = {
        ...form,
        src_port: Number(form.src_port),
        dst_port: Number(form.dst_port),
        protocol: Number(form.protocol),
        bytes_sent: Number(form.bytes_sent),
        bytes_received: Number(form.bytes_received),
        flags: Number(form.flags),
        duration: parseFloat(form.duration),
      };
      const res = await submitFeedback({ log, action });
      showMsg(res.message || 'Feedback submitted and model retrained!');
      setAction(null);
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Failed to submit feedback.', true);
    } finally {
      setLoading(false);
    }
  };

  const onRetrain = async () => {
    setRetraining(true);
    try {
      const res = await retrain();
      showMsg(res.message || 'Model retrained successfully!');
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Retraining failed.', true);
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Feedback & Retraining</h1>
        <p className="page-subtitle">// Correct model decisions and retrain with EWC continual learning</p>
      </div>

      <div style={{ background: 'var(--blue-dim)', border: '1px solid #38bdf830', borderRadius: 10, padding: '14px 18px', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ color: 'var(--blue)', fontSize: 20 }}>⚡</div>
        <div>
          <div style={{ fontFamily: 'Syne', fontSize: 14, fontWeight: 700, color: 'var(--blue)', marginBottom: 2 }}>Continual Learning via EWC</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.5 }}>
            Each feedback correction is logged and used to retrain the model using Elastic Weight Consolidation — so the model learns without forgetting past knowledge.
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <path d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
          </svg>
          Log Entry (Paste the misclassified log here)
        </div>

        <div className="field-grid">
          {FIELDS.map(([key, label]) => (
            <div className="field" key={key}>
              <label>{label}</label>
              <input
                value={form[key]}
                onChange={e => onChange(key, e.target.value)}
              />
            </div>
          ))}
        </div>

        <hr className="divider" />

        <div style={{ marginBottom: 16 }}>
          <div className="card-title" style={{ marginBottom: 10 }}>What should the correct decision be?</div>
          <div className="feedback-select">
            <button
              className={`option-btn allow ${action === 0 ? 'selected' : ''}`}
              onClick={() => setAction(0)}
            >
              <div style={{ fontSize: 20, marginBottom: 4 }}>✓</div>
              <div>ALLOW</div>
              <div style={{ fontSize: 11, marginTop: 4, opacity: 0.7 }}>This traffic is safe</div>
            </button>
            <button
              className={`option-btn reject ${action === 1 ? 'selected' : ''}`}
              onClick={() => setAction(1)}
            >
              <div style={{ fontSize: 20, marginBottom: 4 }}>✗</div>
              <div>REJECT</div>
              <div style={{ fontSize: 11, marginTop: 4, opacity: 0.7 }}>This traffic is malicious</div>
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn btn-primary" onClick={onSubmit} disabled={loading || action === null}>
            {loading ? <div className="spinner" /> : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
              </svg>
            )}
            {loading ? 'Submitting...' : 'Submit Feedback & Retrain'}
          </button>
        </div>

        {msg && (
          <div className={`msg ${isError ? 'msg-error' : 'msg-success'}`} style={{ marginTop: 12 }}>
            {msg}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
          </svg>
          Manual Retrain Trigger
        </div>
        <p style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-dim)', marginBottom: 16, lineHeight: 1.6 }}>
          Trigger a full retrain on all accumulated feedback logs. This runs 10 epochs with EWC loss (BCE + Fisher penalty) to preserve prior knowledge.
        </p>
        <button className="btn btn-ghost" onClick={onRetrain} disabled={retraining}>
          {retraining ? <div className="spinner" /> : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
              <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
              <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
            </svg>
          )}
          {retraining ? 'Retraining...' : 'Retrain Model Now'}
        </button>
      </div>

      <div className="card">
        <div className="card-title">EWC Retraining Pipeline</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0, position: 'relative' }}>
          {[
            { icon: '📥', label: 'Load Feedback', desc: 'Reads feedback_log.json — each entry is a log + correct label' },
            { icon: '⚖️', label: 'EWC Loss = BCE + λ·Σ Fᵢ(θᵢ − θᵢ*)²', desc: 'Fisher Information Matrix penalizes drift from important past weights' },
            { icon: '🔁', label: 'Train 10 Epochs', desc: 'Adam optimizer with lr=0.001 on feedback data only' },
            { icon: '💾', label: 'Save Model', desc: 'Updated model_with_ewc.pt saved with new state_dict, fisher, opt_params' },
            { icon: '📊', label: 'Log Metrics', desc: 'Accuracy, precision, recall, F1 evaluated on test_set.csv and stored in metadata.json' },
          ].map((s, i, arr) => (
            <div key={i} style={{ display: 'flex', gap: 16, position: 'relative', paddingBottom: i < arr.length - 1 ? 20 : 0 }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div style={{ width: 36, height: 36, background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, flexShrink: 0 }}>
                  {s.icon}
                </div>
                {i < arr.length - 1 && (
                  <div style={{ width: 1, flex: 1, background: 'var(--border)', marginTop: 4 }} />
                )}
              </div>
              <div style={{ paddingTop: 6, paddingBottom: i < arr.length - 1 ? 0 : 0 }}>
                <div style={{ fontFamily: 'JetBrains Mono', fontSize: 13, fontWeight: 600, color: 'var(--green)', marginBottom: 2 }}>{s.label}</div>
                <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.5 }}>{s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}