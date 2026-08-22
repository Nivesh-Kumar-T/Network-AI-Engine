import React, { useState } from 'react';
import './Sidebar.css';

const NAV = [
  {
    id: 'classify',
    label: 'Classify Log',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
      </svg>
    ),
  },
  {
    id: 'csv',
    label: 'CSV Classifier',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
    ),
  },
  {
    id: 'whitelist',
    label: 'Whitelist',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M5 13l4 4L19 7"/>
      </svg>
    ),
  },
  {
    id: 'blacklist',
    label: 'Blacklist',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
      </svg>
    ),
  },
  {
    id: 'feedback',
    label: 'Feedback',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
      </svg>
    ),
  },
  {
    id: 'health',
    label: 'Model Health',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
  id: 'live',
  label: 'Live Monitor',
  icon: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  ),
},
];

export default function Sidebar({ activePage, onNavigate }) {
  const [open, setOpen] = useState(false);

  const handleNav = (id) => {
    onNavigate(id);
    setOpen(false);
  };

  return (
    <>
      {/* Mobile top bar */}
      <div className="hamburger">
        <button className="hamburger-btn" onClick={() => setOpen(o => !o)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="22" height="22">
            <line x1="3" y1="6" x2="21" y2="6"/>
            <line x1="3" y1="12" x2="21" y2="12"/>
            <line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        <span className="hamburger-title">NetGuard AI</span>
      </div>

      {/* Overlay */}
      <div className={`mobile-overlay ${open ? 'open' : ''}`} onClick={() => setOpen(false)} />

      {/* Sidebar */}
      <nav className={`sidebar ${open ? 'open' : ''}`}>
        <div className="sidebar-logo">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div>
            <div className="logo-title">NetGuard AI</div>
            <div className="logo-sub">ADMIN CONSOLE</div>
          </div>
        </div>

        <div className="nav-section-label">ENGINE</div>

        <ul className="nav-list">
          {NAV.map(item => (
            <li key={item.id}>
              <button
                className={`nav-item ${activePage === item.id ? 'active' : ''}`}
                onClick={() => handleNav(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                {activePage === item.id && <span className="nav-active-dot" />}
              </button>
            </li>
          ))}
        </ul>

        <div className="sidebar-footer">
          <div className="status-row">
            <div className="pulse-dot" />
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              Backend: 8000
            </span>
          </div>
        </div>
      </nav>
    </>
  );
}