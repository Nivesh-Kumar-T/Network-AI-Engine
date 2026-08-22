import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import ClassifyLog from './pages/ClassifyLog';
import CsvClassifier from './pages/CsvClassifier';
import WhitelistManager from './pages/WhitelistManager';
import BlacklistManager from './pages/BlacklistManager';
import FeedbackPanel from './pages/FeedbackPanel';
import ModelHealth from './pages/ModelHealth';
import LiveMonitor from './pages/LiveMonitor';
import './index.css';
import './App.css';

const PAGES = {
  classify: ClassifyLog,
  csv: CsvClassifier,
  whitelist: WhitelistManager,
  blacklist: BlacklistManager,
  feedback: FeedbackPanel,
  health: ModelHealth,
  live: LiveMonitor,
};

export default function App() {
  const [page, setPage] = useState('classify');
  const PageComponent = PAGES[page] || ClassifyLog;

  return (
    <div className="app-shell">
      <Sidebar activePage={page} onNavigate={setPage} />
      <main className="main-content">
        <PageComponent />
      </main>
    </div>
  );
}