const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.static('public'));
app.use(express.json());

// This endpoint is called by the browser on page load
app.post('/check-visitor', async (req, res) => {
  const clientIp =
    req.headers['x-forwarded-for']?.split(',')[0].trim() ||
    req.socket.remoteAddress?.replace('::ffff:', '') ||
    '0.0.0.0';

  const now = new Date();
  const timestamp = `${String(now.getDate()).padStart(2,'0')}-${String(now.getMonth()+1).padStart(2,'0')}-${now.getFullYear()} ${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;

  const log = {
    timestamp,
    src_ip: clientIp,
    dst_ip: '192.168.1.7',      // your laptop IP
    src_port: req.body.src_port || 50000,
    dst_port: 3001,              // demo site port
    protocol: 6,                 // TCP
    bytes_sent: req.body.bytes_sent || 512,
    bytes_received: req.body.bytes_received || 2048,
    flags: 24,                   // PSH-ACK (normal web traffic)
    duration: req.body.duration || 0.3,
  };

  try {
    const result = await axios.post('http://localhost:8000/classify', log);
    res.json({ log, result: result.data });
  } catch (e) {
    res.status(500).json({ error: 'Engine unreachable' });
  }
});

app.listen(3001, '0.0.0.0', () => {
  console.log('Demo site running at http://0.0.0.0:3001');
});