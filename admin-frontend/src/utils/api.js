import axios from 'axios';

const BASE = 'http://localhost:8000';

export const api = axios.create({ baseURL: BASE });

export const getLogs = () => api.get('/logs').then(r => r.data);
export const clearLogs = () => api.delete('/logs').then(r => r.data);
export const classifyLog = (data) => api.post('/classify', data).then(r => r.data);
export const submitFeedback = (data) => api.post('/feedback', data).then(r => r.data);
export const retrain = () => api.post('/retrain').then(r => r.data);
export const getStatus = () => api.get('/status').then(r => r.data);

// Whitelist
export const addWhitelistIp = (item) => api.post('/whitelist/ip/add', { item }).then(r => r.data);
export const removeWhitelistIp = (item) => api.post('/whitelist/ip/remove', { item }).then(r => r.data);
export const addWhitelistCountry = (item) => api.post('/whitelist/country/add', { item }).then(r => r.data);
export const removeWhitelistCountry = (item) => api.post('/whitelist/country/remove', { item }).then(r => r.data);

// Blacklist
export const addBlacklistIp = (item) => api.post('/blacklist/ip/add', { item }).then(r => r.data);
export const removeBlacklistIp = (item) => api.post('/blacklist/ip/remove', { item }).then(r => r.data);