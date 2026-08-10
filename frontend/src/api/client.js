const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, { method = 'GET', body, headers = {}, signal } = {}) {
  const options = { method, headers: { ...headers }, signal };

  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${BASE_URL}${path}`, options);

  if (response.status === 204) return null;

  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json().catch(() => null)
    : await response.text();

  if (!response.ok) {
    const detail =
      (payload && typeof payload === 'object' && payload.detail) ||
      (typeof payload === 'string' && payload) ||
      `Request failed with status ${response.status}`;
    throw new ApiError(
      typeof detail === 'string' ? detail : JSON.stringify(detail),
      response.status,
      payload,
    );
  }

  return payload;
}

const toQuery = (params) => {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    if (Array.isArray(value)) value.forEach((v) => search.append(key, v));
    else search.append(key, value);
  });
  const qs = search.toString();
  return qs ? `?${qs}` : '';
};

export const api = {
  health: () => request('/api/health'),
  stats: () => request('/api/stats'),

  // --- papers ---------------------------------------------------------------
  listPapers: (params) => request(`/api/papers${toQuery(params)}`),
  getPaper: (id) => request(`/api/papers/${id}`),
  getPaperChunks: (id, params) => request(`/api/papers/${id}/chunks${toQuery(params)}`),
  deletePaper: (id) => request(`/api/papers/${id}`, { method: 'DELETE' }),
  reindexPaper: (id) => request(`/api/papers/${id}/reindex`, { method: 'POST' }),
  paperFileUrl: (id) => `${BASE_URL}/api/papers/${id}/file`,
  uploadPapers: (files, signal) => {
    const form = new FormData();
    Array.from(files).forEach((file) => form.append('files', file));
    return request('/api/papers/upload', { method: 'POST', body: form, signal });
  },
  search: (payload) => request('/api/papers/search', { method: 'POST', body: payload }),

  // --- agents ---------------------------------------------------------------
  ask: (payload) => request('/api/agents/ask', { method: 'POST', body: payload }),
  summarize: (payload) => request('/api/agents/summarize', { method: 'POST', body: payload }),
  extract: (payload) => request('/api/agents/extract', { method: 'POST', body: payload }),
  gaps: (payload) => request('/api/agents/gaps', { method: 'POST', body: payload }),
  buildGraph: (payload) => request('/api/agents/graph/build', { method: 'POST', body: payload }),
  review: (payload) => request('/api/agents/review', { method: 'POST', body: payload }),
  listRuns: (params) => request(`/api/agents/runs${toQuery(params)}`),

  // --- graph ----------------------------------------------------------------
  getGraph: (params) => request(`/api/graph${toQuery(params)}`),
  graphStats: () => request('/api/graph/stats'),
  clearGraph: () => request('/api/graph', { method: 'DELETE' }),

  // --- reports --------------------------------------------------------------
  createReport: (payload) => request('/api/reports', { method: 'POST', body: payload }),
  listReports: () => request('/api/reports'),
  getReport: (id) => request(`/api/reports/${id}`),
  deleteReport: (id) => request(`/api/reports/${id}`, { method: 'DELETE' }),
  reportPdfUrl: (id) => `${BASE_URL}/api/reports/${id}/pdf`,
  reportMarkdownUrl: (id) => `${BASE_URL}/api/reports/${id}/markdown`,
};

export default api;
