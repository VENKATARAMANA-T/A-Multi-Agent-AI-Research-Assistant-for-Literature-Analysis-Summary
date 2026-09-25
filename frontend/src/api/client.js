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
  // Returns a job to follow, not the finished result — see streamJob below.
  uploadPapers: (files, signal) => {
    const form = new FormData();
    Array.from(files).forEach((file) => form.append('files', file));
    return request('/api/papers/upload', { method: 'POST', body: form, signal });
  },
  uploadPapersAndWait: (files, signal) => {
    const form = new FormData();
    Array.from(files).forEach((file) => form.append('files', file));
    return request('/api/papers/upload?wait=true', { method: 'POST', body: form, signal });
  },
  search: (payload) => request('/api/papers/search', { method: 'POST', body: payload }),

  // --- reader ---------------------------------------------------------------
  highlight: (params) => request(`/api/reader/highlight${toQuery(params)}`),
  explain: (payload) => request('/api/reader/explain', { method: 'POST', body: payload }),
  citation: (id, style) => request(`/api/reader/citation/${id}${toQuery({ style })}`),
  bibliographyUrl: (params) => `${BASE_URL}/api/reader/citations${toQuery({ ...params, download: true })}`,

  // --- comparison matrix ----------------------------------------------------
  runMatrix: (payload) => request('/api/matrix', { method: 'POST', body: payload }),
  listMatrixRuns: () => request('/api/matrix'),
  getMatrixRun: (id) => request(`/api/matrix/${id}`),
  deleteMatrixRun: (id) => request(`/api/matrix/${id}`, { method: 'DELETE' }),
  matrixCsvUrl: (id) => `${BASE_URL}/api/matrix/${id}/csv`,

  // --- discovery ------------------------------------------------------------
  discoverRelated: (params) => request(`/api/discover/related${toQuery(params)}`),
  discoverSearch: (params) => request(`/api/discover/search${toQuery(params)}`),
  discoverGaps: (params) => request(`/api/discover/gaps${toQuery(params)}`),

  // --- literature-based discovery -------------------------------------------
  lbdTerms: (params) => request(`/api/lbd/terms${toQuery(params)}`),
  runLbd: (payload) => request('/api/lbd', { method: 'POST', body: payload }),
  runClosedLbd: (payload) => request('/api/lbd/closed', { method: 'POST', body: payload }),
  listHypotheses: (params) => request(`/api/lbd/hypotheses${toQuery(params)}`),
  starHypothesis: (id, starred) =>
    request(`/api/lbd/hypotheses/${id}/star${toQuery({ starred })}`, { method: 'POST' }),
  deleteHypothesis: (id) => request(`/api/lbd/hypotheses/${id}`, { method: 'DELETE' }),

  // --- conversations --------------------------------------------------------
  listConversations: () => request('/api/agents/conversations'),
  getConversation: (id) => request(`/api/agents/conversations/${id}`),
  deleteConversation: (id) => request(`/api/agents/conversations/${id}`, { method: 'DELETE' }),

  // --- figures --------------------------------------------------------------
  listFigures: (params) => request(`/api/figures${toQuery(params)}`),
  getFigure: (id) => request(`/api/figures/${id}`),
  figureImageUrl: (id) => `${BASE_URL}/api/figures/${id}/image`,
  figureEstimate: (params) => request(`/api/figures/estimate${toQuery(params)}`),
  analyseFigures: (payload) => request('/api/figures/analyse', { method: 'POST', body: payload }),

  // --- jobs -----------------------------------------------------------------
  listJobs: (params) => request(`/api/jobs${toQuery(params)}`),
  getJob: (id) => request(`/api/jobs/${id}`),
  streamJob: (id) => new EventSource(`${BASE_URL}/api/jobs/${id}/stream`),

  // --- cache ----------------------------------------------------------------
  cacheStats: () => request('/api/cache'),
  clearCache: () => request('/api/cache', { method: 'DELETE' }),

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
