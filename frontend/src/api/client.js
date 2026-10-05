const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const TOKEN_KEY = 'researchcompass.token';

// Held in memory as well as localStorage so a request issued before the context
// has mounted still carries the token.
let accessToken = null;
try {
  accessToken = localStorage.getItem(TOKEN_KEY);
} catch {
  accessToken = null;
}

// Set by the auth context so a rejected token can end the session everywhere at
// once, rather than every page discovering it separately.
let onUnauthorized = null;

export function setAccessToken(token) {
  accessToken = token || null;
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // A browser with storage disabled still works; the session just ends when
    // the tab closes.
  }
}

export function getAccessToken() {
  return accessToken;
}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, { method = 'GET', body, headers = {}, signal, auth = true } = {}) {
  const options = { method, headers: { ...headers }, signal };

  if (auth && accessToken) {
    options.headers.Authorization = `Bearer ${accessToken}`;
  }

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
    // An expired or rejected token ends the session once, centrally.
    if (response.status === 401 && auth && onUnauthorized) {
      onUnauthorized();
    }

    const detail =
      (payload && typeof payload === 'object' && payload.detail) ||
      (typeof payload === 'string' && payload) ||
      `Request failed with status ${response.status}`;

    throw new ApiError(flattenDetail(detail, response.status), response.status, payload);
  }

  return payload;
}

/**
 * FastAPI returns validation problems as {field: [message, ...]}, which is what
 * a form needs but not what a banner can show. This keeps the structure on the
 * error object for the form and produces a readable sentence for everything else.
 */
function flattenDetail(detail, status) {
  if (typeof detail === 'string') return detail;

  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const parts = Object.values(detail)
      .flatMap((value) => (Array.isArray(value) ? value : [value]))
      .filter((value) => typeof value === 'string');
    if (parts.length) return parts.join(' ');
  }

  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => (typeof item === 'string' ? item : item?.msg))
      .filter(Boolean);
    if (parts.length) return parts.join(' ');
  }

  return `Request failed with status ${status}`;
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

  // --- auth -----------------------------------------------------------------
  register: (payload) => request('/api/auth/register', { method: 'POST', body: payload, auth: false }),
  login: (payload) => request('/api/auth/login', { method: 'POST', body: payload, auth: false }),
  activate: (token) => request('/api/auth/activate', { method: 'POST', body: { token }, auth: false }),
  resendActivation: (identifier) =>
    request('/api/auth/resend-activation', { method: 'POST', body: { identifier }, auth: false }),
  me: () => request('/api/auth/me'),
  changePassword: (payload) => request('/api/auth/password', { method: 'POST', body: payload }),
  passwordStrength: (password) =>
    request('/api/auth/password/strength', { method: 'POST', body: { password }, auth: false }),
  passwordRules: () => request('/api/auth/rules', { auth: false }),
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
  // toQuery already repeats array values as separate params, which is what
  // FastAPI's `list[str] = Query(...)` expects.

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

  // --- verification ---------------------------------------------------------
  verify: (payload) => request('/api/verify', { method: 'POST', body: payload }),
  listVerifications: (params) => request(`/api/verify${toQuery(params)}`),
  getVerification: (id) => request(`/api/verify/${id}`),
  deleteVerification: (id) => request(`/api/verify/${id}`, { method: 'DELETE' }),

  // --- graph ----------------------------------------------------------------
  getGraph: (params) => request(`/api/graph${toQuery(params)}`),
  graphStats: () => request('/api/graph/stats'),
  clearGraph: () => request('/api/graph', { method: 'DELETE' }),

  // --- literature review ----------------------------------------------------
  generateReview: (payload) => request('/api/review', { method: 'POST', body: payload }),
  listReviews: () => request('/api/review'),
  getReview: (id) => request(`/api/review/${id}`),
  deleteReview: (id) => request(`/api/review/${id}`, { method: 'DELETE' }),
  reviewMarkdownUrl: (id) => `${BASE_URL}/api/review/${id}/markdown`,
  reviewPdfUrl: (id) => `${BASE_URL}/api/review/${id}/pdf`,

  // --- reports --------------------------------------------------------------
  createReport: (payload) => request('/api/reports', { method: 'POST', body: payload }),
  listReports: () => request('/api/reports'),
  getReport: (id) => request(`/api/reports/${id}`),
  deleteReport: (id) => request(`/api/reports/${id}`, { method: 'DELETE' }),
  reportPdfUrl: (id) => `${BASE_URL}/api/reports/${id}/pdf`,
  reportMarkdownUrl: (id) => `${BASE_URL}/api/reports/${id}/markdown`,
};

export default api;
