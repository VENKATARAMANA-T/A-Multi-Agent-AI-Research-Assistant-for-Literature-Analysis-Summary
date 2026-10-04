import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Layout from './components/Layout';
import { AuthProvider, useAuth } from './context/AuthContext';
import { CorpusProvider } from './context/CorpusContext';
import { FlashProvider } from './context/FlashContext';
import Activate from './pages/Activate';
import Ask from './pages/Ask';
import Dashboard from './pages/Dashboard';
import Discover from './pages/Discover';
import Extraction from './pages/Extraction';
import Figures from './pages/Figures';
import Gaps from './pages/Gaps';
import Home from './pages/Home';
import Hypotheses from './pages/Hypotheses';
import Matrix from './pages/Matrix';
import Profile from './pages/Profile';
import Reader from './pages/Reader';
import GraphView from './pages/GraphView';
import Library from './pages/Library';
import Reports from './pages/Reports';
import SignIn from './pages/SignIn';
import SignUp from './pages/SignUp';
import Summaries from './pages/Summaries';
import Upload from './pages/Upload';

/**
 * Nothing renders until the stored token has been checked.
 *
 * Deciding on the presence of a string in localStorage would flash the whole
 * workspace before an expired token bounced the user to sign-in, and would
 * send every page's opening request with a token already known to be dead.
 */
function RequireAuth({ children }) {
  const { isAuthenticated, ready } = useAuth();
  const location = useLocation();

  if (!ready) {
    return (
      <div className="boot">
        <span className="auth-spinner" aria-hidden="true" />
        <p className="muted">Restoring your session…</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    // Remembered so sign-in returns the user to where they were going.
    return <Navigate to="/signin" replace state={{ from: location.pathname + location.search }} />;
  }

  return children;
}

/** Signing in again when already signed in just means "open the workspace". */
function RedirectIfAuthenticated({ children }) {
  const { isAuthenticated, ready } = useAuth();
  if (ready && isAuthenticated) return <Navigate to="/app" replace />;
  return children;
}

export default function App() {
  return (
    <FlashProvider>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route
            path="/signin"
            element={
              <RedirectIfAuthenticated>
                <SignIn />
              </RedirectIfAuthenticated>
            }
          />
          <Route
            path="/signup"
            element={
              <RedirectIfAuthenticated>
                <SignUp />
              </RedirectIfAuthenticated>
            }
          />
          <Route path="/activate" element={<Activate />} />

          <Route
            path="/app"
            element={
              <RequireAuth>
                {/* Inside the guard: the corpus is per account, so it must not
                    be fetched before there is one. */}
                <CorpusProvider>
                  <Layout />
                </CorpusProvider>
              </RequireAuth>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="upload" element={<Upload />} />
            <Route path="library" element={<Library />} />
            <Route path="ask" element={<Ask />} />
            <Route path="summaries" element={<Summaries />} />
            <Route path="extraction" element={<Extraction />} />
            <Route path="figures" element={<Figures />} />
            <Route path="matrix" element={<Matrix />} />
            <Route path="discover" element={<Discover />} />
            <Route path="hypotheses" element={<Hypotheses />} />
            <Route path="reader/:paperId" element={<Reader />} />
            <Route path="gaps" element={<Gaps />} />
            <Route path="graph" element={<GraphView />} />
            <Route path="reports" element={<Reports />} />
            <Route path="profile" element={<Profile />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </FlashProvider>
  );
}
