import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import { CorpusProvider } from './context/CorpusContext';
import Ask from './pages/Ask';
import Dashboard from './pages/Dashboard';
import Discover from './pages/Discover';
import Extraction from './pages/Extraction';
import Figures from './pages/Figures';
import Gaps from './pages/Gaps';
import Hypotheses from './pages/Hypotheses';
import Matrix from './pages/Matrix';
import Reader from './pages/Reader';
import GraphView from './pages/GraphView';
import Library from './pages/Library';
import Reports from './pages/Reports';
import Summaries from './pages/Summaries';
import Upload from './pages/Upload';

export default function App() {
  return (
    <CorpusProvider>
      <Routes>
        <Route element={<Layout />}>
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
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </CorpusProvider>
  );
}
