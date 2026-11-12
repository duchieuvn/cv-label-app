import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { AnnotatePage } from './annotate/AnnotatePage.tsx';
import { ProjectPage } from './pages/ProjectPage.tsx';
import { ProjectsPage } from './pages/ProjectsPage.tsx';
import { SettingsPage } from './pages/SettingsPage.tsx';
import { useRoute } from './router.ts';
import './styles.css';

function App() {
  const route = useRoute();
  switch (route.page) {
    case 'projects':
      return <ProjectsPage />;
    case 'project':
      return <ProjectPage key={route.projectId} projectId={route.projectId} />;
    case 'settings':
      return <SettingsPage key={route.projectId} projectId={route.projectId} />;
    case 'annotate':
      return <AnnotatePage key={route.projectId} projectId={route.projectId} imageId={route.imageId} filter={route.filter} />;
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
