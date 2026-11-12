import { useEffect, useState, type FormEvent } from 'react';
import type { Project, ProjectMode } from '../../../shared/schema.ts';
import { api } from '../api.ts';
import { MODES } from '../modes.ts';
import { navigate } from '../router.ts';

export function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState('');
  const [mode, setMode] = useState<ProjectMode>('bbox');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listProjects().then(setProjects).catch((e) => setError(e.message));
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const project = await api.createProject(name, mode);
      navigate(`/projects/${project.id}/settings`);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <h1>LabelApp</h1>
      </header>
      {error && <p className="error">{error}</p>}
      <form className="inline-form" onSubmit={create}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New project name" aria-label="New project name" />
        <select value={mode} aria-label="Project mode" onChange={(e) => setMode(e.target.value as ProjectMode)}>
          {(Object.keys(MODES) as ProjectMode[]).map((m) => (
            <option key={m} value={m}>
              {MODES[m].name}
            </option>
          ))}
        </select>
        <button className="btn primary" disabled={!name.trim()}>
          Create project
        </button>
      </form>
      <ul className="project-list">
        {projects?.map((p) => (
          <li key={p.id}>
            <a href={`#/projects/${p.id}`}>
              <strong>{p.name}</strong>
              <span className="muted">
                {MODES[p.mode].name} · {p.labels.length} {p.labels.length === 1 ? 'label' : 'labels'} · {p.doneCount} of {p.imageCount} images done
              </span>
              <progress value={p.doneCount} max={Math.max(p.imageCount, 1)} />
            </a>
          </li>
        ))}
        {projects?.length === 0 && <li className="empty">No projects yet. Create one to get started.</li>}
      </ul>
    </div>
  );
}
