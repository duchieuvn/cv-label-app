import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { BundlePreview, Project, ProjectMode } from '../../../shared/schema.ts';
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
      <OpenSharedProject />
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

/**
 * Opens a project file a teammate exported. Each of its folders is placed by typing where it is on this computer,
 * and a preview checks a few of its files there before anything is created. See specs.md section 13.
 */
function OpenSharedProject() {
  const [file, setFile] = useState<File | null>(null);
  const [paths, setPaths] = useState<Record<string, string>>({});
  const [name, setName] = useState('');
  const [preview, setPreview] = useState<BundlePreview | null>(null);
  /** The paths the shown preview checked, so Create waits for a preview of what is typed now. */
  const [previewed, setPreviewed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<{ projectId: string; warnings: string[] } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  /** Only the newest preview may apply, since paths change while one is in flight. */
  const requestId = useRef(0);
  const key = JSON.stringify(paths);

  useEffect(() => {
    if (!file) return;
    const request = ++requestId.current;
    // Typing a path previews after a pause rather than on every key.
    const timer = window.setTimeout(() => {
      api
        .openBundle(file, { paths }, true)
        .then((result) => {
          if (request !== requestId.current) return;
          setPreview(result);
          setPreviewed(key);
          setName((current) => current || result.name);
          setError(null);
        })
        .catch((e) => {
          if (request !== requestId.current) return;
          setPreview(null);
          setError((e as Error).message);
        });
    }, preview ? 400 : 0);
    return () => window.clearTimeout(timer);
  }, [file, key]);

  const pick = (next: File | null) => {
    requestId.current++;
    setFile(next);
    setPaths({});
    setName('');
    setPreview(null);
    setPreviewed(null);
    setError(null);
    setCreated(null);
  };

  const create = async () => {
    if (!file) return;
    requestId.current++;
    setBusy(true);
    setError(null);
    try {
      const result = await api.openBundle(file, { name: name.trim() || undefined, paths }, false);
      if (result.warnings.length === 0) {
        navigate(`/projects/${result.project.id}`);
        return;
      }
      setCreated({ projectId: result.project.id, warnings: result.warnings });
      pick(null);
      if (input.current) input.current.value = '';
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const folderErrors = preview?.folders.some((f) => f.error) ?? true;

  return (
    <details className="callout open-shared" data-testid="open-shared" open={file !== null || created !== null}>
      <summary>Open a shared project</summary>
      <p className="muted">
        Opens a project file exported by a teammate (Export, then Project file) as a new project here. The file holds labels and annotations, not images, so
        say where each of its folders is on this computer. Images are matched by their path inside the folder.
      </p>
      <input ref={input} type="file" accept=".labelapp" aria-label="Project file" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
      {error && <p className="error">{error}</p>}
      {created && (
        <div role="status">
          <p>The project was created, with some images left out:</p>
          <ul>
            {created.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
          <a className="btn primary" href={`#/projects/${created.projectId}`}>
            Open project
          </a>
        </div>
      )}
      {preview && (
        <div className="open-shared-form">
          <label>
            Name <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Project name" />
          </label>
          <p className="muted">
            {MODES[preview.mode].name} · {preview.labelCount} {preview.labelCount === 1 ? 'label' : 'labels'} · {preview.imageCount.toLocaleString()} images
          </p>
          {preview.folders.map((folder) => (
            <div className="shared-folder" key={folder.key} data-testid="shared-folder">
              <strong>{folder.name}</strong>
              <span className="muted">
                {' '}
                · {folder.imageCount.toLocaleString()} images{folder.originalPath && <> · was <span className="mono">{folder.originalPath}</span></>}
              </span>
              <input
                className="mono"
                value={paths[folder.key] ?? ''}
                placeholder="/absolute/path/to/this/folder"
                aria-label={`Path of ${folder.name}`}
                onChange={(e) => setPaths({ ...paths, [folder.key]: e.target.value })}
              />
              <span className="scan-line" data-testid="shared-folder-status">
                {!paths[folder.key]?.trim() ? (
                  'Left out unless a path is given.'
                ) : previewed !== key ? (
                  'Checking…'
                ) : folder.error ? (
                  <span className="error">{folder.error}</span>
                ) : folder.found === 0 ? (
                  <span className="error">None of {folder.checked} sample files are here. Is this the right folder?</span>
                ) : (
                  `${folder.found} of ${folder.checked} sample files found.`
                )}
                {folder.samples.length > 0 && <span className="mono"> e.g. {folder.samples.join(', ')}</span>}
              </span>
            </div>
          ))}
          <button className="btn primary" disabled={busy || previewed !== key || folderErrors || !name.trim()} onClick={create}>
            Create project
          </button>
        </div>
      )}
    </details>
  );
}
