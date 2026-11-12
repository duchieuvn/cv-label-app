import { useEffect, useState } from 'react';
import type { Label, Project } from '../../../shared/schema.ts';
import { newId, nextLabelColor } from '../annotate/doc.ts';
import { api } from '../api.ts';
import { MODES } from '../modes.ts';

const COCO_PERSON_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];
const COCO_PERSON_SKELETON = [
  [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2],
  [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
];

function cocoPerson(color: string, takenNames: string[]): Label {
  const keypoints = COCO_PERSON_KEYPOINTS.map((name) => ({ id: newId(), name }));
  return {
    id: newId(),
    name: takenNames.includes('person') ? `person_${takenNames.length + 1}` : 'person',
    color,
    keypoints,
    skeleton: COCO_PERSON_SKELETON.map(([a, b]) => [keypoints[a - 1].id, keypoints[b - 1].id]),
  };
}

export function SettingsPage({ projectId }: { projectId: string }) {
  const [project, setProject] = useState<Project | null>(null);
  const [name, setName] = useState('');
  const [labels, setLabels] = useState<Label[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    api
      .getProject(projectId)
      .then((p) => {
        setProject(p);
        setName(p.name);
        setLabels(p.labels);
      })
      .catch((e) => setError(e.message));
  }, [projectId]);

  const update =(id: string, change: (label: Label) => Label) => setLabels((list) => list.map((l) => (l.id === id ? change(l) : l)));

  const addLabel = () => setLabels((list) => [...list, { id: newId(), name: '', color: nextLabelColor(list), keypoints: [], skeleton: [] }]);

  const mode = project && MODES[project.mode];
  const dirty = project !== null && (name !== project.name || JSON.stringify(labels) !== JSON.stringify(project.labels));

  const save = async () => {
    if (!project) return;
    setError(null);
    setMessage(null);

    const problem = labels.find((l) => !l.name.trim() || l.keypoints.some((k) => !k.name.trim()));
    if (problem) {
      setError('Every label and keypoint needs a name.');
      return;
    }

    const newIds = new Set(labels.flatMap((l) => l.keypoints.map((k) => k.id)));
    const removed = project.labels.flatMap((l) =>
      labels.some((n) => n.id === l.id) ? l.keypoints.filter((k) => !newIds.has(k.id)) : [],
    );
    if (
      removed.length > 0 &&
      !window.confirm(`Removing ${removed.map((k) => k.name).join(', ')} also deletes these points from every annotation in this project. Continue?`)
    ) {
      return;
    }

    try {
      const result = await api.updateProject(projectId, { name, labels });
      setProject(result.project);
      setName(result.project.name);
      setLabels(result.project.labels);
      setMessage(result.prunedKeypoints > 0 ? `Saved. Removed ${result.prunedKeypoints} placed keypoints.` : 'Saved.');
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="page settings">
      <header className="page-header">
        <div>
          <a className="muted" href={`#/projects/${projectId}`}>
            ← Back to project
          </a>
          <h1>Project settings</h1>
        </div>
        <div className="spacer" />
        {message && !dirty && <span className="muted">{message}</span>}
        <button className="btn primary" disabled={!dirty} onClick={save}>
          Save changes
        </button>
      </header>
      {error && <p className="error">{error}</p>}

      <label className="field wide">
        Project name
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>

      {mode && (
        <p className="muted" data-testid="project-mode">
          Mode: {mode.name}. A project's mode is set when it is created and cannot be changed.
        </p>
      )}

      <div className="section-title">
        <h2>Labels</h2>
        <div className="spacer" />
        <button className="btn" onClick={addLabel}>
          Add label
        </button>
        {mode?.keypoints && (
          <button className="btn" onClick={() => setLabels((list) => [...list, cocoPerson(nextLabelColor(list), list.map((l) => l.name))])}>
            Add COCO person (17 keypoints)
          </button>
        )}
      </div>

      {labels.length === 0 && <p className="empty">No labels yet. A label is an object class, such as "car" or "person".</p>}

      {labels.map((label) => (
        <LabelCard
          key={label.id}
          label={label}
          showKeypoints={mode?.keypoints ?? false}
          onChange={(change) => update(label.id, change)}
          onDelete={() => setLabels((list) => list.filter((l) => l.id !== label.id))}
        />
      ))}
    </div>
  );
}

function LabelCard({ label, showKeypoints, onChange, onDelete }: { label: Label; showKeypoints: boolean; onChange: (change: (l: Label) => Label) => void; onDelete: () => void }) {
  const [bulk, setBulk] = useState('');

  const addKeypoints = () => {
    const taken = new Set(label.keypoints.map((k) => k.name));
    const names = [...new Set(bulk.split(',').map((n) => n.trim()).filter(Boolean))].filter((n) => !taken.has(n));
    if (names.length > 0) onChange((l) => ({ ...l, keypoints: [...l.keypoints, ...names.map((n) => ({ id: newId(), name: n }))] }));
    setBulk('');
  };

  const moveUp = (index: number) =>
    onChange((l) => {
      const keypoints = [...l.keypoints];
      [keypoints[index - 1], keypoints[index]] = [keypoints[index], keypoints[index - 1]];
      return { ...l, keypoints };
    });

  const removeKeypoint = (id: string) =>
    onChange((l) => ({
      ...l,
      keypoints: l.keypoints.filter((k) => k.id !== id),
      skeleton: l.skeleton.filter(([a, b]) => a !== id && b !== id),
    }));

  const addEdge = () => {
    const [a, b] = label.keypoints;
    onChange((l) => ({ ...l, skeleton: [...l.skeleton, [a.id, b.id]] }));
  };

  const setEdge = (index: number, position: 0 | 1, id: string) =>
    onChange((l) => ({
      ...l,
      skeleton: l.skeleton.map((edge, i) => (i === index ? (position === 0 ? [id, edge[1]] : [edge[0], id]) : edge)),
    }));

  return (
    <section className="label-card" data-testid="label-card">
      <div className="label-head">
        <input type="color" value={label.color} aria-label="Color" onChange={(e) => onChange((l) => ({ ...l, color: e.target.value }))} />
        <input value={label.name} placeholder="Label name" aria-label="Label name" onChange={(e) => onChange((l) => ({ ...l, name: e.target.value }))} />
        <button className="btn danger" onClick={onDelete}>
          Delete label
        </button>
      </div>

      {showKeypoints && (
      <div className="label-columns">
        <div>
          <h3>Keypoints ({label.keypoints.length})</h3>
          <p className="muted">Order sets the export order and placement order. A label with none can only be an image label.</p>
          <ol className="kp-editor">
            {label.keypoints.map((k, i) => (
              <li key={k.id}>
                <input
                  value={k.name}
                  aria-label="Keypoint name"
                  onChange={(e) =>
                    onChange((l) => ({ ...l, keypoints: l.keypoints.map((x) => (x.id === k.id ? { ...x, name: e.target.value } : x)) }))
                  }
                />
                <button className="mini" disabled={i === 0} aria-label={`Move ${k.name} up`} onClick={() => moveUp(i)}>
                  ↑
                </button>
                <button className="x" aria-label={`Remove ${k.name}`} onClick={() => removeKeypoint(k.id)}>
                  ×
                </button>
              </li>
            ))}
          </ol>
          <form
            className="inline-form"
            onSubmit={(e) => {
              e.preventDefault();
              addKeypoints();
            }}
          >
            <input value={bulk} placeholder="Add keypoints: head, tail, ..." aria-label="Add keypoints" onChange={(e) => setBulk(e.target.value)} />
            <button className="btn" disabled={!bulk.trim()}>
              Add
            </button>
          </form>
        </div>

        <div>
          <h3>Skeleton ({label.skeleton.length})</h3>
          <p className="muted">Optional. Lines drawn between keypoints.</p>
          <ul className="edge-editor">
            {label.skeleton.map(([a, b], i) => (
              <li key={i}>
                {([a, b] as const).map((value, position) => (
                  <select key={position} value={value} aria-label={position === 0 ? 'From' : 'To'} onChange={(e) => setEdge(i, position as 0 | 1, e.target.value)}>
                    {label.keypoints.map((k) => (
                      <option key={k.id} value={k.id}>
                        {k.name}
                      </option>
                    ))}
                  </select>
                ))}
                <button className="x" aria-label="Remove edge" onClick={() => onChange((l) => ({ ...l, skeleton: l.skeleton.filter((_, j) => j !== i) }))}>
                  ×
                </button>
              </li>
            ))}
          </ul>
          <button className="btn" disabled={label.keypoints.length < 2} onClick={addEdge}>
            Add edge
          </button>
        </div>
      </div>
      )}
    </section>
  );
}
