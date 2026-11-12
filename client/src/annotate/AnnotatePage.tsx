import { useEffect, useRef, useState } from 'react';
import type { Annotation, BBox, ImageDetail, ImageMeta, Project } from '../../../shared/schema.ts';
import { api, imageUrl, saveQueue } from '../api.ts';
import { MODES } from '../modes.ts';
import { filterQuery, navigate, type FolderFilter } from '../router.ts';
import { Canvas, type CanvasHandle, type Tool } from './Canvas.tsx';
import {
  boxAnnotation,
  changeLabel,
  emptyHistory,
  moveAnnotation,
  newId,
  nextUnplaced,
  placeKeypoint,
  pushHistory,
  redo,
  removeAnnotation,
  removeKeypoint,
  resetHistory,
  setAnnotation,
  toggleVisibility,
  undo,
  type Doc,
  type History,
  type Selection,
} from './doc.ts';
import { clampPoint, type Point } from './geometry.ts';
import { ImageList } from './ImageList.tsx';
import type { SaveState } from './saveQueue.ts';

const COMMON_SHORTCUTS: [string, string][] = [
  ['1–9', 'Choose active label'],
  ['A / D', 'Previous / next image'],
  ['Ctrl+Enter', 'Mark done, go to next todo'],
  ['Ctrl+Z', 'Undo'],
  ['Ctrl+Shift+Z, Ctrl+Y', 'Redo'],
  ['Esc', 'Cancel drag, or deselect'],
  ['H', 'Hide or show annotations'],
  ['F, +, -', 'Fit, zoom in, zoom out'],
  ['Space+drag', 'Pan (also middle mouse)'],
  ['Wheel', 'Zoom at cursor'],
];

const TOOLS: { id: Tool; name: string; key: string }[] = [
  { id: 'select', name: 'Select', key: 'V' },
  { id: 'bbox', name: 'Box', key: 'B' },
  { id: 'keypoint', name: 'Keypoint', key: 'K' },
];

export function AnnotatePage({ projectId, imageId, filter }: { projectId: string; imageId: string; filter: FolderFilter | null }) {
  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [imageError, setImageError] = useState(false);
  const [history, setHistory] = useState<History>(emptyHistory);
  /** Null until a tool is chosen, so the annotator starts on the mode's own tool. */
  const [pickedTool, setPickedTool] = useState<Tool | null>(null);
  const [activeLabelId, setActiveLabelId] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [hidden, setHidden] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>(saveQueue.state);

  const canvasRef = useRef<CanvasHandle>(null);
  /** The document last known to match the server, so loading an image doesn't trigger a save. */
  const syncedDoc = useRef<Doc | null>(null);

  const doc = history.present;
  const mode = project && MODES[project.mode];
  // Select does nothing destructive, so it is safe for the moment before the project has loaded.
  const tool: Tool = pickedTool && mode?.tools.includes(pickedTool) ? pickedTool : (mode?.defaultTool ?? 'select');
  const labels = project?.labels ?? [];
  const activeLabel = labels.find((l) => l.id === activeLabelId) ?? labels[0] ?? null;
  const selectedAnn = selection ? doc.find((a) => a.id === selection.annotationId) : undefined;
  const selectedLabel = selectedAnn && labels.find((l) => l.id === selectedAnn.labelId);
  const index = images.findIndex((i) => i.id === imageId);
  const readOnly = detail !== null && detail.fileState !== 'ok';
  const query = filterQuery(filter);

  useEffect(() => saveQueue.subscribe(setSaveState), []);

  useEffect(() => {
    Promise.all([api.getProject(projectId), saveQueue.flush().then(() => api.listImages(projectId, filter ?? {}))])
      .then(([p, list]) => {
        setProject(p);
        setImages(list);
      })
      .catch((e) => setLoadError(e.message));
  }, [projectId, filter?.folder, filter?.recursive]);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setImageError(false);
    setLoadError(null);
    setSelection(null);
    saveQueue
      .flush()
      .then(() => api.getImage(imageId))
      .then((image) => {
        if (cancelled) return;
        // A failed save may still be queued, and it is newer than what the server returned.
        const merged = { ...image, ...saveQueue.pendingFor(imageId) };
        setDetail(merged);
        syncedDoc.current = merged.annotations;
        setHistory(resetHistory(imageId, merged.annotations));
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e.message);
      });
    return () => {
      cancelled = true;
      void saveQueue.flush();
    };
  }, [imageId]);

  useEffect(() => {
    if (history.imageId && history.present !== syncedDoc.current) {
      syncedDoc.current = history.present;
      saveQueue.schedule(history.imageId, { annotations: history.present });
    }
  }, [history]);

  useEffect(() => {
    const onPageHide = () => saveQueue.flushOnUnload();
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (saveQueue.state.kind === 'error') e.preventDefault();
    };
    window.addEventListener('pagehide', onPageHide);
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => {
      window.removeEventListener('pagehide', onPageHide);
      window.removeEventListener('beforeunload', onBeforeUnload);
    };
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 4000);
    return () => clearTimeout(timer);
  }, [notice]);

  // Preload the next image so moving through a dataset feels instant.
  useEffect(() => {
    const next = images[index + 1];
    if (next) new Image().src = imageUrl(next.id);
  }, [index, images]);

  const commit = (next: Doc) => {
    if (!readOnly) setHistory((h) => pushHistory(h, next));
  };

  const toggleImageLabel = (labelId: string) => {
    if (!detail) return;
    const imageLabels = detail.imageLabels.includes(labelId) ? detail.imageLabels.filter((l) => l !== labelId) : [...detail.imageLabels, labelId];
    setDetail({ ...detail, imageLabels });
    saveQueue.schedule(detail.id, { imageLabels });
  };

  const chooseTool = (next: Tool) => {
    if (mode?.tools.includes(next)) setPickedTool(next);
  };

  const select = (next: Selection) => {
    setSelection(next);
    const ann = next && doc.find((a) => a.id === next.annotationId);
    if (ann) setActiveLabelId(ann.labelId);
  };

  const chooseLabel = (labelId: string) => {
    setActiveLabelId(labelId);
    // An unfinished instance of another label would otherwise capture the next keypoint click.
    if (selectedAnn && selectedAnn.labelId !== labelId) setSelection(null);
  };

  const goTo = (id: string) => navigate(`/projects/${projectId}/annotate/${id}${query}`);

  const step = (delta: number) => {
    const next = images[index + delta];
    if (next) goTo(next.id);
  };

  const setStatus = (status: ImageMeta['status']) => {
    saveQueue.schedule(imageId, { status });
    setImages((list) => list.map((i) => (i.id === imageId ? { ...i, status } : i)));
  };

  const markDoneAndNext = () => {
    setStatus('done');
    const after = [...images.slice(index + 1), ...images.slice(0, Math.max(index, 0))];
    const next = after.find((i) => i.status === 'todo');
    if (next) goTo(next.id);
    else setNotice(filter ? 'Every image in this folder is done.' : 'Every image in this project is done.');
  };

  const onDrawBox = (box: BBox) => {
    if (readOnly) return;
    if (!activeLabel) {
      setNotice('Add a label in project settings first.');
      return;
    }
    const ann = boxAnnotation(activeLabel.id, box);
    commit([...doc, ann]);
    setSelection({ annotationId: ann.id, keypointId: null });
  };

  const onPlace = (point: Point, occluded: boolean) => {
    if (readOnly) return;
    const v = occluded ? 1 : 2;
    let target = selectedAnn;
    let targetLabel = selectedLabel;
    let keypointId: string | null = null;

    if (target && targetLabel && targetLabel.keypoints.length > 0) {
      const current = selection?.keypointId;
      keypointId = current && !target.keypoints[current] ? current : nextUnplaced(targetLabel, target, current ?? null);
    }

    let next: Doc;
    let placed: Annotation;
    if (target && keypointId) {
      placed = placeKeypoint(target, keypointId, point, v);
      next = setAnnotation(doc, placed);
    } else {
      if (!activeLabel) {
        setNotice('Add a label in project settings first.');
        return;
      }
      if (activeLabel.keypoints.length === 0) {
        setNotice(`Label "${activeLabel.name}" has no keypoints. Define them in project settings.`);
        return;
      }
      targetLabel = activeLabel;
      keypointId = activeLabel.keypoints[0].id;
      placed = placeKeypoint({ id: newId(), labelId: activeLabel.id, bbox: null, keypoints: {} }, keypointId, point, v);
      next = [...doc, placed];
    }
    commit(next);
    const following = nextUnplaced(targetLabel!, placed, keypointId);
    setSelection(following ? { annotationId: placed.id, keypointId: following } : null);
  };

  const deleteSelection = () => {
    if (!selection) return;
    if (selection.keypointId && selectedAnn?.keypoints[selection.keypointId]) {
      commit(removeKeypoint(doc, selection.annotationId, selection.keypointId));
    } else {
      commit(removeAnnotation(doc, selection.annotationId));
      setSelection(null);
    }
  };

  const nudge = (dx: number, dy: number) => {
    if (!selectedAnn || !detail) return;
    const keypointId = selection?.keypointId;
    const point = keypointId ? selectedAnn.keypoints[keypointId] : undefined;
    if (keypointId && point) {
      const moved = clampPoint({ x: point.x + dx, y: point.y + dy }, detail);
      commit(setAnnotation(doc, placeKeypoint(selectedAnn, keypointId, moved, point.v)));
    } else {
      commit(setAnnotation(doc, moveAnnotation(selectedAnn, dx, dy, detail)));
    }
  };

  const toggleOcclusion = (annotation: Annotation, keypointId: string) =>
    commit(setAnnotation(doc, toggleVisibility(annotation, keypointId)));

  const onChangeLabel = (annotation: Annotation, labelId: string) => {
    const hasKeypoints = Object.keys(annotation.keypoints).length > 0;
    if (hasKeypoints && !window.confirm('Changing the label removes this instance’s keypoints. Continue?')) return;
    const changed = changeLabel(annotation, labelId);
    if (!changed) return;
    commit(setAnnotation(doc, changed));
    setActiveLabelId(labelId);
    setSelection({ annotationId: annotation.id, keypointId: null });
  };

  // Rebound every render through a ref so the handler always sees current state.
  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandler.current = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;

    if (ctrl) {
      if (key === 'z') setHistory((h) => (e.shiftKey ? redo(h) : undo(h)));
      else if (key === 'y') setHistory(redo);
      else if (key === 'enter') {
        // Holding the keys must not mark a run of unseen images as done.
        if (!e.repeat) markDoneAndNext();
      }
      else return;
      e.preventDefault();
      return;
    }

    const arrow = { arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1] }[key];
    if (arrow) {
      if (selectedAnn) {
        const amount = e.shiftKey ? 10 : 1;
        nudge(arrow[0] * amount, arrow[1] * amount);
        e.preventDefault();
      }
      return;
    }
    if (/^[1-9]$/.test(key)) {
      const label = labels[Number(key) - 1];
      if (label) chooseLabel(label.id);
      return;
    }

    switch (key) {
      case 'v':
        chooseTool('select');
        break;
      case 'b':
        chooseTool('bbox');
        break;
      case 'k':
        chooseTool('keypoint');
        break;
      case 'a':
        step(-1);
        break;
      case 'd':
        step(1);
        break;
      case 'h':
        setHidden((h) => !h);
        break;
      case 'f':
        canvasRef.current?.fit();
        break;
      case '+':
      case '=':
        canvasRef.current?.zoom(1.25);
        break;
      case '-':
        canvasRef.current?.zoom(0.8);
        break;
      case '?':
        setShowHelp((s) => !s);
        break;
      case 'delete':
      case 'backspace':
        deleteSelection();
        break;
      case 'escape':
        if (showHelp) setShowHelp(false);
        else setSelection(null);
        break;
      case 'n':
        if (selectedAnn && selectedLabel) {
          const next = nextUnplaced(selectedLabel, selectedAnn, selection?.keypointId ?? null);
          if (next) setSelection({ annotationId: selectedAnn.id, keypointId: next });
        }
        break;
      case 'o':
        if (selectedAnn && selection?.keypointId && selectedAnn.keypoints[selection.keypointId]) {
          toggleOcclusion(selectedAnn, selection.keypointId);
        }
        break;
    }
  };
  useEffect(() => {
    const listener = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, []);

  const currentStatus = images[index]?.status ?? detail?.status;
  const placedCount = (a: Annotation) => Object.keys(a.keypoints).length;
  const hint = (() => {
    if (labels.length === 0) return null;
    if (tool === 'keypoint' && selectedAnn && selectedLabel) {
      const current = selection?.keypointId;
      const target = current && !selectedAnn.keypoints[current] ? current : nextUnplaced(selectedLabel, selectedAnn, current ?? null);
      const name = selectedLabel.keypoints.find((k) => k.id === target)?.name;
      if (name) return `Click to place "${name}". N skips it, Shift+click marks it occluded, Esc finishes.`;
    }
    if (tool === 'keypoint' && activeLabel && activeLabel.keypoints.length === 0) {
      return `Label "${activeLabel.name}" has no keypoints. Choose another label or define them in settings.`;
    }
    return null;
  })();

  if (loadError && !detail) {
    return (
      <div className="page">
        <p className="error">Could not load this image: {loadError}</p>
        <a href={`#/projects/${projectId}`}>Back to project</a>
      </div>
    );
  }

  return (
    <div className="annotator">
      <header className="topbar">
        <a className="btn ghost" href={`#/projects/${projectId}`}>
          ← {project?.name ?? 'Project'}
        </a>
        <div className="group" role="group" aria-label="Tools">
          {TOOLS.filter((t) => mode?.tools.includes(t.id)).map((t) => (
            <button key={t.id} className={`btn ${tool === t.id ? 'active' : ''}`} aria-pressed={tool === t.id} onClick={() => chooseTool(t.id)}>
              {t.name} <kbd>{t.key}</kbd>
            </button>
          ))}
        </div>
        <div className="group">
          <button className="btn" disabled={history.past.length === 0} onClick={() => setHistory(undo)}>
            Undo
          </button>
          <button className="btn" disabled={history.future.length === 0} onClick={() => setHistory(redo)}>
            Redo
          </button>
        </div>
        <div className="group">
          <button className="btn" disabled={index <= 0} onClick={() => step(-1)} aria-label="Previous image">
            ‹ <kbd>A</kbd>
          </button>
          <span className="counter">{index >= 0 ? `${index + 1} / ${images.length}` : ''}</span>
          <button className="btn" disabled={index < 0 || index >= images.length - 1} onClick={() => step(1)} aria-label="Next image">
            <kbd>D</kbd> ›
          </button>
        </div>
        <span className="filename" title={detail?.filename}>
          {detail?.filename}
        </span>
        <div className="spacer" />
        <span className={`save-state ${saveState.kind}`} data-testid="save-state" title={saveState.message}>
          {saveState.kind === 'saved' && 'Saved'}
          {saveState.kind === 'pending' && 'Saving…'}
          {saveState.kind === 'error' && `Not saved, retrying: ${saveState.message}`}
        </span>
        <button className="btn ghost" onClick={() => setShowHelp(true)} aria-label="Keyboard shortcuts">
          ?
        </button>
        {currentStatus === 'done' ? (
          <button className="btn" onClick={() => setStatus('todo')}>
            Reopen
          </button>
        ) : null}
        <button className="btn primary" onClick={markDoneAndNext}>
          {currentStatus === 'done' ? 'Next todo' : 'Done'} <kbd>Ctrl+↵</kbd>
        </button>
      </header>

      <div className="annotator-body">
        <ImageList images={images} currentId={imageId} currentCount={doc.length} hrefFor={(id) => `#/projects/${projectId}/annotate/${id}${query}`} />

        <main className="stage">
          {detail ? (
            <Canvas
              ref={canvasRef}
              url={imageUrl(detail.id)}
              size={detail}
              doc={doc}
              labels={labels}
              tool={tool}
              selection={selection}
              hidden={hidden}
              readOnly={readOnly}
              showBoxes={mode?.showBoxes ?? false}
              onSelect={select}
              onCommit={commit}
              onDrawBox={onDrawBox}
              onPlace={onPlace}
              onImageError={() => setImageError(true)}
            />
          ) : (
            <div className="stage-message">Loading…</div>
          )}
          {readOnly && (
            <div className="banner error" data-testid="unavailable-banner">
              {detail?.fileState === 'missing' ? 'This file is missing from its folder.' : 'This file has new dimensions.'} Annotations are read-only.
            </div>
          )}
          {imageError && !readOnly && <div className="banner error">The image file could not be displayed.</div>}
          {project && labels.length === 0 && (
            <div className="banner">
              This project has no labels yet. <a href={`#/projects/${projectId}/settings`}>Add labels in settings</a>
            </div>
          )}
          {(notice || hint) && <div className="banner">{notice ?? hint}</div>}
        </main>

        <aside className="panel">
          <section>
            <h2>Labels</h2>
            <ul className="label-list">
              {labels.map((label, i) => (
                <li key={label.id}>
                  <button className={label.id === activeLabel?.id ? 'current' : ''} onClick={() => chooseLabel(label.id)}>
                    <span className="swatch" style={{ background: label.color }} />
                    <span className="name">{label.name}</span>
                    {label.keypoints.length > 0 && <span className="muted">{label.keypoints.length} pts</span>}
                    {i < 9 && <kbd>{i + 1}</kbd>}
                  </button>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h2>Image labels</h2>
            <div className="chips">
              {labels.map((label) => {
                const on = detail?.imageLabels.includes(label.id) ?? false;
                return (
                  <button key={label.id} className={`chip ${on ? 'on' : ''}`} aria-pressed={on} onClick={() => toggleImageLabel(label.id)} style={{ '--c': label.color } as React.CSSProperties}>
                    {label.name}
                  </button>
                );
              })}
            </div>
          </section>

          <section className="grow">
            <h2>Instances ({doc.length})</h2>
            <ul className="instance-list">
              {doc.map((a) => {
                const label = labels.find((l) => l.id === a.labelId);
                return (
                  <li key={a.id} className={a.id === selection?.annotationId ? 'current' : ''}>
                    <button className="row" onClick={() => select({ annotationId: a.id, keypointId: null })}>
                      <span className="swatch" style={{ background: label?.color }} />
                      <span className="name">{label?.name}</span>
                      {label && label.keypoints.length > 0 && (
                        <span className="muted">
                          {placedCount(a)}/{label.keypoints.length}
                        </span>
                      )}
                    </button>
                    <button
                      className="x"
                      aria-label={`Delete ${label?.name}`}
                      onClick={() => {
                        commit(removeAnnotation(doc, a.id));
                        if (a.id === selection?.annotationId) setSelection(null);
                      }}
                    >
                      ×
                    </button>
                  </li>
                );
              })}
              {doc.length === 0 && <li className="empty">{mode?.emptyHint}</li>}
            </ul>
          </section>

          {selectedAnn && selectedLabel && (
            <section className="details">
              <h2>Selected</h2>
              {mode?.showBoxes && (
                <label className="field">
                  Label
                  <select value={selectedAnn.labelId} disabled={!selectedAnn.bbox} onChange={(e) => {
                      onChangeLabel(selectedAnn, e.target.value);
                      // Keyboard shortcuts are ignored while a form control has focus.
                      e.target.blur();
                    }}>
                    {labels.map((l) => (
                      <option key={l.id} value={l.id}>
                        {l.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {mode?.showBoxes && selectedAnn.bbox && (
                <p className="muted mono">
                  x {selectedAnn.bbox.x.toFixed(1)} y {selectedAnn.bbox.y.toFixed(1)} w {selectedAnn.bbox.w.toFixed(1)} h {selectedAnn.bbox.h.toFixed(1)}
                </p>
              )}
              {selectedLabel.keypoints.length > 0 && (
                <ul className="keypoint-list">
                  {selectedLabel.keypoints.map((k) => {
                    const p = selectedAnn.keypoints[k.id];
                    const current = selection?.keypointId === k.id;
                    return (
                      <li key={k.id} className={current ? 'current' : ''}>
                        <button className="row" onClick={() => setSelection({ annotationId: selectedAnn.id, keypointId: k.id })}>
                          <span className={`kp-dot ${p ? (p.v === 2 ? 'visible' : 'occluded') : ''}`} style={{ '--c': selectedLabel.color } as React.CSSProperties} />
                          <span className="name">{k.name}</span>
                        </button>
                        {p && (
                          <>
                            <button className="mini" title="Toggle occluded (O)" onClick={() => toggleOcclusion(selectedAnn, k.id)}>
                              {p.v === 2 ? 'visible' : 'occluded'}
                            </button>
                            <button
                              className="x"
                              aria-label={`Remove ${k.name}`}
                              onClick={() => {
                                commit(removeKeypoint(doc, selectedAnn.id, k.id));
                                setSelection({ annotationId: selectedAnn.id, keypointId: k.id });
                              }}
                            >
                              ×
                            </button>
                          </>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          )}
        </aside>
      </div>

      {showHelp && (
        <div className="modal-backdrop" onClick={() => setShowHelp(false)}>
          <div className="modal" role="dialog" aria-label="Keyboard shortcuts" onClick={(e) => e.stopPropagation()}>
            <h2>Keyboard shortcuts</h2>
            <table>
              <tbody>
                {[...(mode?.shortcuts ?? []), ...COMMON_SHORTCUTS].map(([keys, action]) => (
                  <tr key={keys}>
                    <td>
                      <kbd>{keys}</kbd>
                    </td>
                    <td>{action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="btn" onClick={() => setShowHelp(false)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
