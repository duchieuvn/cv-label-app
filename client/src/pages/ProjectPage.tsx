import { useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from 'react';
import type { AutoBoxCheckpoint, AutoBoxInfo, AutoBoxStatus, FolderNode, ImageMeta, ImageStatus, ImportResult, ImportSummary, Label, Project, RemoteJobStatus, RemoteResultsSummary, Source, UploadResult } from '../../../shared/schema.ts';
import { newId, nextLabelColor } from '../annotate/doc.ts';
import { MODES } from '../modes.ts';
import { api, imageUrl, saveQueue, type BulkRequest } from '../api.ts';
import { filterQuery, navigate } from '../router.ts';

const PAGE_SIZE = 200;
const MAX_FILE_BYTES = 100 * 1024 * 1024;
const BATCH_FILES = 10;
const BATCH_BYTES = 50 * 1024 * 1024;
const NEW_LABEL = '__new__';

/** Splits uploads so a large drop reports progress and one failed request can't lose everything. */
function batches(files: File[]): File[][] {
  const result: File[][] = [];
  let current: File[] = [];
  let bytes = 0;
  for (const file of files) {
    if (current.length >= BATCH_FILES || (current.length > 0 && bytes + file.size > BATCH_BYTES)) {
      result.push(current);
      current = [];
      bytes = 0;
    }
    current.push(file);
    bytes += file.size;
  }
  if (current.length > 0) result.push(current);
  return result;
}

/** A folder selection is resolved by the server, so it stays small however many images it covers. */
type GridSelection =
  | { kind: 'folder'; folder: string; recursive: boolean; except: Set<string> }
  | { kind: 'ids'; ids: Set<string> }
  | null;

type Undoable = { message: string; request: BulkRequest };

const count = (n: number) => `${n.toLocaleString()} ${n === 1 ? 'image' : 'images'}`;

type ImportOptions = { createLabels: boolean; overwrite: boolean; markDone: boolean };

/** One line per thing an import did, or would do, in the order a person wants to check them. */
function describeImport(s: ImportSummary, dryRun: boolean): string[] {
  const will = (text: string) => (dryRun ? `Will ${text}` : text.charAt(0).toUpperCase() + text.slice(1));
  const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;
  const lines = [`${s.format === 'coco' ? 'COCO' : 'LabelApp'} file with ${count(s.imagesInFile)}, ${count(s.matched)} found in this project.`];
  lines.push(`${will(`write ${plural(s.annotations, 'annotation')} to ${count(s.imagesWritten)}`)}.`);
  if (s.newLabels.length > 0) lines.push(`${will(`create ${plural(s.newLabels.length, 'label')}`)}: ${s.newLabels.join(', ')}.`);
  if (s.extendedLabels.length > 0) lines.push(`${will('add keypoints to')} ${s.extendedLabels.join(', ')}.`);
  if (s.alreadyAnnotated > 0) lines.push(`${count(s.alreadyAnnotated)} skipped because they already have annotations. Turn on "Replace existing annotations" to overwrite them.`);
  if (s.unmatched > 0) lines.push(`${count(s.unmatched)} in the file are not in this project.`);
  if (s.ambiguous > 0) lines.push(`${count(s.ambiguous)} in the file could be several images here, so they were skipped.`);
  if (s.sizeMismatch > 0) lines.push(`${count(s.sizeMismatch)} skipped because their size in the file differs from the project.`);
  if (s.unavailable > 0) lines.push(`${count(s.unavailable)} skipped because their files are unavailable.`);
  if (s.droppedAnnotations > 0) lines.push(`${plural(s.droppedAnnotations, 'annotation')} dropped for an unknown label or for lying outside the image.`);
  if (s.keypointsIgnored > 0) lines.push(`Keypoints on ${plural(s.keypointsIgnored, 'annotation')} ignored, because this is a bounding box project.`);
  return lines;
}

function describeCheckpoint(c: AutoBoxCheckpoint): string {
  const accuracy = c.bestMap50 === null ? '' : `, mAP50 ${c.bestMap50.toFixed(2)}`;
  const model = `yolov8${c.modelSize ?? 'n'}${c.task === 'pose' ? '-pose' : ''}${c.source === 'remote' ? ', remote' : ''}`;
  return `${new Date(c.trainedAt).toLocaleString()}: ${count(c.trainImages)}, ${c.epochs} epochs${accuracy} (${model})`;
}

const PHASES: Record<RemoteJobStatus['phase'], string> = {
  bundling: 'Making the bundle',
  connecting: 'Connecting to the server',
  preparing: 'Checking the server\'s conda environment (the first run installs it, which takes a while)',
  installing: 'Installing PyTorch and ultralytics on the server, once',
  uploading: 'Uploading the bundle',
  training: 'Training on the server',
  downloading: 'Downloading the trained model',
  importing: 'Adding the checkpoint',
  drawing: 'Drawing here with the new checkpoint',
};

/** The remote run's status line, phrased like the local job's. */
function describeRemote(job: RemoteJobStatus, unit: [string, string]): string {
  const where = ` on ${job.target}`;
  if (job.state === 'running') {
    const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;
    const detail =
      job.phase === 'uploading' && job.uploadTotal > 0
        ? `: ${mb(job.uploaded)} of ${mb(job.uploadTotal)}`
        : job.phase === 'training' && job.epoch > 0
          ? `: epoch ${job.epoch}${job.bestMap50 === null ? '' : `, best mAP50 ${job.bestMap50.toFixed(2)}`}`
          : '';
    const gpu = job.gpu ? ` GPU: ${job.gpu}.` : '';
    return `${PHASES[job.phase]}${['bundling', 'importing', 'drawing'].includes(job.phase) ? '' : where}${detail}.${gpu}`;
  }
  if (job.state === 'failed') return `Remote run failed${where}: ${job.error}`;
  if (job.state === 'cancelled') {
    const hint = job.bundleId && job.phase === 'training' ? ` The server may still be training; collect it later with: python scripts/remote_train.py fetch ${job.bundleId} ${job.target}` : '';
    return `Remote run cancelled.${hint}`;
  }
  const trained = job.summary ? ` Trained on ${count(job.summary.trainImages)}, ${job.summary.epochs} epochs${job.summary.bestMap50 === null ? '' : `, best mAP50 ${job.summary.bestMap50.toFixed(2)}`}.` : '';
  void unit;
  return `Finished training${where}.${trained} ${job.drawing ?? ''}`.trim();
}

function describeRemoteResults(r: RemoteResultsSummary): string {
  const accuracy = r.bestMap50 === null ? '' : `, best mAP50 ${r.bestMap50.toFixed(2)}`;
  const skipped = r.skipped.length > 0 ? ` The server could not read ${r.skipped.length} files.` : '';
  const checkpoint = r.checkpointAdded ? ' The trained model was added as a checkpoint: use Draw with trained model.' : ' No trained model came back.';
  return `Trained on ${count(r.trainImages)}, ${r.epochs} epochs${accuracy}.${skipped}${checkpoint}`;
}

function describeAutoBox(job: AutoBoxStatus): string {
  const accuracy = job.bestMap50 === null ? '' : `, best mAP50 ${job.bestMap50.toFixed(2)}`;
  const checkpoint = job.checkpointTrainedAt && `the checkpoint from ${new Date(job.checkpointTrainedAt).toLocaleString()}`;
  const trained = `${job.continued ? `Continued ${checkpoint} on` : 'Trained on'} ${count(job.trainImages)}, ${job.epoch} epochs${accuracy}`;
  const drawn = `${job.boxes.toLocaleString()} ${job.boxes === 1 ? 'box' : 'boxes'} on ${count(job.boxedImages)}`;
  const skipped = job.skipped > 0 ? `, ${job.skipped.toLocaleString()} skipped` : '';
  const result = `Drew ${drawn}, out of ${count(job.total)}, at confidence ${job.confidence.toFixed(2)}${skipped}.`;
  switch (job.state) {
    case 'failed':
      return `Auto boxing failed: ${job.error}`;
    case 'cancelled':
      return `Auto boxing cancelled. ${job.phase === 'predicting' ? `Kept ${drawn}.` : 'No boxes were drawn.'}`;
    case 'done':
      if (job.mode === 'predict') return `Finished with ${checkpoint}. ${result}`;
      return `Finished. ${trained}${job.stoppedEarly ? ' (stopped early)' : ''}. ${result}`;
    case 'running':
      if (job.phase === 'preparing') return 'Preparing training images…';
      if (job.phase === 'training') return `${job.continued ? `Continuing ${checkpoint} on` : 'Training on'} ${count(job.trainImages)}: epoch ${job.epoch}${accuracy}. Stops after ${job.patience} epochs without improvement.`;
      return `Drawing boxes: ${job.predicted.toLocaleString()} of ${count(job.total)}, ${drawn}${skipped}.`;
  }
}

export function ProjectPage({ projectId }: { projectId: string }) {
  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [folders, setFolders] = useState<FolderNode[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [folder, setFolder] = useState<string | null>(null);
  const [recursive, setRecursive] = useState(true);
  const [labelFilter, setLabelFilter] = useState('');
  const [filter, setFilter] = useState<ImageStatus | 'all' | 'auto'>('all');
  const [autoBox, setAutoBox] = useState<AutoBoxInfo | null>(null);
  /** Null until the slider is moved, so it starts from the last job's value or the server's default. */
  const [pickedConfidence, setPickedConfidence] = useState<number | null>(null);
  /** Null until chosen, so each selector follows the newest checkpoint. Kept as text so the field can be cleared while typing. */
  const [pickedTrainFrom, setPickedTrainFrom] = useState<string | null>(null);
  const [pickedDrawWith, setPickedDrawWith] = useState<string | null>(null);
  const [pickedPatience, setPickedPatience] = useState<string | null>(null);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [selection, setSelection] = useState<GridSelection>(null);
  const [bulkLabel, setBulkLabel] = useState('');
  const [newLabelName, setNewLabelName] = useState('');
  /** Null until the user picks a color, so each new label starts from one that is still free. */
  const [pickedColor, setPickedColor] = useState<string | null>(null);
  const [undoable, setUndoable] = useState<Undoable | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [sourcePath, setSourcePath] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [rejected, setRejected] = useState<UploadResult['rejected']>([]);
  const [dragOver, setDragOver] = useState(false);
  const [format, setFormat] = useState<'coco' | 'yolo' | 'json' | 'image_labels' | 'project'>('coco');
  const [scope, setScope] = useState<'done' | 'annotated'>('done');
  const [labeledScope, setLabeledScope] = useState<'labeled' | 'done'>('labeled');
  const [includeImages, setIncludeImages] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importOptions, setImportOptions] = useState<ImportOptions>({ createLabels: true, overwrite: false, markDone: false });
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [importBusy, setImportBusy] = useState(false);
  const [remoteModel, setRemoteModel] = useState('m');
  const [remoteEpochs, setRemoteEpochs] = useState('300');
  const [remotePatience, setRemotePatience] = useState('50');
  const [remoteImgsz, setRemoteImgsz] = useState('640');
  const [remoteResultsFile, setRemoteResultsFile] = useState<File | null>(null);
  const [remoteWeightsFile, setRemoteWeightsFile] = useState<File | null>(null);
  const [remoteSummary, setRemoteSummary] = useState<RemoteResultsSummary | null>(null);
  const [remoteBusy, setRemoteBusy] = useState(false);
  const remoteResultsInput = useRef<HTMLInputElement>(null);
  const remoteWeightsInput = useRef<HTMLInputElement>(null);
  const [remoteJob, setRemoteJob] = useState<RemoteJobStatus | null>(null);
  const [remoteTarget, setRemoteTarget] = useState(() => {
    try {
      return localStorage.getItem('labelapp.remoteTarget') ?? '';
    } catch {
      return '';
    }
  });
  const [remotePassword, setRemotePassword] = useState('');
  /** Only the newest preview may apply, since options can change while one is in flight. */
  const importRequest = useRef(0);
  /** Only the newest list request may apply, since filters can change while one is in flight. */
  const listRequest = useRef(0);

  const refreshProject = () => api.getProject(projectId).then(setProject);
  const refreshFolders = () => api.listFolders(projectId).then(setFolders);
  const refreshSources = () => api.listSources(projectId).then(setSources);
  const refreshAutoBox = () => api.getAutoBox(projectId).then(setAutoBox);
  const refreshRemote = () => api.getRemote(projectId).then(setRemoteJob);
  const refreshImages = () => {
    const request = ++listRequest.current;
    return api.listImages(projectId, { folder: folder ?? undefined, recursive, label: labelFilter || undefined }).then((list) => {
      if (request === listRequest.current) setImages(list);
    });
  };
  const report = (e: unknown) => setError((e as Error).message);

  // Choosing a file or changing an option previews the import, so the person sees what would change first.
  useEffect(() => {
    if (!importFile) return;
    const request = ++importRequest.current;
    setImportBusy(true);
    api
      .importAnnotations(projectId, importFile, { ...importOptions, dryRun: true })
      .then((result) => {
        if (request === importRequest.current) setImportResult(result);
      })
      .catch((e) => {
        if (request !== importRequest.current) return;
        setImportResult(null);
        report(e);
      })
      .finally(() => {
        if (request === importRequest.current) setImportBusy(false);
      });
  }, [projectId, importFile, importOptions]);

  const pickImportFile = (file: File | null) => {
    importRequest.current++;
    setImportResult(null);
    setError(null);
    setImportFile(file);
  };

  const runImport = async () => {
    if (!importFile) return;
    const request = ++importRequest.current;
    setImportBusy(true);
    try {
      const result = await api.importAnnotations(projectId, importFile, { ...importOptions, dryRun: false });
      if (request !== importRequest.current) return;
      setImportResult(result);
      setImportFile(null);
      if (importInput.current) importInput.current.value = '';
      setProject(result.project);
      await Promise.all([refreshImages(), refreshFolders()]);
    } catch (e) {
      report(e);
    } finally {
      if (request === importRequest.current) setImportBusy(false);
    }
  };

  useEffect(() => {
    saveQueue
      .flush()
      .then(() => Promise.all([refreshProject(), refreshFolders(), refreshSources()]))
      .catch(report);
  }, [projectId]);

  useEffect(() => {
    setSelection(null);
    setLimit(PAGE_SIZE);
    saveQueue.flush().then(refreshImages).catch(report);
  }, [projectId, folder, recursive, labelFilter]);

  // Every change to the image list can change these counts, so they follow it.
  useEffect(() => {
    refreshAutoBox().catch(report);
  }, [projectId, images]);
  useEffect(() => {
    refreshRemote().catch(report);
  }, [projectId]);

  const scanning = sources.some((s) => s.scan?.state === 'running');
  const autoBoxing = autoBox?.job?.state === 'running';
  const remoting = remoteJob?.state === 'running';
  useEffect(() => {
    if (!remoting) return;
    const timer = setInterval(() => refreshRemote().catch(report), 1000);
    return () => {
      clearInterval(timer);
      Promise.all([refreshRemote(), refreshImages(), refreshAutoBox()]).catch(report);
    };
  }, [remoting, folder, recursive, labelFilter]);
  useEffect(() => {
    if (!scanning && !autoBoxing) return;
    let nextListAt = Date.now() + 3000;
    const timer = setInterval(() => {
      refreshSources().catch(report);
      refreshFolders().catch(report);
      refreshAutoBox().catch(report);
      if (Date.now() < nextListAt) return;
      // The full list gets expensive as a scan grows, so refresh it less often the longer it takes.
      const started = Date.now();
      nextListAt = Infinity;
      refreshImages()
        .catch(report)
        .finally(() => {
          nextListAt = Date.now() + Math.max(3000, (Date.now() - started) * 10);
        });
    }, 1000);
    return () => {
      clearInterval(timer);
      Promise.all([refreshImages(), refreshFolders(), refreshProject()]).catch(report);
    };
  }, [scanning, autoBoxing, folder, recursive, labelFilter]);

  const mode = project && MODES[project.mode];
  const confidence = pickedConfidence ?? autoBox?.job?.confidence ?? autoBox?.defaultConfidence ?? 0.5;

  const ownCheckpoints = autoBox?.checkpoints ?? [];
  const otherCheckpoints = autoBox?.otherCheckpoints ?? [];
  // Another project's checkpoint can draw here too, with its classes matched to labels by name.
  const checkpoints = [...ownCheckpoints, ...otherCheckpoints];
  const baseCheckpoint = autoBox?.baseCheckpoint ?? '';
  // A pick can stop being valid when its checkpoint is deleted or labels change.
  const trainFrom = pickedTrainFrom === baseCheckpoint || checkpoints.some((c) => c.id === pickedTrainFrom && c.canContinue) ? pickedTrainFrom! : (autoBox?.defaultCheckpoint ?? '');
  const drawWith = checkpoints.some((c) => c.id === pickedDrawWith) ? pickedDrawWith! : (checkpoints[0]?.id ?? '');
  const drawing = checkpoints.find((c) => c.id === drawWith);
  const fromBase = trainFrom === baseCheckpoint;
  const defaultPatience = autoBox?.defaultPatience ?? 0;
  const patienceText = fromBase ? String(defaultPatience) : (pickedPatience ?? String(defaultPatience));
  const patience = Number(patienceText);
  const patienceValid = Number.isInteger(patience) && patience >= 1 && patience <= (autoBox?.maxEpochs ?? 0);
  const cannotContinue = ownCheckpoints.filter((c) => !c.canContinue).length;

  const startRemote = async () => {
    setError(null);
    try {
      try {
        localStorage.setItem('labelapp.remoteTarget', remoteTarget);
      } catch {}
      const status = await api.startRemote(projectId, {
        target: remoteTarget,
        password: remotePassword,
        model: remoteModel,
        epochs: Number(remoteEpochs),
        patience: Number(remotePatience),
        imgsz: Number(remoteImgsz),
        confidence,
        checkpoint: fromBase ? undefined : trainFrom,
      });
      // The password was for this one login; the page forgets it as the server does.
      setRemotePassword('');
      setRemoteJob(status);
    } catch (e) {
      report(e);
    }
  };

  const cancelRemote = async () => {
    await api.cancelRemote(projectId).catch(report);
    await refreshRemote().catch(report);
  };

  const importRemoteResults = async () => {
    if (!remoteResultsFile) return;
    setRemoteBusy(true);
    setError(null);
    try {
      const summary = await api.importRemoteResults(projectId, remoteResultsFile, remoteWeightsFile);
      setRemoteSummary(summary);
      setRemoteResultsFile(null);
      setRemoteWeightsFile(null);
      if (remoteResultsInput.current) remoteResultsInput.current.value = '';
      if (remoteWeightsInput.current) remoteWeightsInput.current.value = '';
      await Promise.all([refreshImages(), refreshAutoBox()]);
    } catch (e) {
      report(e);
    } finally {
      setRemoteBusy(false);
    }
  };

  const startAutoBox = async (mode: 'train' | 'predict') => {
    setError(null);
    try {
      // An edit made just before leaving the annotator should count towards this run.
      await saveQueue.flush();
      if (mode === 'predict') {
        await api.startAutoBox(projectId, { mode, confidence, checkpoint: drawWith });
        await refreshProject();
      }
      else await api.startAutoBox(projectId, { mode, confidence, checkpoint: trainFrom, patience: fromBase ? undefined : patience });
      // A small job can finish before the first poll, so polling alone might never refresh the list.
      await refreshAutoBox();
      await refreshImages();
    } catch (e) {
      report(e);
    }
  };

  const cancelAutoBox = async () => {
    setError(null);
    await api.cancelAutoBox(projectId).catch(report);
    await refreshAutoBox().catch(report);
  };

  const deleteAutoBoxCheckpoint = async () => {
    if (!window.confirm('Delete this checkpoint? Other checkpoints and boxes already drawn are kept. This cannot be undone.')) return;
    setError(null);
    await api.deleteAutoBoxCheckpoint(projectId, drawWith).catch(report);
    await refreshAutoBox().catch(report);
  };

  const clearAutoBoxes = async () => {
    const n = autoBox?.autoBoxed ?? 0;
    if (!window.confirm(`Remove the predicted boxes from ${count(n)}? Boxes you edited, and images marked done, are kept. This cannot be undone.`)) return;
    setError(null);
    try {
      await api.clearAutoBoxes(projectId);
      await refreshImages();
    } catch (e) {
      report(e);
    }
  };

  const upload = async (files: File[]) => {
    if (files.length === 0) return;
    setError(null);
    const tooLarge = files.filter((f) => f.size > MAX_FILE_BYTES);
    const failures: UploadResult['rejected'] = tooLarge.map((f) => ({ filename: f.name, reason: 'Larger than the 100 MB limit' }));
    const accepted = files.filter((f) => f.size <= MAX_FILE_BYTES);

    setProgress({ done: 0, total: accepted.length });
    let done = 0;
    for (const batch of batches(accepted)) {
      try {
        const result = await api.uploadImages(projectId, batch);
        failures.push(...result.rejected);
      } catch (e) {
        failures.push(...batch.map((f) => ({ filename: f.name, reason: (e as Error).message })));
      }
      done += batch.length;
      setProgress({ done, total: accepted.length });
    }
    setProgress(null);
    setRejected(failures);
    await Promise.all([refreshImages(), refreshFolders(), refreshProject()]);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    void upload([...e.dataTransfer.files]);
  };

  /**
   * A small scan can finish before the first poll, so polling alone might never refresh the lists.
   * Scan status is read first: if it still says running, polling takes over and refreshes at the end.
   */
  const refreshAfterScanStart = async () => {
    await refreshSources();
    await Promise.all([refreshImages(), refreshFolders(), refreshProject()]);
  };

  const addSource = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await api.addSource(projectId, sourcePath.trim());
      setSourcePath('');
      await refreshAfterScanStart();
    } catch (err) {
      report(err);
    }
  };

  const rescan = async (source: Source, includeExcluded: boolean) => {
    setError(null);
    try {
      await api.rescan(source.id, includeExcluded);
      await refreshAfterScanStart();
    } catch (err) {
      report(err);
    }
  };

  const moveSource = async (source: Source) => {
    const path = window.prompt(`Where is the folder "${source.name}" now? Enter its absolute path. Annotations are kept, and each file is checked again.`, source.path);
    if (!path?.trim() || path.trim() === source.path) return;
    setError(null);
    try {
      await api.moveSource(source.id, path.trim());
      await refreshAfterScanStart();
    } catch (err) {
      report(err);
    }
  };

  const removeSource = async (source: Source) => {
    const message = `Remove ${source.path} from this project, with its ${count(source.imageCount)} and their annotations? The files stay on disk. This cannot be undone.`;
    if (!window.confirm(message)) return;
    try {
      await api.removeSource(source.id);
      setFolder(null);
      await Promise.all([refreshSources(), refreshFolders(), refreshImages(), refreshProject()]);
    } catch (err) {
      report(err);
    }
  };

  const removeImage = async (image: ImageMeta) => {
    const n = image.annotationCount;
    const detail = n > 0 ? ` and its ${n} ${n === 1 ? 'annotation' : 'annotations'}` : '';
    const ending = image.sourceId ? 'The file stays on disk.' : 'This cannot be undone.';
    if (!window.confirm(`Delete ${image.filename}${detail} from this project? ${ending}`)) return;
    try {
      await api.deleteImage(image.id);
      setImages((list) => list.filter((i) => i.id !== image.id));
      await Promise.all([refreshFolders(), refreshProject()]);
    } catch (e) {
      report(e);
    }
  };

  const removeProject = async () => {
    if (!project) return;
    const onDisk = sources.length > 0 ? ' Files in added folders stay on disk.' : '';
    if (!window.confirm(`Delete project "${project.name}", its images and all annotations?${onDisk} This cannot be undone.`)) return;
    try {
      await api.deleteProject(projectId);
      navigate('/');
    } catch (e) {
      report(e);
    }
  };

  const folderFilter = folder ? { folder, recursive } : null;

  const startAnnotating = () => {
    const next = images.find((i) => i.status === 'todo') ?? images[0];
    if (next) navigate(`/projects/${projectId}/annotate/${next.id}${filterQuery(folderFilter)}`);
  };

  const visible = filter === 'all' ? images : images.filter((i) => (filter === 'auto' ? i.autoBoxed : i.status === filter));
  const doneCount = images.filter((i) => i.status === 'done').length;
  const autoBoxedCount = images.filter((i) => i.autoBoxed).length;
  const labels: Label[] = project?.labels ?? [];

  const isSelected = (image: ImageMeta) => {
    if (!selection) return false;
    return selection.kind === 'folder' ? !selection.except.has(image.id) : selection.ids.has(image.id);
  };
  const selected = useMemo(() => images.filter(isSelected), [images, selection]);
  const selectedLabelCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const image of selected) {
      for (const id of image.imageLabels) counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    return counts;
  }, [selected]);

  const selectAll = () => {
    // A folder selector only describes this list when no other filter narrows it.
    if (folder && filter === 'all' && !labelFilter) setSelection({ kind: 'folder', folder, recursive, except: new Set() });
    else setSelection({ kind: 'ids', ids: new Set(visible.map((i) => i.id)) });
  };

  const toggle = (image: ImageMeta) => {
    if (selection?.kind === 'folder') {
      const except = new Set(selection.except);
      if (except.has(image.id)) except.delete(image.id);
      else except.add(image.id);
      setSelection({ ...selection, except });
    } else {
      const ids = new Set(selection?.ids);
      if (ids.has(image.id)) ids.delete(image.id);
      else ids.add(image.id);
      setSelection(ids.size > 0 ? { kind: 'ids', ids } : null);
    }
  };

  const selector = (): BulkRequest['selector'] =>
    selection?.kind === 'folder'
      ? { folder: selection.folder, recursive: selection.recursive, except: [...selection.except] }
      : { imageIds: selected.map((i) => i.id) };

  const runBulk = async (request: BulkRequest, describe: (changed: number) => string, inverse: (ids: string[]) => BulkRequest) => {
    setError(null);
    try {
      const result = await api.bulkUpdate(projectId, request);
      setUndoable(result.changed.length > 0 ? { message: describe(result.changed.length), request: inverse(result.changed) } : null);
      await Promise.all([refreshImages(), refreshProject()]);
    } catch (e) {
      report(e);
    }
  };

  const chosenLabel = labels.find((l) => l.id === bulkLabel);
  const newLabelColor = pickedColor ?? nextLabelColor(labels);

  const applyAddLabel = async () => {
    let label = chosenLabel;
    if (bulkLabel === NEW_LABEL) {
      label = { id: newId(), name: newLabelName.trim(), color: newLabelColor, keypoints: [], skeleton: [] };
      try {
        const updated = await api.updateProject(projectId, { labels: [...labels, label] });
        setProject(updated.project);
        setBulkLabel(label.id);
        setNewLabelName('');
        setPickedColor(null);
      } catch (e) {
        report(e);
        return;
      }
    }
    if (!label) return;
    const { id, name } = label;
    await runBulk(
      { selector: selector(), addLabels: [id] },
      (n) => `Added ${name} to ${count(n)}.`,
      (imageIds) => ({ selector: { imageIds }, removeLabels: [id] }),
    );
  };

  const applyRemoveLabel = async ({ id, name }: Label) => {
    await runBulk(
      { selector: selector(), removeLabels: [id] },
      (n) => `Removed ${name} from ${count(n)}.`,
      (imageIds) => ({ selector: { imageIds }, addLabels: [id] }),
    );
  };

  const applyStatus = async (status: ImageStatus) => {
    const affected = selected.filter((i) => i.status !== status);
    let message = `Mark ${count(affected.length)} as ${status}?`;
    if (status === 'done') {
      const empty = affected.filter((i) => i.annotationCount === 0).length;
      if (empty > 0) message += ` ${count(empty)} with no annotations will count as containing no objects in COCO and YOLO exports.`;
    }
    if (!window.confirm(message)) return;
    // Only images whose status differed are changed, so the undo is the opposite status for all of them.
    await runBulk(
      { selector: selector(), status },
      (n) => `Marked ${count(n)} as ${status}.`,
      (imageIds) => ({ selector: { imageIds }, status: status === 'done' ? 'todo' : 'done' }),
    );
  };

  const undoBulk = async () => {
    if (!undoable) return;
    const { request } = undoable;
    setUndoable(null);
    try {
      await api.bulkUpdate(projectId, request);
      await Promise.all([refreshImages(), refreshProject()]);
    } catch (e) {
      report(e);
    }
  };

  const annotateSelection = () => {
    if (selection?.kind !== 'folder') return;
    const next = images.find((i) => i.status === 'todo' && isSelected(i)) ?? selected[0];
    if (next) navigate(`/projects/${projectId}/annotate/${next.id}${filterQuery({ folder: selection.folder, recursive: selection.recursive })}`);
  };

  if (error && !project) {
    return (
      <div className="page">
        <p className="error">{error}</p>
        <a href="#/">All projects</a>
      </div>
    );
  }

  const addCount = bulkLabel === NEW_LABEL ? selected.length : selected.filter((i) => !i.imageLabels.includes(bulkLabel)).length;
  const canAdd = bulkLabel === NEW_LABEL ? newLabelName.trim() !== '' : Boolean(chosenLabel);
  const labelsOnSelection = labels.filter((l) => selectedLabelCounts.has(l.id));

  const exportParams = new URLSearchParams(
    format === 'project'
      ? { format }
      : {
          format,
          scope: format === 'image_labels' ? labeledScope : scope,
          includeImages: (format === 'yolo' || format === 'image_labels') && includeImages ? '1' : '0',
        },
  );

  const hidden = (node: FolderNode): boolean => {
    // A folder's ancestors are simply the prefixes of its path.
    for (let end = node.path.lastIndexOf('/'); end > 0; end = node.path.lastIndexOf('/', end - 1)) {
      if (collapsed.has(node.path.slice(0, end))) return true;
    }
    return false;
  };

  return (
    <div
      className={`page ${dragOver ? 'drag-over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(false);
      }}
      onDrop={onDrop}
    >
      <header className="page-header">
        <div>
          <a className="muted" href="#/">
            ← All projects
          </a>
          <h1>{project?.name}</h1>
          {mode && (
            <span className="muted" data-testid="project-mode">
              {mode.name}
            </span>
          )}
        </div>
        <div className="spacer" />
        <a className="btn" href={`#/projects/${projectId}/settings`}>
          Labels
        </a>
        <button className="btn" onClick={() => fileInput.current?.click()} disabled={progress !== null}>
          Upload images
        </button>
        <button className="btn primary" onClick={startAnnotating} disabled={images.length === 0}>
          Start annotating
        </button>
      </header>
      <input
        ref={fileInput}
        type="file"
        data-testid="file-input"
        multiple
        accept="image/jpeg,image/png,image/webp,image/gif,image/bmp"
        hidden
        onChange={(e) => {
          void upload([...(e.target.files ?? [])]);
          e.target.value = '';
        }}
      />

      {error && <p className="error" role="alert">{error}</p>}
      {project?.labels.length === 0 && (
        <p className="callout">
          No labels defined yet. <a href={`#/projects/${projectId}/settings`}>Add labels</a> before annotating.
        </p>
      )}
      {progress && (
        <p className="callout">
          Uploading {progress.done} of {progress.total}… <progress value={progress.done} max={progress.total} />
        </p>
      )}
      {rejected.length > 0 && (
        <div className="callout error">
          <strong>
            {rejected.length} {rejected.length === 1 ? 'file was' : 'files were'} not uploaded
          </strong>
          <ul>
            {rejected.slice(0, 20).map((r, i) => (
              <li key={i}>
                {r.filename}: {r.reason}
              </li>
            ))}
            {rejected.length > 20 && <li>…and {rejected.length - 20} more</li>}
          </ul>
          <button className="btn" onClick={() => setRejected([])}>
            Dismiss
          </button>
        </div>
      )}

      <section className="sources" aria-label="Folders on disk">
        <h2>Folders on disk</h2>
        <form className="inline-form" onSubmit={addSource}>
          <input value={sourcePath} onChange={(e) => setSourcePath(e.target.value)} placeholder="/absolute/path/to/images" aria-label="Folder path" />
          <button className="btn" disabled={!sourcePath.trim()}>
            Add folder
          </button>
        </form>
        <p className="muted">Finds images in the folder and all subfolders. Images stay where they are and are never changed.</p>
        {sources.map((source) => {
          const scan = source.scan;
          const running = scan?.state === 'running';
          return (
            <div className="source" key={source.id} data-testid="source">
              <span className="path mono" title={source.name}>
                {source.path}
              </span>
              <span className="muted">{count(source.imageCount)}</span>
              {running ? (
                <button className="btn" onClick={() => api.cancelScan(source.id).then(refreshSources).catch(report)}>
                  Cancel
                </button>
              ) : (
                <>
                  <button className="btn" onClick={() => rescan(source, false)}>
                    Rescan
                  </button>
                  <button className="btn" title="Also brings back images you deleted from this project" onClick={() => rescan(source, true)}>
                    Rescan and restore deleted
                  </button>
                  <button className="btn" title="Point this folder at another path on disk" onClick={() => moveSource(source)}>
                    Change path
                  </button>
                </>
              )}
              <button className="btn danger" onClick={() => removeSource(source)}>
                Remove
              </button>
              {scan && (
                <span className="scan-line" data-testid="scan-line">
                  {running ? 'Scanning' : scan.state === 'done' ? 'Scan finished' : scan.state === 'cancelled' ? 'Scan cancelled' : 'Scan failed'}: {scan.foldersVisited} folders, {scan.added} added
                  {scan.unchanged > 0 && `, ${scan.unchanged} unchanged`}
                  {scan.skipped > 0 && `, ${scan.skipped} other files skipped`}
                  {scan.missing > 0 && `, ${scan.missing} missing`}
                  {scan.changed > 0 && `, ${scan.changed} changed`}
                  {scan.recovered > 0 && `, ${scan.recovered} recovered`}
                  {scan.excluded > 0 && `, ${scan.excluded} previously deleted`}
                  {scan.errorCount > 0 && `, ${scan.errorCount} errors`}
                </span>
              )}
              {scan && scan.errors.length > 0 && (
                <ul className="scan-errors">
                  {scan.errors.slice(0, 10).map((e, i) => (
                    <li key={i}>
                      {e.path}: {e.reason}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </section>

      {mode?.autoBox && autoBox && (project?.imageCount ?? 0) > 0 && (
        <section className="sources" aria-label={mode.autoName}>
          <h2>{mode.autoName}</h2>
          <p className="muted">
            Trains a YOLOv8{mode.modelSuffix} nano model on the {count(autoBox.trainable)} with saved {mode.autoUnit[1]}, then draws {mode.autoUnit[1]} on {count(autoBox.targets)} still todo that you have not annotated. Drawn {mode.autoUnit[1]} are marked <em>auto</em>, and
            drawing again replaces them. They are not exported or trained on until you edit the image or mark it done.
          </p>
          <div className="autobox-confidence">
            <label>
              Confidence
              <input type="range" min={0.05} max={0.95} step={0.05} value={confidence} disabled={autoBoxing} onChange={(e) => setPickedConfidence(Number(e.target.value))} />
            </label>
            <span className="value" data-testid="autobox-confidence">
              {confidence.toFixed(2)}
            </span>
            <span className="muted">Detections below this are not drawn. Lower finds more objects, and more mistakes.</span>
          </div>
          <div className="autobox-actions">
            <label>
              Train from
              <select value={trainFrom} disabled={autoBoxing} onChange={(e) => setPickedTrainFrom(e.target.value)}>
                <option value={autoBox.baseCheckpoint}>{autoBox.baseCheckpoint} (base model)</option>
                {ownCheckpoints.map((c) => (
                  <option key={c.id} value={c.id} disabled={!c.canContinue}>
                    {describeCheckpoint(c)}
                    {c.canContinue ? '' : ' (labels changed)'}
                  </option>
                ))}
                {otherCheckpoints.length > 0 && (
                  <optgroup label="From other projects">
                    {otherCheckpoints.map((c) => (
                      <option key={c.id} value={c.id} disabled={!c.canContinue}>
                        {c.projectName}: {describeCheckpoint(c)}
                        {c.canContinue ? '' : ' (different labels)'}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </label>
            <label>
              Patience
              <input type="number" min={1} max={autoBox.maxEpochs} step={1} value={patienceText} disabled={autoBoxing || fromBase} onChange={(e) => setPickedPatience(e.target.value)} />
            </label>
            <span className="muted">{fromBase ? `Fixed at ${defaultPatience} for ${autoBox.baseCheckpoint}.` : `Epochs without improvement before training stops, from 1 to ${autoBox.maxEpochs}.`}</span>
            {!autoBoxing && (
              <button className="btn primary" disabled={autoBox.trainable < autoBox.minTraining || autoBox.targets === 0 || !patienceValid} onClick={() => startAutoBox('train')}>
                Train and draw boxes
              </button>
            )}
          </div>
          <div className="autobox-actions">
            <label>
              Draw with
              <select value={drawWith} disabled={autoBoxing || checkpoints.length === 0} onChange={(e) => setPickedDrawWith(e.target.value)}>
                {checkpoints.length === 0 && <option value="">No checkpoints</option>}
                {ownCheckpoints.map((c) => (
                  <option key={c.id} value={c.id}>
                    {describeCheckpoint(c)}
                  </option>
                ))}
                {otherCheckpoints.length > 0 && (
                  <optgroup label="From other projects">
                    {otherCheckpoints.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.projectName}: {describeCheckpoint(c)}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </label>
            {!autoBoxing && (
              <button className="btn" disabled={checkpoints.length === 0 || autoBox.targets === 0} onClick={() => startAutoBox('predict')}>
                Draw with trained model
              </button>
            )}
            {!autoBoxing && drawing && drawing.projectId === projectId && (
              <button className="btn danger" onClick={deleteAutoBoxCheckpoint}>
                Delete checkpoint
              </button>
            )}
            {drawing && drawing.missingLabels.length > 0 && (
              <span className="scan-line" data-testid="autobox-adds-labels">
                Drawing with this model from {drawing.projectName} adds the {drawing.missingLabels.length === 1 ? 'label' : 'labels'} {drawing.missingLabels.join(', ')} to this project.
              </span>
            )}
            {(checkpoints.length === 0 || cannotContinue > 0) && (
              <span className="scan-line" data-testid="autobox-model">
                {checkpoints.length === 0
                  ? 'No trained model yet.'
                  : `Labels were added or removed since ${cannotContinue === 1 ? '1 checkpoint was' : `${cannotContinue} checkpoints were`} trained, so training cannot continue from ${cannotContinue === 1 ? 'it' : 'them'}. Drawing still works, but never predicts a newer label.`}
              </span>
            )}
          </div>
          <div className="autobox-actions">
            {autoBoxing && (
              <button className="btn" onClick={cancelAutoBox}>
                Cancel
              </button>
            )}
            {!autoBoxing && autoBox.autoBoxed > 0 && (
              <button className="btn danger" onClick={clearAutoBoxes}>
                Remove predicted {mode.autoUnit[1]} ({autoBox.autoBoxed.toLocaleString()})
              </button>
            )}
            {!autoBoxing && autoBox.trainable < autoBox.minTraining && <span className="muted">Training needs at least {autoBox.minTraining} images with saved {mode.autoUnit[1]}.</span>}
          </div>
          <details className="remote-training" data-testid="remote-training" open={remoting || remoteJob !== null}>
            <summary>Train on a remote server</summary>
            <p className="muted">
              A larger model can train on a machine with a GPU. The app bundles only the {count(autoBox.trainable)} with saved {mode.autoUnit[1]}, uploads them over SSH, trains there, downloads the trained model as a checkpoint, and then draws here with it on the{' '}
              {count(autoBox.targets)} still todo, exactly as a local run would. The rest of the dataset never leaves this machine. The password is used for one login and is not stored; leave it empty to use your SSH key. Training starts from{' '}
              {fromBase ? `the base yolov8${remoteModel}${mode.modelSuffix}.pt weights` : 'the checkpoint chosen in Train from, which fixes the model size'}.
            </p>
            <div className="autobox-actions">
              <label>
                Server
                <input type="text" placeholder="user@host" value={remoteTarget} disabled={remoting} onChange={(e) => setRemoteTarget(e.target.value)} />
              </label>
              <label>
                Password
                <input type="password" autoComplete="off" value={remotePassword} disabled={remoting} onChange={(e) => setRemotePassword(e.target.value)} />
              </label>
              <label>
                Remote model
                <select value={remoteModel} disabled={!fromBase || remoting} onChange={(e) => setRemoteModel(e.target.value)}>
                  <option value="n">YOLOv8{mode.modelSuffix} nano</option>
                  <option value="s">YOLOv8{mode.modelSuffix} small</option>
                  <option value="m">YOLOv8{mode.modelSuffix} medium</option>
                  <option value="l">YOLOv8{mode.modelSuffix} large</option>
                  <option value="x">YOLOv8{mode.modelSuffix} extra large</option>
                </select>
              </label>
              <label>
                Remote epochs
                <input type="number" min={1} max={1000} step={1} value={remoteEpochs} disabled={remoting} onChange={(e) => setRemoteEpochs(e.target.value)} />
              </label>
              <label>
                Remote patience
                <input type="number" min={1} max={1000} step={1} value={remotePatience} disabled={remoting} onChange={(e) => setRemotePatience(e.target.value)} />
              </label>
              <label>
                Remote image size
                <select value={remoteImgsz} disabled={remoting} onChange={(e) => setRemoteImgsz(e.target.value)}>
                  {[480, 640, 800, 960, 1280].map((n) => (
                    <option key={n} value={String(n)}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="autobox-actions">
              {!remoting && (
                <button className="btn primary" disabled={autoBoxing || !remoteTarget.trim() || autoBox.trainable < autoBox.minTraining} onClick={startRemote}>
                  Train on server and draw
                </button>
              )}
              {remoting && (
                <button className="btn" onClick={cancelRemote}>
                  Cancel
                </button>
              )}
              {remoteJob && (
                <span className="scan-line" data-testid="remote-status" role={remoteJob.state === 'failed' ? 'alert' : 'status'}>
                  {describeRemote(remoteJob, mode.autoUnit)}
                </span>
              )}
            </div>
            <details className="remote-manual">
              <summary>Manual steps</summary>
              <p className="muted">The same run by hand: download the bundle, run the script in a terminal, import the model it brings back, and then use Draw with trained model.</p>
              <div className="autobox-actions">
                <a
                  className={`btn${autoBox.trainable < autoBox.minTraining ? ' disabled' : ''}`}
                  aria-disabled={autoBox.trainable < autoBox.minTraining}
                  href={`/api/projects/${projectId}/remote/bundle?${new URLSearchParams({
                    model: remoteModel,
                    epochs: remoteEpochs,
                    patience: remotePatience,
                    imgsz: remoteImgsz,
                    confidence: confidence.toFixed(2),
                    ...(fromBase ? {} : { checkpoint: trainFrom }),
                  })}`}
                  download
                  onClick={(e) => {
                    if (autoBox.trainable < autoBox.minTraining) e.preventDefault();
                  }}
                >
                  Download bundle
                </a>
                <code>conda activate ml && python scripts/remote_train.py train ~/Downloads/{'<'}bundle{'>'}.zip {remoteTarget.trim() || 'user@host'}</code>
              </div>
              <div className="autobox-actions">
                <label>
                  results.json
                  <input ref={remoteResultsInput} type="file" accept=".json,application/json" data-testid="remote-results" onChange={(e) => setRemoteResultsFile(e.target.files?.[0] ?? null)} />
                </label>
                <label>
                  best.pt
                  <input ref={remoteWeightsInput} type="file" accept=".pt" data-testid="remote-weights" onChange={(e) => setRemoteWeightsFile(e.target.files?.[0] ?? null)} />
                </label>
                <button className="btn" disabled={!remoteResultsFile || remoteBusy || autoBoxing || remoting} onClick={importRemoteResults}>
                  Import results
                </button>
              </div>
              {remoteSummary && (
                <p className="scan-line" data-testid="remote-summary" role="status">
                  {describeRemoteResults(remoteSummary)}
                </p>
              )}
            </details>
          </details>
          {autoBox.job && (
            <p className="scan-line" data-testid="autobox-status" role={autoBox.job.state === 'failed' ? 'alert' : undefined}>
              {describeAutoBox(autoBox.job)}
            </p>
          )}
          {autoBox.job && autoBox.job.errors.length > 0 && (
            <ul className="scan-errors">
              {autoBox.job.errors.slice(0, 10).map((e, i) => (
                <li key={i}>
                  {e.path}: {e.reason}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      <section className="export-bar" aria-label="Export">
        <strong>Export</strong>
        <label>
          Format
          <select value={format} onChange={(e) => setFormat(e.target.value as typeof format)}>
            <option value="coco">{mode?.cocoFormat}</option>
            <option value="yolo">{mode?.yoloFormat}</option>
            <option value="image_labels">Image labels (zip)</option>
            <option value="json">LabelApp JSON</option>
            <option value="project">Project file, to share with teammates</option>
          </select>
        </label>
        {format === 'project' ? (
          <span className="muted" data-testid="project-export-note">
            Every image's labels, annotations and status, without the image files. A teammate opens it from the project list and chooses where these folders are on their computer.
          </span>
        ) : (
          <label>
            Images
            {format === 'image_labels' ? (
              <select value={labeledScope} onChange={(e) => setLabeledScope(e.target.value as typeof labeledScope)}>
                <option value="labeled">All with an image label</option>
                <option value="done">Done, with an image label</option>
              </select>
            ) : (
              <select value={scope} onChange={(e) => setScope(e.target.value as typeof scope)}>
                <option value="done">Done only</option>
                <option value="annotated">Done, plus todo with annotations</option>
              </select>
            )}
          </label>
        )}
        {(format === 'yolo' || format === 'image_labels') && (
          <label className="check">
            <input type="checkbox" checked={includeImages} onChange={(e) => setIncludeImages(e.target.checked)} />
            Include image files
          </label>
        )}
        <a className="btn" href={`/api/projects/${projectId}/export?${exportParams}`} download>
          Download
        </a>
      </section>

      <section className="export-bar" aria-label="Import annotations">
        <strong>Import</strong>
        <label>
          {mode?.cocoFormat.replace(' JSON', '')} or LabelApp JSON
          <input
            ref={importInput}
            type="file"
            accept=".json,application/json"
            data-testid="import-file"
            onChange={(e) => pickImportFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="check">
          <input type="checkbox" checked={importOptions.createLabels} onChange={(e) => setImportOptions({ ...importOptions, createLabels: e.target.checked })} />
          Create missing labels
        </label>
        <label className="check">
          <input type="checkbox" checked={importOptions.overwrite} onChange={(e) => setImportOptions({ ...importOptions, overwrite: e.target.checked })} />
          Replace existing annotations
        </label>
        <label className="check">
          <input type="checkbox" checked={importOptions.markDone} onChange={(e) => setImportOptions({ ...importOptions, markDone: e.target.checked })} />
          Mark imported images done
        </label>
        <button
          className="btn primary"
          disabled={!importFile || importBusy || !importResult?.dryRun || importResult.summary.imagesWritten + importResult.summary.newLabels.length === 0}
          onClick={runImport}
        >
          Import
        </button>
        {importResult && (
          <div className="scan-line" data-testid="import-summary" role="status">
            {describeImport(importResult.summary, importResult.dryRun).map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            {importResult.summary.warnings.length > 0 && (
              <details>
                <summary>{importResult.summary.warnings.length} detail{importResult.summary.warnings.length === 1 ? '' : 's'}</summary>
                <ul className="scan-errors">
                  {importResult.summary.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </section>

      {undoable && (
        <p className="callout" role="status">
          {undoable.message}{' '}
          <button className="btn" onClick={undoBulk}>
            Undo
          </button>
        </p>
      )}

      <div className={folders.length > 0 ? 'project-body' : ''}>
        {folders.length > 0 && (
          <nav className="folder-tree" aria-label="Folders">
            <div className="folder-row">
              <span className="twist" />
              <button className={`open ${folder === null ? 'current' : ''}`} onClick={() => setFolder(null)}>
                <span className="name">All images</span>
              </button>
            </div>
            {folders
              .filter((node) => !hidden(node))
              .map((node) => {
                const depth = node.path.split('/').length - 1;
                const hasChildren = node.total > node.direct;
                return (
                  <div className="folder-row" key={node.path} style={{ paddingLeft: depth * 12 }}>
                    {hasChildren ? (
                      <button
                        className="twist"
                        aria-label={`${collapsed.has(node.path) ? 'Expand' : 'Collapse'} ${node.name}`}
                        onClick={() => {
                          const next = new Set(collapsed);
                          if (next.has(node.path)) next.delete(node.path);
                          else next.add(node.path);
                          setCollapsed(next);
                        }}
                      >
                        {collapsed.has(node.path) ? '▸' : '▾'}
                      </button>
                    ) : (
                      <span className="twist" />
                    )}
                    <button className={`open ${folder === node.path ? 'current' : ''}`} data-folder={node.path} onClick={() => setFolder(node.path)}>
                      <span className="name" title={node.path}>
                        {node.name}
                      </span>
                      <span className="muted">{node.total.toLocaleString()}</span>
                    </button>
                  </div>
                );
              })}
          </nav>
        )}

        <div>
          <div className="tabs" role="tablist">
            {(
              [
                ['all', `All (${images.length})`],
                ['todo', `Todo (${images.length - doneCount})`],
                ['done', `Done (${doneCount})`],
                // A keypoint project can still hold predictions from before modes existed.
                ...(mode?.autoBox || autoBoxedCount > 0 ? ([['auto', `Auto (${autoBoxedCount})`]] as const) : []),
              ] as const
            ).map(([id, text]) => (
              <button
                key={id}
                role="tab"
                aria-selected={filter === id}
                className={filter === id ? 'current' : ''}
                onClick={() => {
                  setFilter(id);
                  setSelection(null);
                }}
              >
                {text}
              </button>
            ))}
          </div>

          {(images.length > 0 || folder || labelFilter) && (
            <div className="grid-toolbar">
              {folder && (
                <label>
                  <input type="checkbox" checked={recursive} onChange={(e) => setRecursive(e.target.checked)} />
                  Include subfolders
                </label>
              )}
              <label>
                Image label
                <select value={labelFilter} onChange={(e) => setLabelFilter(e.target.value)} aria-label="Filter by image label">
                  <option value="">Any</option>
                  {labels.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.name}
                    </option>
                  ))}
                </select>
              </label>
              <button className="btn" disabled={visible.length === 0} onClick={selectAll}>
                {folder ? 'Select all in folder' : 'Select all'} ({visible.length.toLocaleString()})
              </button>
            </div>
          )}

          {selection && (
            <div className="bulk-bar" data-testid="bulk-bar">
              <span className="count">{count(selected.length)} selected</span>
              <select value={bulkLabel} onChange={(e) => setBulkLabel(e.target.value)} aria-label="Label for selected images">
                <option value="">Choose label…</option>
                {labels.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.name}
                  </option>
                ))}
                <option value={NEW_LABEL}>New label…</option>
              </select>
              {bulkLabel === NEW_LABEL && (
                <>
                  <input value={newLabelName} placeholder="Label name" aria-label="New label name" onChange={(e) => setNewLabelName(e.target.value)} />
                  <input type="color" value={newLabelColor} aria-label="New label color" onChange={(e) => setPickedColor(e.target.value)} />
                </>
              )}
              <button className="btn primary" disabled={!canAdd || addCount === 0} onClick={applyAddLabel}>
                Add to {count(addCount)}
              </button>
              <div className="spacer" />
              <button className="btn" onClick={() => applyStatus('done')}>
                Mark done
              </button>
              <button className="btn" onClick={() => applyStatus('todo')}>
                Mark todo
              </button>
              {selection.kind === 'folder' && (
                <button className="btn" onClick={annotateSelection}>
                  Annotate selection
                </button>
              )}
              <button className="btn ghost" onClick={() => setSelection(null)}>
                Clear
              </button>
              {labelsOnSelection.length > 0 && (
                <div className="bulk-labels">
                  <span className="muted">Labels on selection:</span>
                  {labelsOnSelection.map((label) => {
                    const action = `Remove ${label.name} from ${count(selectedLabelCounts.get(label.id) ?? 0)}`;
                    return (
                      <button key={label.id} className="label-remove" title={action} aria-label={action} onClick={() => applyRemoveLabel(label)}>
                        <span className="swatch" style={{ background: label.color }} />
                        {label.name}
                        <span className="muted">{(selectedLabelCounts.get(label.id) ?? 0).toLocaleString()}</span>
                        <span aria-hidden="true">×</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {images.length === 0 && !folder && !labelFilter ? (
            <button className="dropzone" onClick={() => fileInput.current?.click()}>
              Drop images here, or click to choose files
              <span className="muted">JPEG, PNG, WebP, GIF or BMP, up to 100 MB each</span>
            </button>
          ) : (
            <ul className="image-grid">
              {visible.slice(0, limit).map((image) => (
                <li key={image.id} className={isSelected(image) ? 'selected' : ''}>
                  <a href={`#/projects/${projectId}/annotate/${image.id}${filterQuery(folderFilter)}`}>
                    <img src={imageUrl(image.id)} alt={image.filename} loading="lazy" />
                    <span className={`badge ${image.status}`}>{image.status}</span>
                    {image.autoBoxed && <span className="badge auto">auto</span>}
                    {image.fileState !== 'ok' && <span className={`badge ${image.fileState}`}>{image.fileState}</span>}
                    <span className="caption">
                      <span className="name" title={`${image.folder}/${image.filename}`}>
                        {image.filename}
                      </span>
                      <span className="tag-dots">
                        {image.imageLabels.map((id) => {
                          const label = labels.find((l) => l.id === id);
                          return <i key={id} title={label?.name} style={{ background: label?.color }} />;
                        })}
                      </span>
                      <span className="muted">{image.annotationCount}</span>
                    </span>
                  </a>
                  <input type="checkbox" className="pick" aria-label={`Select ${image.filename}`} checked={isSelected(image)} onChange={() => toggle(image)} />
                  <button className="delete" aria-label={`Delete ${image.filename}`} onClick={() => removeImage(image)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          {visible.length > limit && (
            <button className="btn more" onClick={() => setLimit((l) => l + PAGE_SIZE)}>
              Show more ({(visible.length - limit).toLocaleString()} remaining)
            </button>
          )}
        </div>
      </div>

      <footer className="danger-zone">
        <button className="btn danger" onClick={removeProject}>
          Delete project
        </button>
      </footer>
    </div>
  );
}
