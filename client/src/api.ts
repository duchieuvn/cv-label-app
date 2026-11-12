import type { AutoBoxInfo, AutoBoxStatus, BulkResult, FolderNode, ImageDetail, ImageMeta, ImportResult, Label, Project, ProjectMode, ScanStatus, Source, UploadResult } from '../../shared/schema.ts';
import { SaveQueue, type SavePayload } from './annotate/saveQueue.ts';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error ?? `Request failed with status ${response.status}`);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

const json = (method: string, body: unknown, keepalive = false): RequestInit => ({
  method,
  keepalive,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export type ImageQuery = { folder?: string; recursive?: boolean; label?: string };
export type BulkRequest = {
  selector: { folder: string; recursive: boolean; except: string[] } | { imageIds: string[] };
  addLabels?: string[];
  removeLabels?: string[];
  status?: 'todo' | 'done';
};

export const api = {
  listProjects: () => request<Project[]>('/projects'),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  createProject: (name: string, mode: ProjectMode) => request<Project>('/projects', json('POST', { name, mode })),
  updateProject: (id: string, changes: { name?: string; labels?: Label[] }) =>
    request<{ project: Project; prunedKeypoints: number }>(`/projects/${id}`, json('PATCH', changes)),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: 'DELETE' }),
  listImages: (projectId: string, filter: ImageQuery = {}) => {
    const query = new URLSearchParams();
    if (filter.folder) query.set('folder', filter.folder);
    if (filter.recursive === false) query.set('recursive', '0');
    if (filter.label) query.set('label', filter.label);
    return request<ImageMeta[]>(`/projects/${projectId}/images?${query}`);
  },
  listFolders: (projectId: string) => request<FolderNode[]>(`/projects/${projectId}/folders`),
  listSources: (projectId: string) => request<Source[]>(`/projects/${projectId}/sources`),
  addSource: (projectId: string, path: string) => request<Source>(`/projects/${projectId}/sources`, json('POST', { path })),
  rescan: (sourceId: string, includeExcluded: boolean) => request<ScanStatus>(`/sources/${sourceId}/scan`, json('POST', { includeExcluded })),
  cancelScan: (sourceId: string) => request<void>(`/sources/${sourceId}/scan`, { method: 'DELETE' }),
  removeSource: (sourceId: string) => request<void>(`/sources/${sourceId}`, { method: 'DELETE' }),
  bulkUpdate: (projectId: string, body: BulkRequest) => request<BulkResult>(`/projects/${projectId}/images/bulk`, json('POST', body)),
  getAutoBox: (projectId: string) => request<AutoBoxInfo>(`/projects/${projectId}/autobox`),
  startAutoBox: (projectId: string, options: { mode: 'train' | 'predict'; confidence: number; checkpoint: string; patience?: number }) =>
    request<AutoBoxStatus>(`/projects/${projectId}/autobox`, json('POST', options)),
  cancelAutoBox: (projectId: string) => request<void>(`/projects/${projectId}/autobox`, { method: 'DELETE' }),
  deleteAutoBoxCheckpoint: (projectId: string, checkpoint: string) => request<void>(`/projects/${projectId}/autobox/checkpoints/${encodeURIComponent(checkpoint)}`, { method: 'DELETE' }),
  clearAutoBoxes: (projectId: string) => request<{ cleared: number }>(`/projects/${projectId}/autobox/predictions`, { method: 'DELETE' }),
  uploadImages: (projectId: string, files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append('files', file);
    return request<UploadResult>(`/projects/${projectId}/images`, { method: 'POST', body: form });
  },
  importAnnotations: (projectId: string, file: File, options: { createLabels: boolean; overwrite: boolean; markDone: boolean; dryRun: boolean }) => {
    const form = new FormData();
    form.append('file', file);
    for (const [key, value] of Object.entries(options)) form.append(key, value ? '1' : '0');
    return request<ImportResult>(`/projects/${projectId}/import`, { method: 'POST', body: form });
  },
  getImage: (id: string) => request<ImageDetail>(`/images/${id}`),
  deleteImage: (id: string) => request<void>(`/images/${id}`, { method: 'DELETE' }),
  saveImage: (id: string, payload: SavePayload, keepalive = false) =>
    request<ImageDetail>(`/images/${id}/annotations`, json('PUT', payload, keepalive)),
};

export const imageUrl = (id: string) => `/api/images/${id}/file`;

/** Shared across pages so a pending save survives leaving the annotator. */
export const saveQueue = new SaveQueue(api.saveImage);
