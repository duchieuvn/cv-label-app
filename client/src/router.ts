import { useEffect, useState } from 'react';

export type Route =
  | { page: 'projects' }
  | { page: 'project'; projectId: string }
  | { page: 'settings'; projectId: string }
  | { page: 'annotate'; projectId: string; imageId: string; filter: FolderFilter | null };

/** Limits the annotator's image list to one folder. */
export type FolderFilter = { folder: string; recursive: boolean };

export const filterQuery = (filter: FolderFilter | null) =>
  filter ? `?${new URLSearchParams({ folder: filter.folder, recursive: filter.recursive ? '1' : '0' })}` : '';

export function parseRoute(hash: string): Route {
  const [path, search] = hash.replace(/^#\/?/, '').split('?');
  const parts = path.split('/').filter(Boolean);
  if (parts[0] === 'projects' && parts[1]) {
    if (parts[2] === 'settings') return { page: 'settings', projectId: parts[1] };
    if (parts[2] === 'annotate' && parts[3]) {
      const query = new URLSearchParams(search);
      const folder = query.get('folder');
      const filter = folder ? { folder, recursive: query.get('recursive') !== '0' } : null;
      return { page: 'annotate', projectId: parts[1], imageId: parts[3], filter };
    }
    return { page: 'project', projectId: parts[1] };
  }
  return { page: 'projects' };
}

const listeners = new Set<() => void>();

/**
 * Updates the route immediately. The browser delivers hashchange later, and a key pressed in that gap
 * would act on the previous image: quickly pressing A then D could land two images away.
 */
export const navigate = (path: string) => {
  window.location.hash = path;
  for (const listener of listeners) listener();
};

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => {
      const next = parseRoute(window.location.hash);
      setRoute((current) => (JSON.stringify(current) === JSON.stringify(next) ? current : next));
    };
    listeners.add(onChange);
    window.addEventListener('hashchange', onChange);
    return () => {
      listeners.delete(onChange);
      window.removeEventListener('hashchange', onChange);
    };
  }, []);
  return route;
}
