import type { Annotation, ImageStatus } from '../../../shared/schema.ts';

export type SavePayload = { annotations?: Annotation[]; status?: ImageStatus; imageLabels?: string[] };
export type SaveState = { kind: 'saved' | 'pending' | 'error'; message?: string };
type SaveFn = (imageId: string, payload: SavePayload, keepalive: boolean) => Promise<unknown>;

const DEBOUNCE_MS = 400;
const RETRY_MS = 3000;

/**
 * Serializes autosaves so an older write can never land after a newer one. Payloads for one image
 * merge while waiting, and a failed payload stays queued (under any newer edits) until it succeeds.
 */
export class SaveQueue {
  private pending = new Map<string, SavePayload>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running: Promise<void> | null = null;
  private listeners = new Set<(state: SaveState) => void>();
  private save: SaveFn;
  state: SaveState = { kind: 'saved' };

  constructor(save: SaveFn) {
    this.save = save;
  }

  subscribe(listener: (state: SaveState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private setState(state: SaveState) {
    this.state = state;
    for (const listener of this.listeners) listener(state);
  }

  private arm(delay: number) {
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => void this.flush(), delay);
  }

  schedule(imageId: string, payload: SavePayload) {
    this.pending.set(imageId, { ...this.pending.get(imageId), ...payload });
    if (this.state.kind !== 'error') this.setState({ kind: 'pending' });
    this.arm(DEBOUNCE_MS);
  }

  /** The not-yet-saved changes for an image, which take priority over what the server returns. */
  pendingFor(imageId: string): SavePayload | undefined {
    return this.pending.get(imageId);
  }

  get hasUnsaved(): boolean {
    return this.pending.size > 0 || this.running !== null;
  }

  /** Resolves once everything queued has been attempted; failures remain queued and retry. */
  async flush(): Promise<void> {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    while (this.running) await this.running;
    if (this.pending.size === 0) return;
    this.running = this.drain().finally(() => {
      this.running = null;
    });
    await this.running;
  }

  private async drain() {
    for (const [imageId, payload] of [...this.pending]) {
      this.pending.delete(imageId);
      try {
        await this.save(imageId, payload, false);
      } catch (error) {
        this.pending.set(imageId, { ...payload, ...this.pending.get(imageId) });
        this.setState({ kind: 'error', message: error instanceof Error ? error.message : String(error) });
        this.arm(RETRY_MS);
        return;
      }
    }
    if (this.pending.size > 0) this.arm(DEBOUNCE_MS);
    else this.setState({ kind: 'saved' });
  }

  /** Last-chance send as the page unloads, when normal requests would be cancelled. */
  flushOnUnload() {
    for (const [imageId, payload] of this.pending) {
      void this.save(imageId, payload, true).catch(() => {});
    }
  }
}
