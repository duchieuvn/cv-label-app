# LabelApp

Self-hosted image labeling for bounding boxes and keypoints. `specs.md` is the source of truth for
behavior: read it before changing features, and update it in the same change when behavior changes.

## Commands

```sh
npm run dev         # server on :3001 (with --watch) and Vite on :5173; open http://localhost:5173
npm run build       # build the client into dist/
npm start           # serve the API and dist/ on http://127.0.0.1:3001
npm run typecheck   # tsc --noEmit
npm test            # unit and API tests (vitest)
npm run test:e2e    # browser tests (Playwright with the system Chrome; builds first)
```

Run a single test with `npx vitest run tests/unit/export.test.ts` or
`npx playwright test -g "box-then-keypoints"`.

Environment: `PORT` (3001), `HOST` (127.0.0.1), `DATA_DIR` (./data), `IMPORT_ROOTS` (folders allowed for
folder import, separated by `:`; required on non-loopback addresses), `LABELAPP_PYTHON` (`python3`; runs auto
boxing and needs the `ultralytics` package).

**Python:** always use the `ml` conda environment (`conda activate ml`). pyenv can come before it on `PATH`, so
activating is not enough by itself: check that `which python3` points into `envs/ml`, and start the server with
`LABELAPP_PYTHON="$CONDA_PREFIX/bin/python" npm run dev` so that auto boxing uses it.

## Architecture

- **Node 24 runs the server's TypeScript directly** with no build step. So server and shared code must
  use erasable syntax only (no `enum`, no parameter properties) and import with explicit `.ts` extensions.
- `shared/schema.ts`: zod schemas, types, and the annotation invariants. The client uses `import type`
  from here, which keeps zod out of the browser bundle.
- `server/app.ts`: Express routes. `server/store.ts`: SQLite through `node:sqlite`, with migrations in
  the `MIGRATIONS` array (append only; `user_version` tracks them). `server/export.ts`: pure exporters. `server/import.ts`: pure COCO and native JSON import (parse, match images by path, plan writes).
  `server/scan.ts`: background folder scans, with status held in memory. `server/imageInfo.ts`: identifies
  an image from its contents, for both uploads and scans. `server/autobox.ts`: the auto box job, with status held
  in memory, which runs `server/autobox_worker.py` and reads its progress as JSON lines on file descriptor 3.
- `client/src/annotate/`: the annotator.
  - `geometry.ts` and `doc.ts` are pure and unit tested. Put new annotation logic here, not in components.
  - `Canvas.tsx` owns zoom, pan, and pointer interaction. A drag is previewed locally through `applyDrag`
    and reported only on release, which is why one drag is one undo step.
  - `AnnotatePage.tsx` owns the document, tools, selection, and keyboard shortcuts.
  - `saveQueue.ts` serializes autosaves. All writes of annotations go through it.
- `client/src/pages/ProjectPage.tsx` owns folders, the grid, selection, group actions, and export.
- `client/src/modes.ts`: what each project mode (`bbox`, `keypoint`) shows. See `specs.md` section 11. Add new
  mode-specific UI through it rather than testing `project.mode` in components.
- Routing is a small hash router in `client/src/router.ts`. Navigate with `navigate()`, which updates the
  route immediately. Waiting for `hashchange` lets a fast key press act on the previous image.

## Conventions that are easy to break

- Coordinates are **image pixels in the EXIF-corrected frame**, everywhere: storage, API, and client
  state. Only the exporters normalize, and only `Canvas.tsx` deals in screen coordinates.
- Placed keypoints are keyed by keypoint **id**, never by index. Array order in the schema only matters
  for export order.
- Stored annotations always satisfy the invariants in `findAnnotationError`, so the exporters and
  renderer don't handle orphans. Any change that could break them must also migrate data, as
  `updateLabels` does when pruning.
- An untouched `todo` image must never be exported: an empty YOLO label file means "true negative".
- Uploads are identified by their contents, never by name or declared MIME type. Don't add a format
  that can carry scripts (such as SVG), because files are served from the app's origin.
- **Nothing may modify, move, or delete a file in a source folder.** Only uploads under `data/images/` are
  ever deleted. Use `imagePath` to locate a file, and check `sourceId` before any filesystem write.
- Serve scanned files with the type detected from their contents, since a scanned file can have any name.
  Never accept a file path from a request: files are reached by image id only.
- Auto boxes are guesses. An image with `autoBoxed` must never be trained on or exported as if a person drew its
  boxes, and a job must never overwrite an image a person has edited. See `specs.md` section 10.
- Image labels never affect which images COCO and YOLO export; they have their own `image_labels` format.
- An image with `fileState` other than `ok` has read-only annotations, enforced by the server, because its
  stored dimensions may not match the file. Labels and status can still change.
- A project's mode is fixed at creation. Don't let keypoints into a `bbox` project: `findLabelModeError` is what
  makes every annotation there a box. A `keypoint` project can hold old boxes, which must be kept, not stripped.
- The image list can hold 100,000 rows. Avoid rendering or re-rendering per-image work in the annotator, and
  re-run the numbers in `specs.md` section 9 after changing how images are listed or stored.
- The app has no auth and binds to loopback by default; the `Origin` check in `app.ts` protects it from
  other websites. Don't add permissive CORS.

## Testing

Pure logic gets unit tests in `tests/unit`, HTTP behavior goes in `tests/unit/api.test.ts` against a real
server and temp directory, and interaction goes in `tests/e2e`. Pointer behavior can only be trusted from
the e2e tests, so run them after touching `Canvas.tsx` or `AnnotatePage.tsx`.

Automated tests replace the auto box worker with `tests/fake_autobox_worker.ts`, so they need no Python and
do not exercise `server/autobox_worker.py`. After changing the worker, run a real job on a small project.
