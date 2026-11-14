# LabelApp — Specification

A self-hosted image labeling tool in the spirit of Label Studio, narrowed to the two
annotation types that matter for detection and pose datasets: **bounding boxes** and
**keypoints**, each in its own project mode (section 11). It trades Label Studio's generality (XML label configs, many data types,
multi-user workflows) for fast keyboard-driven annotation and first-class COCO/YOLO export.

## 1. Scope

**In scope (v1)**

- Projects, each with one mode and its own label schema
- Uploading images (browser, multi-file, drag and drop)
- Annotating bounding boxes, or named keypoints with skeletons
- Autosave, undo/redo, zoom/pan, keyboard shortcuts
- Export to native JSON, COCO, and YOLO (detection and pose), and a separate image label export
- Importing a folder by detecting every image in it and its subfolders (section 7)
- Selecting all images under a folder and adding a label to them as a group (section 8)
- Training a detector or pose model on a project's annotations and drawing on its remaining images, locally or on a
  remote GPU machine (section 10)
- Importing annotations from a COCO file or a LabelApp export into a project's existing images (section 12)
- Sharing a project as a file that a teammate opens on another computer, with the images in their own folders
  (section 13). It is a copy, not shared editing

**Out of scope (v1)**

- Multiple users, authentication, roles, review workflows. The app is a single-user local
  tool and binds to `127.0.0.1` by default. Concurrent edits of one image are last-write-wins.
  A shared project (section 13) is an independent copy: nothing syncs or merges back.
- Other annotation types (polygons, masks, video, text)
- Importing YOLO label files, and importing images along with annotations. Import only annotates images already in the project
- Generated thumbnails (the browser lazy-loads originals)

## 2. Concepts

| Term | Meaning |
| --- | --- |
| **Project** | A named set of images plus a label schema, in one mode |
| **Mode** | What a project annotates: `bbox` or `keypoint`. Fixed when the project is created (section 11) |
| **Label** | An object class: name, color, optional keypoint definitions and skeleton |
| **Keypoint definition** | A named slot on a label, e.g. `left_eye`. Has a stable id |
| **Skeleton** | Edges between keypoint definitions of one label, used for drawing and exported to COCO |
| **Annotation** (instance) | One object in one image: a label, an optional bbox, and placed keypoints |
| **Status** | Per image: `todo` or `done`. A `done` image with no annotations is a true negative |
| **Source folder** | A folder on the server's disk registered with a project and scanned for images |
| **Image label** | A project label applied to a whole image rather than to an object in it. It has no geometry |

A single annotation holds both the box and the keypoints of one object. This matches COCO's
object model and is what makes pose export possible without a separate linking step.

An annotation must have a bbox, at least one placed keypoint, or both. A bbox project only ever
holds boxes, and a keypoint project places keypoints without a box (section 11).

## 3. Data model

```ts
type Label = {
  id: string;
  name: string;              // unique within the project
  color: string;             // #rrggbb
  keypoints: { id: string; name: string }[];   // order defines export order
  skeleton: [string, string][];                // pairs of keypoint ids
};

type Annotation = {
  id: string;
  labelId: string;
  bbox: { x: number; y: number; w: number; h: number } | null;
  keypoints: Record<string, { x: number; y: number; v: 1 | 2 }>;  // by keypoint id
};
```

- **Coordinates** are floating point image pixels with the origin at the top-left corner, in
  the *display-oriented* frame. EXIF orientation is applied, so a phone photo stored rotated has
  its width and height swapped accordingly. Training code must apply EXIF orientation too
  (for example with `ImageOps.exif_transpose` in PIL).
- **Visibility** `v` follows COCO: `2` = visible, `1` = placed but occluded. An unplaced keypoint
  is absent from the record and exports as `v = 0`.
- **Keypoints are keyed by id, not position**, so reordering definitions in the schema never
  corrupts existing annotations.

### Invariants enforced by the server

1. Every annotation references a label in the project and only that label's keypoint ids.
2. Every coordinate lies inside the image, with a tolerance of 0.5px.
3. Every annotation has a bbox or at least one keypoint.
4. Label and keypoint names are unique in their scope; skeleton edges reference existing
   keypoints and are not self-loops.
5. Every id in an image's `imageLabels` references a label in the project, with no
   duplicates.
6. A label in a `bbox` project defines no keypoints. With 1 and 3, this means an annotation there
   always has a box and nothing else.

### Schema edits

- **Deleting a label** that any annotation uses, or that any image has as an image
  label, is rejected with `409`.
- **Deleting a keypoint definition** removes its placed points from all annotations in the same
  transaction, and removes annotations left with neither a bbox nor keypoints. The UI asks for
  confirmation first. This keeps invariant 1 true without any orphan handling elsewhere.

### Storage

```
data/
  labelapp.db                  SQLite: projects, images
  images/<projectId>/<uuid>.<ext>
```

`projects.labels` and `images.annotations` are JSON columns. An image's annotations are always
read and written as a whole, so per-annotation rows would add joins without benefit.

Images found by a folder scan stay where they are and are not copied into `data/`. See
section 7.

## 4. HTTP API

JSON unless noted. Errors are `{ "error": string }`. Requests with a cross-site `Origin` are
rejected with `403`.

| Method and path | Purpose |
| --- | --- |
| `GET /api/projects` | List with `imageCount`, `doneCount` |
| `POST /api/projects` | Create from `{ name, mode?, labels? }`. `mode` defaults to `bbox`. Returns `422` if the labels do not suit the mode |
| `GET /api/projects/:id` | Fetch one |
| `PATCH /api/projects/:id` | Update `{ name?, labels? }`; see schema edits. The mode cannot change. Returns the project and `prunedKeypoints` |
| `DELETE /api/projects/:id` | Delete with images and files |
| `GET /api/projects/:id/images?status=` | Metadata list with `annotationCount` |
| `POST /api/projects/:id/images` | Multipart field `files`. Returns `{ uploaded, rejected }` |
| `GET /api/images/:id` | Metadata with annotations |
| `PUT /api/images/:id/annotations` | Replace with `{ annotations?, status? }` |
| `DELETE /api/images/:id` | Delete one image |
| `GET /api/images/:id/file` | The image bytes |
| `GET /api/projects/:id/export?format=&scope=&includeImages=` | Download an export. `format=project` downloads a project file (section 13) and ignores the other parameters |
| `POST /api/projects/open` | Multipart field `file` (a project file) plus `options` (JSON `{ name?, paths }`) and a `dryRun` flag; see section 13. A dry run returns a preview; otherwise `201` with `{ project, warnings }` |
| `POST /api/projects/:id/import` | Multipart field `file` plus `createLabels`, `overwrite`, `markDone`, `dryRun` flags; see section 12. Returns `{ summary, dryRun, project }` |
| `GET /api/projects/:id/sources` | List source folders |
| `POST /api/projects/:id/sources` | Register `{ path }` and start a scan. Returns `202` with the source and its scan |
| `POST /api/sources/:id/scan` | Start a rescan. `{ includeExcluded? }`. Returns `202`, or `409` if one is running |
| `GET /api/sources/:id/scan` | Scan progress and result |
| `DELETE /api/sources/:id/scan` | Cancel the running scan |
| `PATCH /api/sources/:id` | Point the source at another folder, `{ path }`, and rescan it. Returns `202`, or `409` if the folder overlaps another source or a scan is running |
| `DELETE /api/sources/:id` | Remove the source and its images from the project. Files on disk are untouched |
| `GET /api/projects/:id/folders` | Folder tree with image counts |
| `POST /api/projects/:id/images/bulk` | Change a group of images; see section 8 |
| `GET /api/projects/:id/autobox` | Auto box job status, image counts, the project's checkpoints and other projects' (`otherCheckpoints`); see section 10 |
| `POST /api/projects/:id/autobox` | Start a job from `{ mode?, confidence?, checkpoint?, patience? }`, where `mode` is `train` or `predict`. Returns `202`, `409` if one is running, or `422` if the project is not ready |
| `DELETE /api/projects/:id/autobox` | Cancel the running job |
| `DELETE /api/projects/:id/autobox/checkpoints/:checkpoint` | Delete one saved checkpoint. Drawn boxes are kept. Returns `404` if it does not exist, or `409` during a job |
| `DELETE /api/projects/:id/autobox/predictions` | Remove all unreviewed predictions. Returns `{ cleared }` |
| `GET /api/projects/:id/remote` | Status of the one-click remote run, or `null` |
| `POST /api/projects/:id/remote` | Start a one-click remote training from `{ target, password?, model?, epochs?, patience?, imgsz?, confidence?, checkpoint?, keepOriginals? }`; see section 10. Returns `202`, `409` if one is going, or `422` if the project is not ready |
| `DELETE /api/projects/:id/remote` | Cancel the local side of the remote run |
| `GET /api/projects/:id/remote/bundle?model=&epochs=&patience=&imgsz=&confidence=&checkpoint=` | Download a remote training bundle; see section 10. Returns `422` if the project is not ready |
| `POST /api/projects/:id/remote/results` | Multipart `results` (JSON) and optional `weights` (`.pt`) from a remote run. Returns the counts, or `422` if the results are not from this project's bundle |

**Uploads** accept JPEG, PNG, WebP, GIF and BMP up to 100 MB each. The type comes from the file's
contents, never its name or declared MIME type, so SVG and other active content cannot be stored
and later served from the app's origin. Rejected files are reported per file and do not fail the
batch. Stored names are generated; the original name is kept only as metadata.

## 5. Export formats

`scope` selects images: `done` (default) or `annotated`, which is every `done` image plus any
`todo` image that has annotations. Untouched `todo` images are never exported, because an empty
YOLO label file would claim the image is a true negative. A `todo` image holding only unreviewed
predictions (section 10) counts as untouched.

Image labels never affect which images `coco` and `yolo` select. Otherwise a `todo` image
with an image label and no boxes would be exported with an empty YOLO label file, wrongly claiming it
contains no objects. Image labels have their own export, `image_labels`, described below.

- **`project`**: a project file for a teammate's computer (section 13). It is not a training format, so it holds
  every image, untouched `todo` images included, and the rule above does not apply to it.
- **`json`**: native format, lossless: project with its mode, labels, images, annotations, and each
  image's `folder` and `imageLabels`.
- **`coco`**: one JSON file. `categories[].keypoints` and `skeleton` (1-indexed) are included for
  labels that define keypoints. `keypoints` is a flat `[x, y, v, ...]` in schema order.
  `file_name` is the image's relative path, so identical names in different folders stay
  distinct. Uploaded images have no folder structure, so theirs is the plain file name.
- **`yolo`**: a zip with `data.yaml`, `labels/*.txt`, and optionally `images/*`.
  - No label defines keypoints: `cls cx cy w h`, normalized.
  - Any label defines keypoints: pose format `cls cx cy w h x1 y1 v1 ...` with
    `kpt_shape: [K, 3]`, K being the largest keypoint count. YOLO requires one K per dataset, so
    labels with fewer keypoints are padded with `0 0 0` and `data.yaml` records the order per class.
  - Label files are matched to images by stem, so colliding stems (`a.jpg`, `a.png`) get a suffix.
    Folders are flattened, so `cats/1.jpg` and `dogs/1.jpg` collide and are handled the
    same way.

COCO and YOLO both need a box on every object. A keypoint-only annotation, which is what a keypoint
project creates, exports with the tight box around its keypoints, at least 1px on each side.

With `includeImages=1`, an image whose file is unavailable (section 7) is left out of the
zip along with its label file, and listed in `export_warnings.txt`. It does not fail the export.

### Image label export

`format=image_labels` is a separate export for classification and tagging. It is a zip of only image
labels, and contains no boxes or keypoints. It appears as **Image labels (zip)** in the export format list.

- **Which images:** every image with at least one image label, whatever its status and annotations.
  `scope` here takes `labeled` (the default) or `done`, which keeps only `done` images. An image with no
  image label is never included, so its absence says nothing about its class.
- **`labels.csv`:** one `image,label` row per image label, so an image with two labels has two rows. Values
  are quoted as CSV requires, so commas and non-ASCII names are safe.
- **`classes.txt`:** every project label name, one per line, in schema order. Labels no image uses are
  included so class indices stay stable between exports.
- **`images/`:** included with `includeImages=1`. Folder structure is kept, so `image` in `labels.csv` is the
  file's path relative to `images/`, such as `2024/cam1/a.jpg`. That column uses the same path when images
  are left out. Two images with the same path, which only uploads can produce, get a suffix.
- **Unavailable files:** with `includeImages=1`, these are handled as above, including removing their rows
  from `labels.csv`. Without images, their rows stay, because labels do not depend on the file.

## 6. Annotation UI

### Layout

Top bar (navigation, tools, undo/redo, zoom, save state, Done), image list on the left, canvas in
the center, and on the right the labels, the instances in this image, and the selected instance's
details. The right side also shows the image's image labels, which can be added and removed
there, and the image list follows any folder filter the annotator was opened with (section 8).

### Tools

| Tool | Behavior |
| --- | --- |
| **Select** (both modes) | Click to select. Drag a box to move it. Drag a handle to resize. Drag empty space to pan |
| **Box** (`bbox`) | Drag to draw with the active label. Drags under 2px are discarded |
| **Keypoint** (`keypoint`) | Click to place the next unplaced keypoint of the selected instance. If there is none, a new instance of the active label starts. Shift+click places it as occluded |

In every tool, dragging a keypoint moves it and dragging a handle of the selected box resizes it. A project
starts on its mode's own tool. Placing an instance's last keypoint, or pressing Esc, finishes it, so the next
click starts a new instance.

Geometry is always clamped to the image. Moving an instance keeps its shape, stopping when any part
would leave the image.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `V` `B` | Select, Box tool (`bbox`) |
| `V` `K` | Select, Keypoint tool (`keypoint`) |
| `1`–`9` | Choose active label |
| `A` / `D` | Previous / next image |
| `Ctrl+Enter` | Mark done and go to the next `todo` image |
| `Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y` | Undo, redo |
| `Delete` / `Backspace` | Delete selected keypoint, otherwise selected instance |
| Arrow keys | Nudge selection 1px; with `Shift`, 10px |
| `N` | Skip to the next unplaced keypoint (`keypoint`) |
| `O` | Toggle selected keypoint visible / occluded (`keypoint`) |
| `Esc` | Cancel the current drag, otherwise deselect |
| `H` | Hide or show annotations |
| `F`, `+`, `-` | Fit to window, zoom in, zoom out |
| `Space` + drag, or middle mouse | Pan |
| Mouse wheel | Zoom at the cursor |
| `?` | Show shortcut help |

### Saving

- Changes save automatically 400 ms after the last edit, and immediately on navigation.
- Saves are serialized, and a failed save is retried every 3 s and shown in the top bar. An edit is
  never silently dropped: leaving the page with unsaved changes prompts when the last save failed.
- A drag creates one undo step, on release. History is per image and resets on navigation, with a
  limit of 100 steps.

## 7. Folder import

Point a project at a folder and the app finds every image inside it, at any depth. This suits datasets
too large to upload through a browser.

### Choosing a folder

The user types or pastes an absolute path on the server's disk, such as `/data/street/2024`. A browser
cannot supply this path or read a folder without copying it, so the path is entered as text. The server
returns a specific error if it does not exist, is not a directory, or cannot be read.

### What a scan finds

- It walks the folder and all subfolders. Hidden files and folders (names starting with `.`) are
  skipped. Symbolic links are not followed, which prevents loops and escaping the chosen folder.
- A file counts as an image based on its contents, exactly as for uploads: JPEG, PNG, WebP, GIF or BMP.
  File extensions are not trusted, and the 100 MB limit does not apply because nothing is transferred.
- Dimensions are read with EXIF orientation applied, as in section 3.
- Other files are counted as `skipped` without being listed, because datasets often contain many
  unrelated files. Image-like files that cannot be read or decoded are listed as `errors`, with reasons.
- A folder that overlaps one already registered in the project (the same folder, a parent, or a child)
  is rejected with `409`, so no file can appear twice. The user rescans or removes the existing one.

### Files stay in place

Found images are referenced by path, not copied, so importing uses little disk space. **The app never
modifies, moves or deletes anything in a source folder.** So:

- Deleting an image, a source or a project removes only the app's records and annotations. Confirmation
  text must say that the files stay on disk.
- Moving or renaming the folder makes its images unavailable until it is available again at that path.
- Uploaded images keep working as before, stored in `data/images/`.

### Scanning runs in the background

A scan can take minutes, so it runs as a background job and the page shows live counts of folders
visited, images found, files skipped and errors. Images appear as they are found. A scan can be
cancelled, keeping what it has already added, and only one runs per source at a time. A server restart
abandons a running scan; the user starts it again and nothing is duplicated.

### Rescanning

Rescanning is manual; the app does not watch folders. Images are matched by path, so a rescan is safe to
repeat:

| Found on disk | Result |
| --- | --- |
| New file | Added as `todo` |
| Known file, unchanged | Nothing changes |
| Known file, now absent | Marked `missing`. Annotations are kept, and it recovers if the file returns |
| Known file, new dimensions | Marked `changed`. Stored dimensions and annotations stay as they are, since the coordinates may no longer fit |
| Image the user deleted from the project | Not added back, unless the rescan sets `includeExcluded`, which restores every image deleted from that source. Deleted images therefore leave a record of their path |

An image is **unavailable** if it is `missing` or `changed`. Unavailable images are flagged in the grid
and the rescan result, and they open in the annotator with a banner and read-only annotations. A
`changed` image returns to normal if its file regains the stored dimensions; otherwise the user deletes
it from the project and re-adds it through a rescan with `includeExcluded`, which starts it fresh.

A source can be pointed at another path (`PATCH /api/sources/:id`) when its folder moves, or when a project opened from
a project file was given the wrong one. The path passes the same checks as adding a folder. Every image of the source
then has its stored size and time forgotten, and the rescan that follows checks each file's dimensions, so images found
at the new path recover and the rest become `missing`. Annotations are never touched.

### Order and naming

Scanned images sort by relative path in natural order, so `img2.jpg` comes before `img10.jpg`. An image's
`folder` is its directory relative to the source, prefixed with the source folder's name, for example
`2024/cam1`. Uploaded images have the folder `uploads`.

Folders become the tree in section 8, so source names must be unique within a project. A second source
with the same folder name gets a suffix (`images_2`), and `uploads` is reserved.

### Security

This feature lets a request make the server read from any path it can access, so:

- The existing `Origin` check is required on these routes.
- Image bytes are served only by image id, never by a path from the request, and only for files that
  passed the content check.
- Registration is allowed by default only when the server is bound to a loopback address. On any other
  address it is disabled (`403`) unless `IMPORT_ROOTS` lists allowed parent folders, and a path must
  resolve to somewhere inside one of them, with symbolic links resolved first.

## 8. Folder selection and group labeling

Select every image under a folder in one action, then add a label to all of them at once.

### Image labels

"Add a label" here means an **image label**: the image as a whole gets a project label, such as tagging
everything under `night/` as `night`. It uses the same label list as annotations and draws nothing.

- `imageLabels: string[]` (label ids) is added to each image, and an image can have several.
- A label with keypoints can be used, but the keypoints are irrelevant at the image level.
- Image labels are separate from annotations: adding one creates no box, and does not change the status.
- They are exported by `image_labels` and the native `json` format only (section 5).

### Selecting by folder

The project page gains a collapsible folder tree beside the grid. Each folder shows its total image count,
including subfolders.

- **Clicking a folder** filters the grid to it, including subfolders by default. A *Include subfolders*
  toggle limits it to the folder's direct images.
- **Select all in folder** selects every image in the current filter, including those not yet shown
  because of the grid's *Show more* paging. The bar shows the count, for example "1,240 images selected".
- Individual images can be added or removed using a checkbox on each thumbnail. Removing one from a
  folder selection keeps the rest selected.
- Changing the filter clears the selection, so an action cannot reach images no longer on screen.

### Group actions

With a selection active, an action bar offers:

| Action | Behavior |
| --- | --- |
| **Add label** | Choose an existing label, or create one by name and color in place. A new label's color starts as one no other label in the project has, and the user can change it. Images already having it are unaffected |
| **Remove label** | The bar lists each label on the selected images with its image count. Clicking one removes it from the images that have it |
| **Set status** | Mark `done` or `todo`. Marking `done` warns that images with no annotations will count as containing no objects (section 5) |
| **Annotate selection** | Open the annotator with its image list limited to the selected folder. Available for folder selections only |

**Add label** and **Remove label** show the number of images affected before running. **Set status** asks
for confirmation, because it can overwrite work across many images.

### Request

```ts
// POST /api/projects/:id/images/bulk
{
  selector:
    | { folder: string; recursive: boolean; except?: string[] }  // image ids to leave out
    | { imageIds: string[] },
  addLabels?: string[];     // label ids
  removeLabels?: string[];  // label ids
  status?: 'todo' | 'done';
}
// Response
{ matched: number; changed: string[] }  // changed: ids of images actually altered
```

- A folder selector is resolved by the server when the request runs, so large selections do not send
  thousands of ids. Folder matching works on whole path segments, so `cats` does not match `cats2`.
- The request runs in one transaction: all selected images change, or none do.
- Naming the same label in both `addLabels` and `removeLabels` is a `400`.
- It applies only to images that exist at that moment. It is not a rule, so images found by a later
  rescan do not inherit the label.
- Unavailable images (section 7) can still receive labels and status, because neither depends on the
  file.

### Undo

After a group action, a notice offers **Undo** until the next group action or until leaving the page. It
reverts only the images in `changed`, so an image that already had the label before the action keeps it.
This is separate from the annotator's per-image undo history.

### Filtering by label

`GET /api/projects/:id/images` gains `folder`, `recursive` and `label` parameters. The grid can filter by
image label, and its counts combine with the folder filter.

### In the annotator

Changes to one image's labels go through the save queue (`imageLabels` is added to the
`PUT /api/images/:id/annotations` body). They are not part of the undo history, which covers only
annotations. Number keys continue to choose the active label for drawing.

## 9. Performance

Measured at 100,000 images in 111 folders, using small PNG files on a local disk. Real photos take longer
to scan, because more of each file is read.

| Operation | Time |
| --- | --- |
| First scan | 62 s (about 1,600 images/s) |
| Rescan, nothing changed | 3 s |
| Add a label to a 10,000-image folder | 0.6 s |
| Add a label to all 100,000 images | 2.4 s |
| List all images | 0.9 s, 30 MB response |
| List one 1,000-image folder | 0.07 s |
| Open the project page | 1.5 s |
| Open the annotator on the whole project | 2.9 s |
| Draw a box, from input to render | 0.08 s |
| Export a project file, half the images with 2 boxes each | 1.5 s, and 1 MB for this repetitive data. Real annotations compress less |
| Open that file, before its scan | 2.4 s (0.7 s to read and check it, 1.7 s to create the project) |

Known limits of the design:

- The canvas uses SVG, which gives precise hit testing and simple code. Thousands of instances per
  image would need a Canvas 2D renderer.
- The project grid loads original files instead of thumbnails. Only images scrolled into view load, but
  grids of large photos have not been measured and will be slower than the figures above.
- The client loads the full image list, so opening a page over 100,000 images costs a 30 MB response. The
  annotator's list renders only the rows on screen, and refreshes during a scan back off as the list grows.
- SQLite blocks the server while a large query runs, so other requests wait during it.
- The scan that follows opening a project file inspects every file, since no stored size or time can match, so it
  takes as long as a first scan.

## 10. Adaptive auto bounding box

Train a detector on the boxes already drawn in a project, then let it draw boxes on the remaining images.
This is only available in `bbox` projects.
Reviewing some of its work and running it again trains a better model, which is what makes it adaptive.
Every trained model is kept as a checkpoint, so any of them can draw again later or be trained further.

### What a run does

1. **Trains** a YOLOv8 nano model (`yolov8n.pt`, or `yolov8n-pose.pt` in a keypoint project) from the checkpoint
   chosen in **Train from**, until validation accuracy stops improving for the chosen patience, with a limit of 300 epochs. The image size is 640. The model with
   the best validation score is used, and saved as a new checkpoint.
2. **Predicts** on the target images one at a time, and saves each image's boxes as soon as they exist,
   so they appear in the grid during the run.

About a fifth of the training images are held out for validation, chosen by a hash of the image id. The
hash alone decides, so adding images never moves one from training into validation, where a continued model
would be scored on images it has already learned. At least a tenth are held out, and always at least one.

**Train and draw boxes** does both steps, and draws with the checkpoint it just trained. **Draw with trained
model** (`mode: "predict"`) does only the second, using the checkpoint chosen in **Draw with**, so it is much
faster and needs no minimum number of training images.

A slider sets the **confidence** threshold for either action, from 0.05 to 0.95, with a default of 0.65.
Detections scoring below it are not drawn. It does not affect training, so trying another value only takes
**Draw with trained model**. Scores are not stored, so changing the value means predicting again.

| Images | Used for |
| --- | --- |
| Has a saved box, and is not an unreviewed prediction | **Training** |
| `todo` with no annotations, or holding only unreviewed predictions | **Target.** Unreviewed predictions from an earlier run are replaced |
| `done` with no annotations | Neither. It is a true negative chosen by the user, so it is not drawn on |
| Unavailable (section 7) | Neither |

Training needs at least 10 training images, and any run needs one target. In a `bbox` project a predicted
annotation has a box and no keypoints. In a `keypoint` project it has keypoints and no box, as section 11 requires:
each predicted keypoint scoring at least 0.5 is placed as visible, the others stay unplaced, and an instance left
with none is not drawn. Pose training lines pad every label to the largest keypoint count, as the YOLO export does,
and a checkpoint records the keypoint ids behind its slots, so a keypoint deleted since is simply never placed.
Either way the annotation uses the label of the class the model predicted.

### Checkpoints

Each project keeps every model it has trained, as a checkpoint. A checkpoint is added only when training
completes, so a failed or cancelled training adds nothing, and nothing replaces an existing one. Checkpoints
survive a server restart, are about 6 MB each, and stay until deleted. **Delete checkpoint** removes the one
selected in **Draw with**, and boxes already drawn are kept. Deleting the project deletes them all.

| | **Train from** | **Draw with** |
| --- | --- | --- |
| Base weights pretrained on COCO: `yolov8n.pt`, or `yolov8n-pose.pt` in a keypoint project | Yes, always offered | No: it knows COCO's classes, not the project's labels |
| Checkpoint with the project's current set of labels | Yes | Yes |
| Checkpoint from before labels were added or removed | No: shown disabled | Yes |
| Another `bbox` project's checkpoint, listed under **From other projects** | Only if its classes are exactly this project's labels, by name | Yes. Labels it predicts that this project lacks are added first |

A checkpoint records its classes as label ids and names. Within its own project the ids decide, so renaming is
safe; in another project the names decide, ignoring case, so a model trained on `bottle` in one project draws
`bottle` in another. That lets a new project start from a model trained elsewhere before it has drawn a single box.
Its handle in another project is `<projectId>/<checkpoint>`, it is listed in every `bbox` project, and it can
only be deleted from the project that trained it. A request that leaves out `checkpoint` never gets another
project's: drawing without a trained model of its own is refused, and training starts from the base weights.

**Train from** starts on the newest checkpoint that can continue, or the base weights (`yolov8n.pt`, or
`yolov8n-pose.pt` in a keypoint project) if there is none, and
**Draw with** starts on the newest checkpoint. Each follows new checkpoints until the user picks another, and
that pick lasts until the page is reloaded or the checkpoint stops being valid. An API request leaving out
`checkpoint` gets the same defaults, and a request naming an unusable one gets `422`.

**Patience** is the number of epochs without improvement that ends training. From the base weights it is always 20
and cannot be changed, and the server rejects a request that sets it. From a checkpoint the user chooses it,
from 1 to the 300 epoch limit, starting at 20. Training always runs at least one more epoch than the patience,
even when the checkpoint is already at its best.

Training from a checkpoint keeps its class numbers even if the labels were reordered, since the weights mean
those classes. It needs the same set of labels because after adding or removing one the classes no longer line
up. The panel says which checkpoints this affects.

A checkpoint knows only what it was trained on. Class numbers are stored as label ids, so renaming or
reordering labels is safe, but **Draw with trained model** never predicts a label added since training, and
drops predictions for a label deleted since then. Boxes reviewed since training improve the model only after
training again.

### Remote training

A larger model needs a GPU, which this machine may not have. The remote machine's only jobs are training and
prediction; nothing about the app's format lives there, and every conversion happens locally.

**One click.** The panel's **Train on server and draw** does the whole round trip (`POST /api/projects/:id/remote`,
`GET` for status, `DELETE` to cancel): the app makes the bundle, runs `scripts/remote_train.py` itself as a child
process with `--json`, adds the trained model as a checkpoint when it comes back, and then starts the ordinary local
**Draw with trained model** job with it, showing the phase, upload progress, the server's GPU and epochs as it goes.
The server only ever trains. It gets only the labelled images (the training images of the table above), never the
rest of the dataset, so drawing has to happen here, and it does: the same local job, with the same rules, that a
local training ends with. If a local job is running at that moment, or nothing is left to draw on, the status says
so and the checkpoint waits in **Draw with**. The script's own output is
printed on the server's console too, prefixed with the bundle id, as are state changes of local jobs and every API
request that changes something or fails. The request carries the
server as `user@host` and a password, which the app hands to the script on stdin for one login; it is never stored,
logged, or returned in any status. An empty password means the SSH agent or key files. One remote run goes at a time
per project, and not alongside a local job. Cancelling stops the local side only: training already started keeps
running on the server, and `scripts/remote_train.py fetch <bundle-id> user@host` can still collect it, which is why
the status shows the bundle id. The steps below are the same thing done by hand, and stay available for a run that
needs it.

1. **Download bundle** (`GET /api/projects/:id/remote/bundle?model=&epochs=&patience=&imgsz=&confidence=&checkpoint=`)
   is a compressed zip holding only what training needs: `job.json`, `worker.py` (a copy of the worker), the
   training images with their labels, and `start.pt` when continuing from a checkpoint. No other image, and nothing
   from the database, is in it. `model` is the YOLOv8 size letter, `n` to `x`, default `m`, so a run starts from
   `yolov8m.pt`, or `yolov8m-pose.pt` in a keypoint project; a checkpoint fixes the size instead. A bundle needs the same 10 training images as a local run. The app records the bundle: its id, task,
   class label ids and names, and the keypoint ids per class.
2. `scripts/remote_train.py train <bundle.zip> user@host` runs locally in the `ml` conda env and needs `paramiko`. It
   asks for the password once, keeps one SSH connection, and does everything through it. Before upload it resizes every
   image to the training size on its long side (photos are already compressed, so only this makes the transfer small;
   `--keep-originals` skips it), and targets keep their original size in `job.json` so predictions come back in the
   original pixels (a bundle made by the app has no targets, so this only matters for a hand-made one). On the server it
   uses `~/labelapp-remote/` only. It finds conda, or installs Miniconda there, and
   creates the conda env `labelapp-remote` from conda-forge (Anaconda's default channels would first demand that their
Terms of Service be accepted) with PyTorch and ultralytics **once**, on the first run; later runs reuse it
   (`--reinstall` rebuilds it). It starts `worker.py --standalone` under `nohup`, so a dropped connection does not
   stop training, follows `progress.jsonl`, then downloads `results.json` (the training figures) and `best.pt` as one zip into
   `remote-results/<bundle>/`. `fetch <bundle-id> user@host` collects a run that outlived the terminal, and
   `status` shows how far it is.
3. **Import results** (`POST /api/projects/:id/remote/results`, multipart `results` and optional `weights`) is
   accepted only if the bundle id names a bundle this app made for this project, and for the same task. With `weights`,
   the file becomes a checkpoint of the project, marked *remote* with its model size, which **Draw with** and **Train
   from** offer like any other; the response names it. The format also allows predictions, which the app's own bundles
   never ask for: they are mapped through the bundle's record, so a label or keypoint deleted since loses them, and
   written exactly as a local run writes them, as unreviewed and never over an image a person edited or finished.

A `.pt` file is a PyTorch pickle, and loading it runs code. It is accepted only here, together with the results of
the app's own bundle, which is the user's statement that they trained it on a server they trust. Nothing else ever
imports a model file.

### Reviewing predictions

Predictions are guesses, so they must not be mistaken for ground truth. An image holding them has
`autoBoxed: true`, shows an *auto* badge, stays `todo`, and can be listed with the grid's **Auto** tab.

- **Saving annotations** on the image, which any edit in the annotator does, makes them reviewed for good.
- **Marking the image `done`** accepts them, individually or with the group action. Marking it `todo`
  again takes that back, so undoing a group action restores the *auto* state.
- **Removing predicted boxes** clears every unreviewed prediction in the project at once.

Until reviewed, predictions are not used for training, which would make the model learn from its own
mistakes, and are not exported in any scope. Merely opening an image changes nothing.

If a user deletes every predicted box and leaves the image `todo`, it looks untouched and the next run
draws on it again. Marking it `done` records that it contains no objects.

### Running the job

The job runs in a Python child process, `server/autobox_worker.py`, which needs the `ultralytics` package.
`LABELAPP_PYTHON` chooses the interpreter (default `python3`), and a GPU is used if PyTorch finds one.

- Only one job runs at a time on the server, because it uses all available CPU cores.
- The page shows live progress. Status lives in memory, and a server restart abandons a running job, which
  also ends the worker.
- Cancelling stops the job at once. Predictions already saved are kept. Cancelling during training loses
  its progress and adds no checkpoint; cancelling while drawing keeps the newly trained checkpoint.
- A file that cannot be read or no longer matches its stored size is skipped and listed, and does not fail
  the run. If too few training images can be read, the run fails.
- A write is dropped if the image was edited or marked `done` after the job started, so a run never
  replaces a person's work. A label deleted during the run loses its predictions.
- Deleting the project cancels its job and deletes its checkpoints.

### Files

The worker only reads images. Source folders stay untouched, as section 7 requires. Each training image is
converted to an upright JPEG of at most 640px under `data/tmp/autobox/<job>/`, which is deleted when the
job ends, and left-overs from a restart are deleted at the next start. This also handles GIF and scanned files
with any name, which YOLO would not read directly. Each checkpoint is in `data/models/projects/<projectId>/`, as `<id>.pt`
and `<id>.json`. A model saved before checkpoints existed becomes the checkpoint `best` at the next start. A
checkpoint is reached by id from this list, never by a path from a request. The only model file ever accepted is
the one a remote run returns with its results, as described under "Remote training". The first run downloads `yolov8n.pt` (about 6 MB) to
`data/models/`, so it needs internet access. Ultralytics keeps its settings in `data/models/config/`, and
usage reporting and automatic package installs are turned off.

## 11. Project modes

A project has one mode, chosen when it is created, which decides everything the app shows for it. It cannot
be changed afterwards. To annotate the same images both ways, make two projects; importing a folder into both
does not copy any files.

| | `bbox` (Bounding box) | `keypoint` (Keypoints) |
| --- | --- | --- |
| Annotator tools | Select, Box | Select, Keypoint |
| Canvas and details | Boxes, their coordinates, and changing an instance's label | Keypoints, skeletons, and the keypoint list |
| Shortcuts in the help dialog | Box shortcuts | Keypoint shortcuts, including `N`, `O` and Shift+click |
| Label settings | Name and color | Name, color, keypoints, skeleton, and **Add COCO person** |
| Auto labeling (section 10) | **Auto bounding box** panel with a detector, **Auto** tab and *auto* badge | **Auto keypoints** panel with a pose model, **Auto** tab and *auto* badge |
| COCO export | Detection | Keypoints |
| YOLO export | Detection | Pose |

Both modes share images, folders, status, image labels and their export, group actions, and the native export.

Labels, status and annotations belong to the project, so a mode needs no separate storage. Instead, invariant
6 keeps keypoints out of `bbox` projects, and auto labeling trains the task the mode needs. A checkpoint belongs
to a task, so a detector is never offered in a keypoint project, nor a pose model in a bounding box project. Everything else about a
mode is what the client shows, described in one place: `MODES` in `client/src/modes.ts`. A later mode, such as
segmentation, adds an entry there, a value in `projectModeSchema`, and the rules for its own annotations.

### Projects from before modes

A project gets its mode the first time the new version starts. If any of its labels defines keypoints it
becomes `keypoint`, and otherwise `bbox`. No annotation is changed. So a `bbox` project always satisfies invariant
6, while a `keypoint` project can hold boxes drawn before modes existed. Those boxes are kept and exported as
before, follow their instance when it moves, and are not shown. A box-only instance appears in the instance list,
where it can be deleted or given keypoints. Unreviewed predictions in such a project keep the **Auto** tab until
they are reviewed. Its detector checkpoints stay on disk, hidden, until the project is deleted.

The API does not reject a box sent to a `keypoint` project, because it cannot tell that from one of these. The
annotator has no way to draw one.

## 12. Importing annotations

Annotations made elsewhere, or exported from another project, can be loaded onto the images a project already
holds. This is how labels drawn for one copy of a dataset reach a new project pointed at the same photos, without
drawing them again. Import never adds images: it annotates images already found by a scan or uploaded.

### Files

- **COCO** (`images`, `annotations`, `categories`): boxes, and keypoints with `v > 0`. Skeletons are 1-indexed, as
  in the export.
- **LabelApp JSON**, the native export: annotations, image labels and status.

YOLO is not accepted, because its label files carry no image sizes or paths. Anything else is rejected with `400`
and a reason. An annotation pointing at an unknown image or category is skipped and listed, not fatal.

### Matching images

An entry's `file_name` is matched to a project image by path. The file was usually made from another folder, so
an exact match is not required: the project image whose relative path (section 7) shares the longest run of
trailing segments with the entry wins, down to the bare file name. Two project images equally close make the entry
ambiguous, and it is skipped and listed. So `dataset/val/a/1.jpg` finds `raw/val/a/1.jpg` when no other
`1.jpg` is as close, and a file name alone is enough when it is unique in the project.

A matched image is skipped, and the reason counted, when:

- its file is unavailable (section 7), since its annotations are read-only;
- the entry gives a width or height that differs from the project's, since coordinates would not fit. The
  project's sizes are EXIF-corrected (section 3), so a file made from the raw pixel size can disagree;
- a person already annotated it and `overwrite` is off. Unreviewed predictions (section 10) are not a person's
  work, so an import replaces them either way.

### Labels

Categories are matched to the project's labels by name, ignoring case and surrounding spaces. With `createLabels`
(the default) a category without a label gets one, with the next free color; otherwise its annotations are dropped
and counted. In a `keypoint` project a category's keypoint names are added to its label when missing, and its
skeleton edges too, so a label is only ever extended, never shrunk. In a `bbox` project keypoints are ignored, the
label is created without any, and the count of annotations that lost keypoints is reported. Invariant 6 therefore
holds, and the import is refused with `422` if it ever would not.

### Writing

Boxes are clamped to the image, and an annotation with nothing left inside it is dropped and counted. Keypoints are
clamped too. Each imported image's annotations replace what it had, as one save would, so they are a person's work
and clear `autoBoxed`. Status does not change unless `markDone` is set, which marks every matched image `done`,
including images the file lists with no annotations, since a COCO file saying an image holds nothing is a true
negative. A LabelApp export's own `done` status is applied regardless. An image the file lists with nothing to add is
left untouched. All writes happen in one transaction, labels first.

`dryRun` computes the same summary without writing. The project page previews every chosen file this way, and again
whenever an option changes, so the person reads what will happen before pressing **Import**. The summary reports
images in the file, matched, written, annotations written, labels created or extended, and every skipped count by
reason, with up to 50 per-entry details.

## 13. Sharing a project

A teammate on another computer can continue a project from a **project file** (`<name>.labelapp`). The file holds the
project, not the photos: each person keeps their own copy of the images, wherever they like, and says where it is when
opening the file. Nothing syncs afterwards. The opened project is an independent copy.

### The file

`GET /api/projects/:id/export?format=project` writes gzipped JSON:

```jsonc
{
  "kind": "labelapp-project", "version": 1, "exportedAt": "…",
  "project": { "name", "mode", "labels" },                  // label and keypoint ids kept
  "sources": [{ "key": "s1", "name": "photos", "originalPath": "/home/a/photos", "exclusions": ["bad/x.jpg"] }],
  "images": [{ "source": "s1", "relPath": "cats/1.jpg", "filename", "width", "height", "status",
               "annotations", "imageLabels", "autoBoxed" }]
}
```

- **Every image is included**, in display order, with its annotations, image labels and status. `autoBoxed` is the
  stored flag, so unreviewed predictions stay unreviewed (section 10).
- **Each source folder** is listed with the paths deleted from the project (section 7), so a rescan on the other
  computer does not bring those images back. `originalPath` is only shown as a hint.
- **Uploaded images** exist only in the exporting computer's `data/`, so they are listed under the source key
  `uploads`, with their original filename as `relPath`.
- **Not included:** image files, file states, sizes and times, auto box checkpoints, jobs and remote bundles.

### Opening

`POST /api/projects/open` reads the file, gzipped or plain, up to 256 MB. It refuses with `400` a file that is not a
project file, has another version, or breaks any invariant of section 3, and checks that every image's folder exists
in the file and that no path appears twice in one folder. Paths cannot contain `\`, or empty, `.` or `..` segments,
so they never leave their folder.

`options.paths` gives a path on this computer for each folder key, `uploads` included. Each path passes the same
checks as adding a folder (section 7, `IMPORT_ROOTS` included), and two chosen folders must not overlap. A folder with
no path is left out with its images, and the response lists a warning for it. Uploaded images placed in a folder become
a source of that folder's name. Only one file of a name can be in it, so a later upload with the same name is left out
with a warning.

With `dryRun`, nothing is created. The response gives the project's name, mode, label and image counts, and per folder
its image count, a few sample paths, the resolved path or why it cannot be used, and how many of up to 20 of its files,
spread over the folder, exist under that path. The project list shows **Open a shared project**, which previews the
file as soon as it is chosen and again after each path is typed. **Create project** stays disabled until every typed
path is usable.

Otherwise the project is created in one transaction, keeping the file's mode, labels and the source names, which form
the top of the folder tree as on the exporting computer. Images get new ids and start `missing`, with no stored
size or time. A scan of every chosen folder then starts, and it matches files by path:

- A file found with the stored dimensions becomes `ok` (counted as recovered).
- A file found with other dimensions becomes `changed`.
- A file not found stays `missing`.
- A file in the folder that the project does not have is added as `todo`, unless the file lists it as deleted.

So annotations only become editable on images whose file is there and fits them. If a folder was placed wrongly, the
project page's **Change path** points it at the right one (section 7).

