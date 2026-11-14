# LabelApp

LabelApp is an image labeling tool that you host yourself. It labels **bounding boxes** and **keypoints**, and is
built for fast annotation from the keyboard. It exports to COCO, YOLO and its own JSON format. It can also train a
YOLOv8 model on your labels, either on this computer or on a remote GPU server, and let that model label the rest of
your images for you to review.

It is a single-user tool with no login. By default it accepts connections only from this computer.

## Requirements

| What | Needed for | Version |
| --- | --- | --- |
| [Node.js](https://nodejs.org/) | The app | 24 or later |
| Python with the packages in `requirements.txt` | Auto labeling and remote training only | 3.10 or later; tested with 3.12 |
| A Linux server with an NVIDIA GPU, reachable over SSH | Remote training only | |

You can label, import and export without Python.

## Install

```sh
git clone <this repository> labelapp
cd labelapp
npm install
npm run build
```

### Python, for auto labeling

Use a dedicated conda environment. The examples here call it `ml`.

```sh
conda create -n ml python=3.12
conda activate ml
pip install -r requirements.txt
```

For training on this computer's NVIDIA GPU, install the CUDA build of PyTorch **before** running
`pip install -r requirements.txt`. Use the command for your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/), for example:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Without a GPU, training runs on the CPU. That works, but it is slow.

Check that `python3` really comes from the environment. Tools such as pyenv can put another Python ahead of it on
your `PATH`:

```sh
conda activate ml
which python3        # should print a path inside .../envs/ml/
```

To be safe, give the app the environment's Python explicitly when you start it (see below).

## Run

```sh
conda activate ml                                   # only needed for auto labeling
LABELAPP_PYTHON="$CONDA_PREFIX/bin/python" npm start
```

Open **http://127.0.0.1:3001**.

All projects, uploaded images and trained models are stored in `data/` in the app's folder. Back up that folder to
back up your work.

### Settings

Set these as environment variables when you start the app.

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `3001` | Port to listen on |
| `HOST` | `127.0.0.1` | Address to listen on |
| `DATA_DIR` | `./data` | Where projects, uploads and models are stored |
| `IMPORT_ROOTS` | none | Folders that folder import may read from, separated by `:`. Required when `HOST` is not a loopback address |
| `LABELAPP_PYTHON` | `python3` | The Python used for auto labeling |

> **Before you change `HOST`:** the app has no login, so anyone who can reach the port can read and change every
> project. Use it only on a network you trust, and set `IMPORT_ROOTS` to limit which folders can be imported.

## Use

### 1. Create a project

On the project list, enter a name and choose a mode. **You cannot change the mode later.**

- **Bounding box**: draw boxes around objects.
- **Keypoints**: place named points, such as `left_eye`, on each object. Each label can define its own keypoints and
  a skeleton connecting them. **Add COCO person** fills in the standard 17 body keypoints.

To label the same images both ways, create two projects. Importing a folder into both does not copy the files.

### 2. Add labels and images

Add labels (object classes) in the project settings. Then add images in one of two ways:

- **Upload**: drag files onto the page or choose them. They are copied into `data/`. Supported formats are JPEG,
  PNG, WebP, GIF and BMP.
- **Import a folder**: type the absolute path of a folder on this computer, such as `/data/street/2024`. The app
  finds every image in it and in its subfolders, and leaves the files where they are. It never changes, moves or
  deletes anything in that folder. Use **Rescan** to pick up files added later. This works best for large datasets.

The image grid shows folders as a tree. You can select every image under a folder and add an image label to all of
them at once.

### 3. Annotate

Open an image from the grid. Changes save automatically, and each image has its own undo history.

- In a **Bounding box** project, drag to draw a box. Drag a box's edges or corners to adjust it.
- In a **Keypoints** project, click to place the next keypoint of the current object. Shift+click places a
  keypoint as occluded.
- When an image is finished, press **Ctrl+Enter**. This marks it **done** and opens the next **todo** image. A done
  image with no annotations counts as a confirmed empty image.

| Key | Action |
| --- | --- |
| `1`–`9` | Choose the active label |
| `B` / `K` / `V` | Box tool / Keypoint tool / Select tool |
| `A` / `D` | Previous / next image |
| `Ctrl+Enter` | Mark done and go to the next todo image |
| `Ctrl+Z`, `Ctrl+Shift+Z` | Undo, redo |
| `Delete` | Delete the selected keypoint or object |
| Arrow keys | Nudge the selection by 1 px; hold `Shift` for 10 px |
| `N` | Skip to the next keypoint not yet placed |
| `O` | Switch the selected keypoint between visible and occluded |
| `H` | Hide or show annotations |
| `F`, `+`, `-` | Fit to window, zoom in, zoom out |
| `Space` + drag, or the middle mouse button | Pan |
| `?` | Show all shortcuts |

### 4. Let a model do some of the work (optional)

This needs Python with `requirements.txt` installed. Once at least **10 images** have saved annotations, open the
**Auto bounding box** or **Auto keypoints** panel on the project page:

1. **Train and draw** trains a model on your annotated images, then annotates the todo images you have not touched.
   A bounding box project trains `yolov8n`. A keypoint project trains the pose model `yolov8n-pose`. The first run
   downloads these base weights, so it needs internet access.
2. Model annotations are marked **auto**. They are never exported or used for training until you review them. Open
   the **Auto** tab, fix what is wrong, and mark each image done.
3. Run it again. Each run trains on more reviewed images, so the model improves. Every trained model is kept as a
   checkpoint: you can continue training from it or use it to draw again. **Draw with** can also use a checkpoint
   from another project with the same labels.

The model never overwrites an image you have edited or marked done.

### 5. Train on a remote GPU server (optional)

When this computer has no GPU, open **Train on a remote server** in the same panel:

1. Enter the server as `user@host`. Enter its password, or leave the password empty to use your SSH key. The
   password is used for one login and is never stored or logged.
2. Choose the model size. The default is **medium**: `yolov8m` for boxes, or `yolov8m-pose` for keypoints.
3. Click **Train on server and draw**.

The app uploads only the annotated images used for training. The rest of your dataset never leaves this computer.
On the first run, the server gets Miniconda (if it has no conda) and a conda environment with PyTorch and ultralytics,
all under `~/labelapp-remote/`. Later runs reuse that environment. When training ends, the model is downloaded, saved
as a checkpoint, and used here to draw on your remaining images.

Training keeps running if the connection drops. To collect a run later, use the bundle id shown in the status:

```sh
conda activate ml
python scripts/remote_train.py fetch <bundle-id> user@host
```

For a server the app cannot reach directly, download the bundle from the panel and run it by hand with
`scripts/remote_train.py train <bundle.zip> user@host`. Then import the results in the panel.

### 6. Export

Use **Export** on the project page. By default only **done** images are exported. Todo images you have not touched
are never exported, because an empty YOLO label file would claim the image contains no objects.

| Format | Contents |
| --- | --- |
| COCO JSON | Detection, or keypoints with skeletons in a keypoint project |
| YOLO (zip) | Detection, or pose in a keypoint project, with `data.yaml`; images optional |
| LabelApp JSON | Everything, without loss of detail. It can be imported back |
| Image labels (zip) | Only the whole-image labels, for classification |
| Project file (`.labelapp`) | The project for sharing; see below |

### Import existing annotations

**Import annotations** loads a COCO file or a LabelApp JSON export onto images already in the project. Entries are
matched to images by file path, so the file can come from another folder layout. Import does not add images.

### Share a project with a teammate

1. Export the **Project file**. It contains the labels, annotations and status, but not the images.
2. Your teammate gets the same images onto their own computer, in any folder.
3. On the project list, they choose **Open a shared project**, pick the file, and enter where each image folder is
   on their computer. The app checks the paths before it creates the project.

The opened project is an independent copy. Later changes are not synced in either direction.

## Development

```sh
npm run dev         # server on :3001 with reload, and Vite on :5173; open http://localhost:5173
npm run typecheck
npm test            # unit and API tests
npm run test:e2e    # browser tests (Playwright, uses the system Chrome)
```

`specs.md` describes the app's full behavior and is the reference when changing it. `CLAUDE.md` describes the code's
structure and the conventions it relies on.
