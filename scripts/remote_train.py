#!/usr/bin/env python
"""Trains a LabelApp bundle on a remote machine over SSH and brings the results back.

    python scripts/remote_train.py train  bundle.zip   user@host [--imgsz 640] [--keep-originals] [--reinstall]
    python scripts/remote_train.py fetch  <bundle-id>  user@host
    python scripts/remote_train.py status <bundle-id>  user@host

The app runs the same script itself for its one-click remote run, with --json (progress as JSON lines on file
descriptor 3), --password-stdin (the password read from stdin, so it is never an argument) and --out <dir>.
With no password the SSH agent and key files are tried instead.

The remote machine only trains (specs.md section 10, "Remote training"). Everything about the
app's format stays local: the app makes the bundle, and the app imports results.json and best.pt afterwards.

One SSH connection carries the whole run, so the password is asked for once and kept only in memory. On the
server everything lives under ~/labelapp-remote/. Miniconda is installed there if no conda exists, and one
conda env, labelapp-remote, is created on the first run and reused afterwards (--reinstall rebuilds it).

Before upload, images are resized so their long side is the training size, since YOLO shrinks them to that
size anyway. Photos are already compressed, so this is what actually makes the transfer small. Predictions
come back in the original pixels, because each target keeps its original size in the job file.
"""

import argparse
import getpass
import io
import json
import os
import posixpath
import shlex
import sys
import tempfile
import time
import zipfile

try:
    import paramiko
except ImportError:
    sys.exit("This script needs paramiko. Run: conda activate ml && pip install paramiko")

JSON_OUT = None  # set to a file object when --json is on

REMOTE_ROOT = "labelapp-remote"
ENV_NAME = "labelapp-remote"
PYTHON_VERSION = "3.11"
# conda-forge only: Anaconda's default channels need their Terms of Service accepted first, and a fresh
# Miniconda refuses to create an env until then. Overriding the channels avoids that question entirely.
CONDA_CHANNELS = "--override-channels -c conda-forge"
MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"


def emit(**message):
    """Machine-readable progress for the app, when asked for. Human output stays on stdout."""
    if JSON_OUT is not None:
        JSON_OUT.write(json.dumps(message) + "\n")
        JSON_OUT.flush()


def connect(target, password_stdin=False):
    user, _, host = target.rpartition("@")
    if not user:
        sys.exit("Give the server as user@host, for example 8f95@oth-aw.de@ki-031.oth-aw.de")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass(f"Password for {user} on {host} (empty to use your SSH key): ")
    emit(type="phase", phase="connecting")
    try:
        if password:
            client.connect(host, username=user, password=password, look_for_keys=False, allow_agent=False, timeout=30)
        else:
            client.connect(host, username=user, look_for_keys=True, allow_agent=True, timeout=30)
    except paramiko.AuthenticationException:
        sys.exit("Login failed: wrong password, or no SSH key the server accepts.")
    except (paramiko.SSHException, OSError) as error:
        sys.exit(f"Could not connect to {host}: {error}")
    finally:
        password = None  # noqa: F841 - nothing keeps it after connecting
    client.get_transport().set_keepalive(30)
    return client


def run(client, command, echo=False, check=True):
    """Runs a login-shell command and returns its output. Login shell, so conda from the profile is found."""
    full = f"bash -lc {shlex.quote(command)}"
    _, stdout, stderr = client.exec_command(full, get_pty=False)
    out = []
    for line in iter(stdout.readline, ""):
        out.append(line)
        if echo:
            print("  | " + line.rstrip())
    code = stdout.channel.recv_exit_status()
    error = stderr.read().decode("utf-8", "replace")
    if check and code != 0:
        sys.exit(f"Remote command failed ({code}): {command}\n{error}")
    return "".join(out), error, code


def progress_bar(done, total, width=40):
    filled = int(width * done / total) if total else width
    sys.stdout.write(f"\r  [{'#' * filled}{'.' * (width - filled)}] {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
    sys.stdout.flush()


def put(sftp, local, remote):
    size = os.path.getsize(local)
    last = [0.0]

    def callback(done, total):
        progress_bar(done, total)
        if time.time() - last[0] > 0.5 or done == total:
            last[0] = time.time()
            emit(type="upload", done=done, total=total)

    sftp.put(local, remote, callback=callback)
    progress_bar(size, size)
    print()


def get(sftp, remote, local):
    sftp.get(remote, local, callback=lambda done, total: progress_bar(done, total))
    print()


def shrink_bundle(path, imgsz):
    """Rewrites the bundle with every image resized to imgsz on its long side, EXIF applied, and the job updated."""
    from PIL import Image, ImageOps

    shrunk = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    shrunk.close()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(shrunk.name, "w", zipfile.ZIP_DEFLATED) as target:
        job = json.loads(source.read("job.json"))
        items = {item["path"]: item for item in job["training"] + job["targets"]}
        before = after = 0
        for info in source.infolist():
            if info.filename == "job.json":
                continue  # written last, with the new sizes
            data = source.read(info.filename)
            before += len(data)
            item = items.get(info.filename)
            if item is None:
                target.writestr(info, data)
                after += len(data)
                continue
            with Image.open(io.BytesIO(data)) as image:
                upright = ImageOps.exif_transpose(image)
                if upright.size != (item["width"], item["height"]):
                    sys.exit(f"{info.filename} is {upright.size}, but the job expects {item['width']}x{item['height']}.")
                if max(upright.size) <= imgsz:
                    target.writestr(info, data)
                    after += len(data)
                    continue
                upright = upright.convert("RGB")
                upright.thumbnail((imgsz, imgsz))
                buffer = io.BytesIO()
                upright.save(buffer, "JPEG", quality=90)
            # The job records the file as it is now; targets remember their original frame for the predictions.
            if "id" in item:
                item["original"] = [item["width"], item["height"]]
            item["width"], item["height"] = upright.size
            target.writestr(info.filename, buffer.getvalue())
            after += buffer.tell()
        target.writestr("job.json", json.dumps(job))
    print(f"Resized images to {imgsz}px: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    emit(type="resized", before=before, after=after)
    return shrunk.name, job


def ensure_env(client, reinstall):
    """Finds conda, or installs Miniconda, and makes sure the env exists with ultralytics. Created once, reused after."""
    home = run(client, "echo $HOME")[0].strip()
    conda, _, code = run(client, "command -v conda || ls $HOME/miniconda3/bin/conda", check=False)
    conda = conda.strip().splitlines()[-1] if conda.strip() else ""
    if not conda:
        print("No conda on the server. Installing Miniconda into ~/miniconda3 ...")
        run(
            client,
            f"mkdir -p {REMOTE_ROOT} && cd {REMOTE_ROOT} && curl -fsSL -o miniconda.sh {MINICONDA_URL}"
            f" && bash miniconda.sh -b -p $HOME/miniconda3 && rm miniconda.sh",
            echo=True,
        )
        conda = f"{home}/miniconda3/bin/conda"
    env_python = f"{posixpath.dirname(posixpath.dirname(conda))}/envs/{ENV_NAME}/bin/python"
    _, _, missing = run(client, f"test -x {env_python}", check=False)
    works = missing == 0 and run(client, f"{env_python} -c 'import ultralytics'", check=False)[2] == 0
    if reinstall or not works:
        emit(type="phase", phase="installing")
        print(f"Creating conda env {ENV_NAME} (python {PYTHON_VERSION}, PyTorch with CUDA, ultralytics). This happens once ...")
        run(client, f"{conda} env remove -y -n {ENV_NAME} >/dev/null 2>&1 || true")
        run(client, f"{conda} create -y -n {ENV_NAME} {CONDA_CHANNELS} python={PYTHON_VERSION}", echo=True)
        run(client, f"{env_python} -m pip install --upgrade pip", echo=False)
        run(client, f"{env_python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", echo=True)
        run(client, f"{env_python} -m pip install ultralytics", echo=True)
    else:
        print(f"Reusing conda env {ENV_NAME}.")
    out, _, _ = run(client, f"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
    print("GPU: " + (out.strip() or "none visible"))
    gpu = out.strip() or None
    out, _, _ = run(client, f"{env_python} -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")
    print("PyTorch, CUDA available: " + out.strip())
    emit(type="server", gpu=gpu, torch=out.strip())
    return env_python


def follow(client, sftp, remote_dir):
    """Prints progress lines as the worker writes them, until it reports done or an error."""
    path = posixpath.join(remote_dir, "progress.jsonl")
    offset = 0
    last_epoch = None
    while True:
        try:
            with sftp.open(path, "r") as file:
                file.seek(offset)
                chunk = file.read().decode("utf-8", "replace")
        except FileNotFoundError:
            chunk = ""
        offset += len(chunk.encode("utf-8"))
        for line in chunk.splitlines():
            try:
                message = json.loads(line)
            except ValueError:
                continue
            kind = message.get("type")
            if kind in ("prepared", "epoch", "trained", "skip", "error", "done"):
                emit(**message)
            elif kind == "pred":
                emit(type="pred", id=message.get("id"))
            if kind == "prepared":
                print(f"Training on {message['trainImages']} images, validating on {message['valImages']}.")
            elif kind == "epoch":
                last_epoch = message["epoch"]
                best = message["bestMap50"]
                sys.stdout.write(f"\r  epoch {last_epoch}, best mAP50 {best:.3f}" if best is not None else f"\r  epoch {last_epoch}")
                sys.stdout.flush()
            elif kind == "trained":
                print(f"\nTrained: {message['epoch']} epochs, best mAP50 {message['bestMap50']}. Predicting ...")
            elif kind == "pred":
                sys.stdout.write(".")
                sys.stdout.flush()
            elif kind == "skip":
                print(f"\n  skipped {message['path']}: {message['reason']}")
            elif kind == "error":
                sys.exit(f"\nThe worker failed: {message['message']}")
            elif kind == "done":
                print("\nDone.")
                return
        _, _, alive = run(client, f"test -e {shlex.quote(posixpath.join(remote_dir, 'worker.pid'))} && kill -0 $(cat {shlex.quote(posixpath.join(remote_dir, 'worker.pid'))})", check=False)
        if alive != 0:
            # No pid alive and no done line: give the file one more read before giving up.
            time.sleep(2)
            continue_reading, _, _ = run(client, f"tail -n 1 {shlex.quote(path)} 2>/dev/null", check=False)
            if '"done"' in continue_reading or '"error"' in continue_reading:
                continue
            sys.exit("\nThe worker is no longer running and reported no result. Check worker.log on the server.")
        time.sleep(5)


def fetch(client, sftp, remote_dir, bundle_id, out=None):
    """Zips results.json and best.pt on the server, downloads that one file, and unpacks it locally."""
    for name in ("results.json", "best.pt"):
        try:
            sftp.stat(posixpath.join(remote_dir, name))
        except FileNotFoundError:
            sys.exit(f"{name} is not there yet. Use 'status' to see how far the run is.")
    emit(type="phase", phase="downloading")
    run(client, f"cd {shlex.quote(remote_dir)} && python3 -c \"import zipfile; z=zipfile.ZipFile('results.zip','w',zipfile.ZIP_DEFLATED); z.write('results.json'); z.write('best.pt'); z.close()\"")
    local_dir = os.path.abspath(out or os.path.join("remote-results", bundle_id[:8]))
    os.makedirs(local_dir, exist_ok=True)
    local_zip = os.path.join(local_dir, "results.zip")
    print("Downloading results ...")
    get(sftp, posixpath.join(remote_dir, "results.zip"), local_zip)
    with zipfile.ZipFile(local_zip) as archive:
        archive.extract("results.json", local_dir)
        archive.extract("best.pt", local_dir)
    os.remove(local_zip)
    print(f"\nResults are in {local_dir}")
    if JSON_OUT is None:
        print("In the app, open the project's auto panel and import results.json and best.pt from there.")
    emit(type="fetched", dir=local_dir)


def main():
    global JSON_OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "fetch", "status"])
    parser.add_argument("bundle", help="bundle.zip for train; the bundle id for fetch and status")
    parser.add_argument("target", help="user@host")
    parser.add_argument("--imgsz", type=int, help="resize images to this long side before upload (default: the bundle's training size)")
    parser.add_argument("--keep-originals", action="store_true", help="upload the images as they are")
    parser.add_argument("--reinstall", action="store_true", help="recreate the conda env on the server")
    parser.add_argument("--json", action="store_true", help="report progress as JSON lines on file descriptor 3 (used by the app)")
    parser.add_argument("--password-stdin", action="store_true", help="read the password from the first line of stdin (used by the app)")
    parser.add_argument("--out", help="folder to put results.json and best.pt in")
    args = parser.parse_args()
    if args.json:
        JSON_OUT = os.fdopen(3, "w", encoding="utf-8")

    if args.command == "train":
        with zipfile.ZipFile(args.bundle) as archive:
            job = json.loads(archive.read("job.json"))
        bundle_id = job["bundleId"]
        upload = args.bundle
        if not args.keep_originals:
            upload, job = shrink_bundle(args.bundle, args.imgsz or job["imgsz"])
    else:
        bundle_id = args.bundle

    client = connect(args.target, args.password_stdin)
    try:
        home = run(client, "echo $HOME")[0].strip()
        remote_dir = posixpath.join(home, REMOTE_ROOT, bundle_id)
        sftp = client.open_sftp()

        if args.command == "train":
            emit(type="phase", phase="preparing")
            env_python = ensure_env(client, args.reinstall)
            run(client, f"mkdir -p {shlex.quote(remote_dir)}")
            print(f"Uploading bundle to {remote_dir} ...")
            emit(type="phase", phase="uploading")
            put(sftp, upload, posixpath.join(remote_dir, "bundle.zip"))
            if upload != args.bundle:
                os.remove(upload)
            run(client, f"cd {shlex.quote(remote_dir)} && python3 -c \"import zipfile; zipfile.ZipFile('bundle.zip').extractall('.')\" && rm bundle.zip")
            if job.get("train", True):
                print(f"Starting training of {job['weights']} for up to {job['maxEpochs']} epochs (patience {job['patience']}) ...")
                emit(type="phase", phase="training")
            else:
                print("Starting prediction with the uploaded checkpoint ...")
                emit(type="phase", phase="predicting")
            run(
                client,
                f"cd {shlex.quote(remote_dir)} && rm -f progress.jsonl results.json && "
                f"nohup {env_python} worker.py --standalone job.json > worker.log 2>&1 & echo $! > {shlex.quote(remote_dir)}/worker.pid",
            )
            print(f"If this terminal closes, run:  python scripts/remote_train.py fetch {bundle_id} {args.target}")
            follow(client, sftp, remote_dir)
            fetch(client, sftp, remote_dir, bundle_id, args.out)
        elif args.command == "status":
            out, _, _ = run(client, f"tail -n 3 {shlex.quote(posixpath.join(remote_dir, 'progress.jsonl'))} 2>/dev/null", check=False)
            print(out.strip() or "No progress recorded for this bundle.")
        else:
            fetch(client, sftp, remote_dir, bundle_id, args.out)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as error:
        # The app reads the reason from the protocol, not from stderr.
        if error.code not in (None, 0) and JSON_OUT is not None:
            emit(type="error", message=str(error.code))
        raise
