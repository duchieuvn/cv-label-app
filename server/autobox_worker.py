"""Predicts boxes, or boxes with keypoints, for a job's target images, first training a YOLOv8 model if the job says to.

Run by server/autobox.ts as:        python autobox_worker.py <job.json>
Run on a remote machine as:          python autobox_worker.py --standalone <job.json>

In the first form, progress goes out as one JSON object per line on file descriptor 3, because Ultralytics
logs to stdout, and the process ends when the server closes its stdin. In standalone mode (specs.md section
10, "Remote training") the job file comes from a bundle: image paths are relative to the job file, progress
is appended to progress.jsonl beside it, and the end result is results.json plus best.pt in the same folder,
for the app to import. Nothing is written anywhere else except the model cache in the working directory.

Every image goes through load_upright, so training and prediction use the EXIF-corrected frame that all
stored coordinates are in (specs.md section 3), whatever a library would do by itself. A bundle may carry
images resized for transfer; a target then records its original size and predictions are scaled back to it.
"""

import datetime
import json
import os
import shutil
import sys
import threading


def main():
    standalone = "--standalone" in sys.argv
    job_path = os.path.abspath(sys.argv[-1])
    with open(job_path, encoding="utf-8") as file:
        job = json.load(file)
    job.setdefault("task", "detect")
    base = os.path.dirname(job_path)

    if standalone:
        # Paths in a bundle are relative to the job file, and the work folder sits beside it.
        for item in job["training"] + job["targets"]:
            item["path"] = os.path.join(base, item["path"])
        if job.get("weights") and not job["weights"].startswith("yolov8"):
            job["weights"] = os.path.join(base, job["weights"])
        if job.get("modelPath"):
            job["modelPath"] = os.path.join(base, job["modelPath"])
        job["workDir"] = os.path.join(base, "work")
        job.setdefault("workers", 4)
        progress = open(os.path.join(base, "progress.jsonl"), "a", encoding="utf-8")
        results = {"predictions": [], "skipped": []}

        def send(**message):
            line = json.dumps(message)
            progress.write(line + "\n")
            progress.flush()
            print(line, flush=True)
            if message["type"] == "pred":
                results["predictions"].append({"id": message["id"], "boxes": message["boxes"]})
            elif message["type"] == "skip":
                results["skipped"].append({"path": message["path"], "reason": message["reason"]})
            elif message["type"] == "trained":
                results.update(
                    epochs=message["epoch"],
                    bestMap50=message["bestMap50"],
                    stoppedEarly=message["stoppedEarly"],
                    trainedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            elif message["type"] == "prepared":
                results["trainImages"] = message["trainImages"]

    else:
        protocol = os.fdopen(3, "w", encoding="utf-8")

        def send(**message):
            protocol.write(json.dumps(message) + "\n")
            protocol.flush()

        # The server holds the other end of stdin, so this ends the job if the server goes away.
        threading.Thread(target=exit_when_stdin_closes, daemon=True).start()

    try:
        weights = train(job, send) if job["train"] else job["modelPath"]
        predict(job, weights, send)
        if standalone:
            shutil.copyfile(weights, os.path.join(base, "best.pt"))
            results.update(bundleId=job["bundleId"], task=job["task"])
            results.setdefault("trainedAt", datetime.datetime.now(datetime.timezone.utc).isoformat())
            results.setdefault("trainImages", 0)
            results.setdefault("epochs", 0)
            results.setdefault("bestMap50", None)
            with open(os.path.join(base, "results.json.tmp"), "w", encoding="utf-8") as file:
                json.dump(results, file)
            os.replace(os.path.join(base, "results.json.tmp"), os.path.join(base, "results.json"))
            send(type="done")
    except ImportError as error:
        send(
            type="error",
            message=f"{sys.executable} cannot run auto boxing ({error}). "
            "Run 'pip install ultralytics' for it, or point LABELAPP_PYTHON at a Python that has it.",
        )
        sys.exit(1)
    except Exception as error:
        send(type="error", message=str(error) or type(error).__name__)
        sys.exit(1)


def exit_when_stdin_closes():
    sys.stdin.read()
    os._exit(1)


def load_upright(path, width, height):
    """Opens an image as RGB with EXIF orientation applied, if it still matches its stored size."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image)
        if upright.size != (width, height):
            raise ValueError(f"the file is {upright.width}x{upright.height}, but {width}x{height} is on record")
        return upright.convert("RGB")


def device_of(job):
    """The device the job asks for, or the first GPU when there is one."""
    if job.get("device") is not None:
        return job["device"]
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def label_line(box):
    parts = [str(box["cls"]), f"{box['cx']:.6f}", f"{box['cy']:.6f}", f"{box['w']:.6f}", f"{box['h']:.6f}"]
    for index, value in enumerate(box.get("keypoints") or []):
        # x and y are normalized; every third value is the visibility flag.
        parts.append(str(int(value)) if index % 3 == 2 else f"{value:.6f}")
    return " ".join(parts) + "\n"


def prepare_dataset(job, send):
    """Writes a dataset of resized JPEGs, which also covers formats and names YOLO would not accept."""
    root = os.path.join(job["workDir"], "dataset")
    counts = {"train": 0, "val": 0}
    for index, item in enumerate(job["training"]):
        split = item["split"]
        try:
            image = load_upright(item["path"], item["width"], item["height"])
        except Exception as error:
            send(type="skip", path=item["path"], reason=str(error))
            continue
        # YOLO would shrink the image to imgsz on every read, so nothing is lost by doing it once here.
        image.thumbnail((job["imgsz"], job["imgsz"]))
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(root, kind, split), exist_ok=True)
        image.save(os.path.join(root, "images", split, f"{index:07}.jpg"), quality=95)
        with open(os.path.join(root, "labels", split, f"{index:07}.txt"), "w", encoding="utf-8") as file:
            for box in item["boxes"]:
                file.write(label_line(box))
        counts[split] += 1

    if counts["train"] == 0 or counts["val"] == 0:
        raise ValueError("Too few of the annotated images could be read to train a model.")
    send(type="prepared", trainImages=counts["train"], valImages=counts["val"])

    # JSON is valid YAML, which keeps any label name safe.
    config = os.path.join(root, "data.yaml")
    with open(config, "w", encoding="utf-8") as file:
        names = {str(index): name for index, name in enumerate(job["classes"])}
        data = {"path": root, "train": "images/train", "val": "images/val", "names": names}
        if job["task"] == "pose":
            data["kpt_shape"] = job["kptShape"]
        json.dump(data, file)
    return config


def train(job, send):
    from ultralytics import YOLO

    config = prepare_dataset(job, send)
    model = YOLO(job["weights"])
    best = {"map50": None}
    # Pose models report box and pose accuracy; the pose figure is what matters for keypoints.
    metric = "metrics/mAP50(P)" if job["task"] == "pose" else "metrics/mAP50(B)"

    def on_epoch_end(trainer):
        map50 = float(trainer.metrics.get(metric, 0))
        if trainer.fitness is not None and trainer.fitness == trainer.best_fitness:
            best["map50"] = map50
        send(type="epoch", epoch=trainer.epoch + 1, bestMap50=best["map50"])

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.train(
        data=config,
        epochs=job["maxEpochs"],
        patience=job["patience"],
        imgsz=job["imgsz"],
        batch=job.get("batch", 16),
        device=device_of(job),
        workers=job.get("workers", 0),  # One process locally, so killing it stops the whole job.
        project=job["workDir"],
        name="train",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    trainer = model.trainer
    send(
        type="trained",
        epoch=trainer.epoch + 1,
        stoppedEarly=trainer.epoch + 1 < job["maxEpochs"],
        bestMap50=best["map50"],
        weights=str(trainer.best),
    )
    return str(trainer.best)


def predict(job, weights, send):
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(weights)
    device = device_of(job)
    for item in job["targets"]:
        try:
            image = load_upright(item["path"], item["width"], item["height"])
            # A bundle may hold a smaller copy of the image; predictions go back to the original frame.
            original = item.get("original") or [item["width"], item["height"]]
            sx, sy = original[0] / item["width"], original[1] / item["height"]
            pixels = np.asarray(image)[:, :, ::-1]  # A NumPy image means BGR to Ultralytics.
            result = model.predict(pixels, conf=job["confidence"], imgsz=job["imgsz"], device=device, verbose=False)[0]
            boxes = [
                {"cls": int(cls), "x": x1 * sx, "y": y1 * sy, "w": (x2 - x1) * sx, "h": (y2 - y1) * sy}
                for (x1, y1, x2, y2), cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())
            ]
            if job["task"] == "pose" and result.keypoints is not None and len(boxes) > 0:
                xy = result.keypoints.xy.tolist()
                conf = result.keypoints.conf
                conf = conf.tolist() if conf is not None else [[1.0] * len(points) for points in xy]
                for box, points, scores in zip(boxes, xy, conf):
                    box["keypoints"] = [[x * sx, y * sy, score] for (x, y), score in zip(points, scores)]
        except Exception as error:
            send(type="skip", path=item["path"], reason=str(error))
            continue
        send(type="pred", id=item["id"], boxes=boxes)


if __name__ == "__main__":
    main()
