"""Predicts boxes for a job's target images, first training a YOLOv8 nano detector if the job says to.

Run by server/autobox.ts as: python autobox_worker.py <job.json>

Progress goes out as one JSON object per line on file descriptor 3, because Ultralytics logs to
stdout. This process only reads the image files, and writes nothing outside the job's work folder
and the model cache in its working directory. The server saves the trained model for later jobs.

Every image goes through load_upright, so training and prediction use the EXIF-corrected frame that
all stored coordinates are in (specs.md section 3), whatever a library would do by itself.
"""

import json
import os
import sys
import threading


def main():
    with open(sys.argv[1], encoding="utf-8") as file:
        job = json.load(file)
    protocol = os.fdopen(3, "w", encoding="utf-8")

    def send(**message):
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    # The server holds the other end of stdin, so this ends the job if the server goes away.
    threading.Thread(target=exit_when_stdin_closes, daemon=True).start()

    try:
        weights = train(job, send) if job["train"] else job["modelPath"]
        predict(job, weights, send)
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
                file.write(f"{box['cls']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
        counts[split] += 1

    if counts["train"] == 0 or counts["val"] == 0:
        raise ValueError("Too few of the annotated images could be read to train a model.")
    send(type="prepared", trainImages=counts["train"], valImages=counts["val"])

    # JSON is valid YAML, which keeps any label name safe.
    config = os.path.join(root, "data.yaml")
    with open(config, "w", encoding="utf-8") as file:
        names = {str(index): name for index, name in enumerate(job["classes"])}
        json.dump({"path": root, "train": "images/train", "val": "images/val", "names": names}, file)
    return config


def train(job, send):
    from ultralytics import YOLO

    config = prepare_dataset(job, send)
    model = YOLO(job["weights"])
    best = {"map50": None}

    def on_epoch_end(trainer):
        map50 = float(trainer.metrics.get("metrics/mAP50(B)", 0))
        if trainer.fitness is not None and trainer.fitness == trainer.best_fitness:
            best["map50"] = map50
        send(type="epoch", epoch=trainer.epoch + 1, bestMap50=best["map50"])

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.train(
        data=config,
        epochs=job["maxEpochs"],
        patience=job["patience"],
        imgsz=job["imgsz"],
        workers=0,  # One process, so killing it stops the whole job.
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
    for item in job["targets"]:
        try:
            image = load_upright(item["path"], item["width"], item["height"])
            pixels = np.asarray(image)[:, :, ::-1]  # A NumPy image means BGR to Ultralytics.
            result = model.predict(pixels, conf=job["confidence"], imgsz=job["imgsz"], verbose=False)[0]
            boxes = [
                {"cls": int(cls), "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                for (x1, y1, x2, y2), cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())
            ]
        except Exception as error:
            send(type="skip", path=item["path"], reason=str(error))
            continue
        send(type="pred", id=item["id"], boxes=boxes)


if __name__ == "__main__":
    main()
