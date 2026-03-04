import argparse
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A

# Allow importing archs.py from the portable/unet folder.
UNET_DIR = Path(__file__).resolve().parents[1]
if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

MODELS_DIR = UNET_DIR / "models"

import archs  # noqa: E402


def read_image(image_path: str) -> np.ndarray:
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None, store_original_size=False):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform
        self.store_original_size = store_original_size

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_id + self.img_ext}")

        if self.store_original_size:
            original_height, original_width = img.shape[0], img.shape[1]

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        img = img.astype("float32") / 255
        img = img.transpose(2, 0, 1)

        meta = {"img_id": img_id}
        if self.store_original_size:
            meta.update({"original_height": original_height, "original_width": original_width})
        return img, meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rail_UNet_woDS_20250617_214524", help="model name")
    parser.add_argument("--input-dir", default=os.path.join("../inputs", "test3", "images"))
    parser.add_argument("--output-dir", default=os.path.join("outputs", "predict1"))
    parser.add_argument("--model-path", default="", help="absolute path to model.pth (overrides --name)")
    parser.add_argument("--image", default="", help="single image path (overrides --input-dir)")
    return parser.parse_args()


def build_model(config):
    if config["arch"] == "FPN":
        blocks = [2, 4, 23, 3]
        model = archs.__dict__[config["arch"]](blocks, config["num_classes"], back_bone="resnet101")
    elif config["arch"] == "SegNet":
        model = archs.__dict__[config["arch"]](config["num_classes"])
    elif config["arch"] == "AttU_Net":
        model = archs.__dict__[config["arch"]]()
    elif config["arch"] == "TransUnet":
        model = archs.__dict__[config["arch"]](num_classes=config["num_classes"])
    elif config["arch"] == "TransUnet2":
        model = archs.__dict__[config["arch"]](num_classes=config["num_classes"])
    elif config["arch"] == "GRUUNet":
        model = archs.__dict__[config["arch"]](num_classes=config["num_classes"])
    elif config["arch"] == "RUNet":
        model = archs.__dict__[config["arch"]](num_classes=config["num_classes"])
    elif config["arch"] == "RUNet2":
        model = archs.__dict__[config["arch"]](num_classes=config["num_classes"])
    elif config["arch"] == "D_UNet":
        model = archs.__dict__[config["arch"]](in_channels=3, num_classes=config["num_classes"])
    elif config["arch"] == "CUNet":
        model = archs.__dict__[config["arch"]](num_classes=config["num_classes"])
    elif config["arch"] == "DFANet":
        ch_cfg = [[8, 48, 96], [240, 144, 288], [240, 144, 288]]
        model = archs.__dict__[config["arch"]](ch_cfg, 64, 1)
    elif config["arch"] == "SETR":
        model = archs.__dict__[config["arch"]](
            num_classes=config["num_classes"],
            image_size=512,
            patch_size=512 // 16,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=2048,
        )
    elif config["arch"] == "SETR2":
        model = archs.__dict__[config["arch"]](
            num_classes=config["num_classes"],
            image_size=512,
            patch_size=512 // 16,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=2048,
        )
    elif config["arch"] == "Double_UNet":
        model = archs.__dict__[config["arch"]]()
    elif config["arch"] == "build_doubleunet":
        model = archs.__dict__[config["arch"]]()
    elif config["arch"] == "C2FNet":
        model = archs.__dict__[config["arch"]]()
    elif config["arch"] == "MENet":
        model = archs.__dict__[config["arch"]]()
    elif config["arch"] == "TwinLiteNet":
        model = archs.__dict__[config["arch"]]()
    elif config["arch"] == "DCSAU_Net":
        model = archs.__dict__[config["arch"]]()
    else:
        model = archs.__dict__[config["arch"]](
            config["num_classes"],
            config["input_channels"],
            config["deep_supervision"],
        )
    return model


def load_config_from_model_path(model_path: str) -> dict:
    model_path = Path(model_path)
    config_path = model_path.parent / "config.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found next to model: {config_path}")
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model_from_path(model_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(config)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def predict_mask_for_image(model_path: str, image_path: str, device: str = "auto") -> np.ndarray:
    config = load_config_from_model_path(model_path)
    device_obj = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)

    model = load_model_from_path(model_path, config, device_obj)
    image = read_image(image_path)
    original_height, original_width = image.shape[:2]

    val_transform = A.Compose([
        A.Resize(height=config["input_h"], width=config["input_w"]),
        A.Normalize(),
    ])

    augmented = val_transform(image=image)
    img = augmented["image"].astype("float32") / 255
    img = img.transpose(2, 0, 1)
    inputs = torch.from_numpy(img).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        if config.get("deep_supervision"):
            output = model(inputs)[-1]
        else:
            output = model(inputs)
        output = torch.sigmoid(output).cpu().numpy()[0, 0]

    resized_output = cv2.resize(
        output,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return (resized_output * 255).astype("uint8")


def main():
    args = parse_args()

    if args.model_path and args.image:
        mask = predict_mask_for_image(args.model_path, args.image)
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
        cv2.imwrite(output_path, mask)
        print(f"Saved mask to: {output_path}")
        return

    config_path = MODELS_DIR / args.name / "config.yml"
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("-" * 20)
    for key in config.keys():
        print(f"{key}: {str(config[key])}")
    print("-" * 20)

    cudnn.benchmark = True

    print(f"=> creating model {config['arch']}")
    model = build_model(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_ids = glob(os.path.join(args.input_dir, "*" + config["img_ext"]))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    if not img_ids:
        raise SystemExit(f"No images found in {args.input_dir}")

    model_path = MODELS_DIR / config["name"] / "model.pth"
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    if model_path.stat().st_size < 1024:
        raise SystemExit(f"Model file looks empty/placeholder: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_transform = A.Compose([
        A.Resize(height=config["input_h"], width=config["input_w"]),
        A.Normalize(),
    ])

    infer_dataset = InferenceDataset(
        img_ids=img_ids,
        img_dir=args.input_dir,
        img_ext=config["img_ext"],
        transform=val_transform,
        store_original_size=True,
    )

    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    for c in range(config["num_classes"]):
        os.makedirs(os.path.join(args.output_dir, config["name"], str(c)), exist_ok=True)

    with torch.no_grad():
        for inputs, meta in infer_loader:
            inputs = inputs.to(device)

            if config["deep_supervision"]:
                output = model(inputs)[-1]
            else:
                output = model(inputs)

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config["num_classes"]):
                    orig_height = meta["original_height"][i].item()
                    orig_width = meta["original_width"][i].item()

                    resized_output = cv2.resize(
                        output[i, c],
                        (orig_width, orig_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    cv2.imwrite(
                        os.path.join(args.output_dir, config["name"], str(c), meta["img_id"][i] + ".jpg"),
                        (resized_output * 255).astype("uint8"),
                    )

    print("Inference completed.")


if __name__ == "__main__":
    main()
