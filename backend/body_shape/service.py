import io
import os
import threading
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy import ndimage as ndi

IMG_H, IMG_W = 192, 96
SIM_TOL, APPLE_RATIO, WAIST_RATIO_HG = 0.05, 0.95, 0.80
TTA = [(0, 0, False), (0, 3, False), (0, -3, False), (2, 0, False), (-2, 0, False), (0, 0, True)]
MODEL_VERSION = "front-gender-v4.2"
SHAPE_LABELS = ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"]
_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "body_shape_front.pt")
_SEGMENTATION_CACHE = Path(__file__).resolve().parents[2] / "models" / "torch-checkpoints"


class Tower(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.net = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 96),
                                 nn.AdaptiveAvgPool2d((6, 3)), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class BMNet(nn.Module):
    def __init__(self, n_aux=1, n_out=5):
        super().__init__()
        self.front = Tower()
        self.head = nn.Sequential(nn.Linear(1728 + n_aux, 256), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x, a):
        return self.head(torch.cat([self.front(x[:, 0:1]), a], 1))


_segmentation = None
_segmentation_lock = threading.Lock()


def get_person_segmenter():
    global _segmentation

    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

    with _segmentation_lock:
        if _segmentation is None:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            state = torch.hub.load_state_dict_from_url(
                weights.url,
                model_dir=str(_SEGMENTATION_CACHE),
                map_location="cpu",
                check_hash=True,
                weights_only=True,
            )
            model = deeplabv3_resnet50(
                weights=None, weights_backbone=None,
                num_classes=len(weights.meta["categories"]), aux_loss=True,
            )
            model.load_state_dict(state)
            _segmentation = model.eval()
    return _segmentation


def person_mask(img, size=800):
    from torchvision import transforms

    model = get_person_segmenter()
    im = ImageOps.exif_transpose(img).convert("RGB")
    s = size / max(im.size)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    tensor = transforms.functional.to_tensor(im)
    tensor = transforms.functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.inference_mode():
        out = model(tensor.unsqueeze(0))["out"][0]
    m = (out.argmax(0) == 15).numpy()
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    return ndi.binary_fill_holes(m)


def norm_mask(m):
    rows = np.where(m.any(1))[0]
    crop = m[rows[0]:rows[-1] + 1]
    s = IMG_H / crop.shape[0]
    cx = ndi.center_of_mass(crop)[1]
    new_w = max(1, int(round(crop.shape[1] * s)))
    arr = np.array(Image.fromarray((crop * 255).astype("uint8")).resize((new_w, IMG_H), Image.BILINEAR))
    x0 = IMG_W // 2 - int(round(cx * s))
    src_l, src_r = max(0, -x0), min(new_w, IMG_W - x0)
    canvas = np.zeros((IMG_H, IMG_W), np.uint8)
    if src_r > src_l:
        canvas[:, src_l + x0: src_l + x0 + (src_r - src_l)] = arr[:, src_l:src_r]
    return canvas


def photo_quality(mask):
    warns = []
    rows = np.where(mask.any(1))[0]
    if len(rows) == 0:
        return ["no_person_detected"]
    H = mask.shape[0]
    if rows[-1] >= H - 2:
        warns.append("feet_cut_off")
    if rows[0] <= 1:
        warns.append("head_cut_off")
    if (rows[-1] - rows[0] + 1) / H < 0.6:
        warns.append("person_too_small_in_frame")
    return warns


def classify_vec(d, rw, sim=SIM_TOL, apple=APPLE_RATIO, hg=WAIST_RATIO_HG):
    out = np.where(d > 0, "inverted_triangle", "pear").astype(object)
    bal = np.abs(d) <= sim
    out[bal] = np.where(rw[bal] < hg, "hourglass", "rectangle")
    out[rw >= apple] = "apple"
    return out


def to3(label):
    return label if label in ("pear", "inverted_triangle") else "balanced"


def _variants(th, m=(0.01, 0.015, 0.02)):
    sim, apple, hg = th
    return [(sim - m[0], apple, hg), (sim + m[0], apple, hg), (sim, apple - m[1], hg),
            (sim, apple + m[1], hg), (sim, apple, hg - m[2]), (sim, apple, hg + m[2])]


def possible_shapes(d, rw, th):
    labels = {classify_vec(d, rw, *th)[0]}
    for t in _variants(th):
        labels.add(classify_vec(d, rw, *t)[0])
    return sorted(str(x) for x in labels)


def is_borderline(d, rw, th):
    return len(possible_shapes(d, rw, th)) > 1


class BodyShapeModel:
    def __init__(self, ckpt_path, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("aux", "full") != "gender":
            raise ValueError("Ye checkpoint height/weight maangta hai. Gender-only checkpoint (v4.2) chahiye.")
        states = ck["states"] if "states" in ck else [ck["state"]]
        self.nets = []
        for st in states:
            net = BMNet(n_aux=1)
            net.load_state_dict(st)
            self.nets.append(net.eval().to(self.dev))
        self.mu, self.sd, self.amu, self.asd, self.th = ck["mu"], ck["sd"], ck["amu"], ck["asd"], ck["th"]

    @torch.no_grad()
    def predict(self, front, gender="male"):
        fmask = person_mask(front)
        warns = photo_quality(fmask)
        if "no_person_detected" in warns:
            return {"ok": False, "warnings": warns}
        fm = norm_mask(fmask)
        x = torch.from_numpy(np.stack([fm, np.zeros_like(fm)])[None]).float().div(255).to(self.dev)
        aux = np.array([[float(gender == "male")]], "float32")
        a = torch.from_numpy(((aux - self.amu) / self.asd).astype("float32")).to(self.dev)
        outs = []
        for net in self.nets:
            acc = 0
            for dy, dx, fl in TTA:
                xv = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
                if fl:
                    xv = xv.clone()
                    xv[:, 0] = xv[:, 0].flip(-1)
                acc = acc + net(xv, a)
            outs.append((acc / len(TTA)).cpu().numpy()[0])
        o = np.mean(outs, axis=0) * self.sd + self.mu
        d, rw = np.array([o[3]]), np.array([o[4]])
        shape = classify_vec(d, rw, *self.th)[0]
        return {
            "ok": True,
            "shape": str(shape),
            "shape_3class": to3(str(shape)),
            "borderline": is_borderline(d, rw, self.th),
            "possible_shapes": possible_shapes(d, rw, self.th),
            "warnings": warns,
        }


_model = None
_lock = threading.Lock()


def get_model(weights_path=None):
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = BodyShapeModel(weights_path or _WEIGHTS)
    return _model


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def _norm_gender(gender):
    return "female" if str(gender).strip().lower() in ("female", "women", "woman", "f") else "male"


def predict_body_shape_detailed(image, gender="male"):
    try:
        res = get_model().predict(_to_pil(image), _norm_gender(gender))
        res["model_version"] = MODEL_VERSION
        return res
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "warnings": []}


def predict_body_shape(image, gender="male"):
    res = predict_body_shape_detailed(image, gender)
    return res["shape"] if res.get("ok") else None
