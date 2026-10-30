"""Turns any product photo into the 28x28 grayscale format the CNN was trained on."""
import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def preprocess(img, invert="auto", crop=True):
    """
    img    : PIL image (any mode)
    invert : "auto" | "yes" | "no"
             Fashion-MNIST has a BLACK background and a light item.
             Normal product photos have a white background, so they must be inverted.
    crop   : crop to the item and centre it, like the training images
    returns: (28, 28) uint8 array
    """
    # Transparent PNGs: put them on a white background first
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)

    arr = np.asarray(ImageOps.grayscale(img.convert("RGB")), dtype=np.uint8)

    # Decide whether to invert by looking at the image border
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    do_invert = (border.mean() > 127) if invert == "auto" else (invert == "yes")
    if do_invert:
        arr = 255 - arr

    pil = Image.fromarray(arr)

    if crop:
        mask = arr > max(40, int(0.25 * arr.max()))
        if mask.any():
            ys, xs = np.where(mask)
            pil = pil.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # Pad to a square (black), keep a small margin, then shrink to 28x28
    w, h = pil.size
    side = max(w, h)
    margin = max(1, int(0.04 * side))
    canvas = Image.new("L", (side + 2 * margin, side + 2 * margin), 0)
    canvas.paste(pil, ((canvas.width - w) // 2, (canvas.height - h) // 2))

    canvas = ImageOps.autocontrast(canvas)
    return np.asarray(canvas.resize((28, 28), _LANCZOS), dtype=np.uint8)
