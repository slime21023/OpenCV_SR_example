import cv2
from typing import Literal
from pathlib import Path
from src import ml_model, assets

_ml_model_dir = Path(ml_model.__file__).parent

_path = {
    "fsrcnn": _ml_model_dir / "FSRCNN_x3.pb",
    "espcn": _ml_model_dir / "ESPCN_x3.pb",
    "edsr": _ml_model_dir / "EDSR_x4.pb",
    "lapsrn": _ml_model_dir / "LapSRN_x2.pb",
}

_sr = cv2.dnn_superres.DnnSuperResImpl_create()


def useModel(
    model: Literal["fsrcnn", "espcn", "edsr", "lapsrn"], upsampling_ratio: int = 2
) -> cv2.dnn_superres.DnnSuperResImpl:
    model_path = f"{_path.get(model)}"
    _sr.readModel(model_path)
    _sr.setModel(model, upsampling_ratio)
    return _sr


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # type: ignore

    assets_dir = Path(assets.__file__).parent
    image = cv2.imread(assets_dir / "image.png", cv2.COLOR_BGR2RGB)
    print(image.shape, image.dtype)
    sr = useModel("espcn", upsampling_ratio=3)

    result = sr.upsample(image)

    # Resized image
    resized = cv2.resize(image, dsize=None, fx=3, fy=3)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    # Original image
    plt.imshow(image[:, :])
    plt.subplot(1, 3, 2)
    # SR upscaled
    plt.imshow(result[:, :])
    plt.subplot(1, 3, 3)
    # OpenCV upscaled
    plt.imshow(resized[:, :])
    plt.show()
