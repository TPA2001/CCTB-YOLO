import warnings

warnings.filterwarnings("ignore")
import os

from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if __name__ == "__main__":
    # model.load('yolo11n.pt')
    model = YOLO(model=r"ultralytics/cfg/models/CCTB-YOLO/CCTB.yaml")
    model.model.info()
    model.train(
        data=r"data.yaml",
        imgsz=1280,
        epochs=500,
        batch=48,
        workers=8,
        device=[2, 3],
        optimizer="SGD",
        close_mosaic=100,
        resume=False,
        project="runs/CCTB-YOLO",
        name="exp",
        single_cls=True,
        cache=False,
    )
