import torch, detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from cv_utils import plot_show
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()


def print_version():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)


def test_object_detection():
    img_path = './input.jpg'
    # OpenCV 的默认通道顺序：OpenCV 读取的图像是 BGR 格式（蓝 - 绿 - 红）。
    # Matplotlib 的默认通道顺序：Matplotlib 显示的图像是 RGB
    # 格式（红 - 绿 - 蓝）。
    im = cv2.imread(img_path)
    im = im[:, :, ::-1]  # 反转通道顺序
    plot_show(im)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image())
    plt.show()
    plt.axis('off')


# Total categories: [{'id': 0, 'name': 'Fish-', 'supercategory': 'none'}, {'id': 1, 'name': 'echinus', 'supercategory': 'Fish-'},
#                    {'id': 2, 'name': 'holothurian', 'supercategory': 'Fish-'}, {'id': 3, 'name': 'scallop', 'supercategory': 'Fish-'},
#                    {'id': 4, 'name': 'starfish', 'supercategory': 'Fish-'}]

classes = ["echinus", "holothurian", "scallop", "starfish"]

def train_object_detection():

    # 1. 构建数据集
    data_root_dir = "/Users/wangfengguo/LocalTools/data/DUO.v4i.coco"
    img_dir = os.path.join(data_root_dir, "images")
    annot_dir = os.path.join(data_root_dir, "annotations")
    # if your dataset is in COCO format, this cell can be replaced by the following three lines:
    # from detectron2.data.datasets import register_coco_instances
    for t in ["train", "val", "test"]:
        register_coco_instances(f"my_dataset_{t}", {}, os.path.join(annot_dir, f"instances_{t}_subset.json"), img_dir)
        # MetadataCatalog.get(f"my_dataset_{t}").set(thing_classes=classes)

    # 2. 开始训练
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train")
    cfg.DATASETS.TEST = ("my_dataset_val")
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def test_show_dataset(dataset_name):
    train_metadata = MetadataCatalog.get(dataset_name)
    train_dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(train_dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plot_show(out.get_image())


if __name__ == '__main__':
    train_object_detection()




