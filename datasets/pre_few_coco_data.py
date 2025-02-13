import json
from pycocotools.coco import COCO
import random

def dataset_sub_sumple(dataType='train'):
    # 设置路径
    dataDir = '/Users/wangfengguo/LocalTools/data/DUO.v4i.coco'  # COCO数据集路径
    annFile = f'{dataDir}/annotations/instances_{dataType}.json'  # 标注文件路径
    output_annFile = f'{dataDir}/annotations/instances_{dataType}_subset.json'  # 输出标注文件路径
    # 初始化COCO API
    coco = COCO(annFile)

    # 获取所有类别
    catIds = coco.getCatIds()
    categories = coco.loadCats(catIds)
    print(f"Total categories: {categories}")

    # 设置每个类别需要的目标数量
    target_anns_per_category = 100  # 每个类别筛选100个目标

    # 初始化新的annotation数据结构
    new_annotations = {
        "info": coco.dataset['info'],
        "licenses": coco.dataset['licenses'],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # 用于记录已选中的图像ID，避免重复
    selected_image_ids = set()

    # 遍历每个类别，筛选目标
    for cat in categories:
        cat_id = cat['id']
        print(f"Processing category: {cat['name']} (ID: {cat_id})")

        # 获取该类别的所有annotations
        annIds = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(annIds)

        # 随机筛选指定数量的annotations
        if len(anns) > target_anns_per_category:
            anns = random.sample(anns, target_anns_per_category)

        # 将筛选的annotations添加到新的annotations列表中
        for ann in anns:
            new_annotations['annotations'].append(ann)

            # 如果对应的图像尚未添加到新的images列表中，则添加
            if ann['image_id'] not in selected_image_ids:
                img_info = coco.loadImgs(ann['image_id'])[0]
                new_annotations['images'].append(img_info)
                selected_image_ids.add(ann['image_id'])

    # 保存新的annotation文件
    with open(output_annFile, 'w') as f:
        json.dump(new_annotations, f)

    print(f"New annotation file saved to {output_annFile}")
    print(f"Total images in new annotation: {len(new_annotations['images'])}")
    print(f"Total annotations in new annotation: {len(new_annotations['annotations'])}")


if __name__ == '__main__':
    dataset_sub_sumple()