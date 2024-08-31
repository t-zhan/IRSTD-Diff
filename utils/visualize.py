from typing import List, Optional, Union
from pathlib import Path

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

# Set directories
data_dir = Path("./NUAA-SIRST")
image_dir = data_dir.joinpath("images")
xml_dir = data_dir.joinpath("bboxes")
viz_dir = Path("viz")
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)


# Get the bounding box from xml file: multiple bbox is NOT supported right now !!!
def get_bbox(xml_path: str):
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    bbox = {child.tag: int(child.text) for child in root.find("object").find("bndbox")}
    left_top = (bbox["xmin"], bbox["ymin"])
    right_bottom = (bbox["xmax"], bbox["ymax"])
    return left_top, right_bottom

# Show bounding box data
def imshow_bboxed(
    img: Union[str, np.ndarray],
    bboxes: Union[list, np.ndarray],
    colors = (255, 0 ,255),
    thickness: int = 1,
    show: bool = False,
    win_name: str = '',
    wait_time: int = 0,
    out_file: Optional[str] = None
):
    img = cv2.imread(img)
    img = np.ascontiguousarray(img)
    
    # Multiple bbox is NOT supported right now !!!
    left_top, right_bottom = bboxes
    cv2.rectangle(
        img, left_top, right_bottom, color = colors, thickness = thickness)
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)

# Randomly select a figure for testing
# idx = np.random.randint(1, 428)
height = []
length = []
diag = []
a = 0



# for idx in sorted(os.listdir(image_dir)):
    
    # idx = int(idx.split(".")[0].split("_")[1])
    # img_path = image_dir.joinpath(f"Misc_{idx}.png")
    # xml_path = xml_dir.joinpath(f"Misc_{idx}.xml")
with open("NUAA-SIRST/idx_427/trainvaltest.txt", "r") as f:
    for misc_idx in f.readlines():
        misc_idx = misc_idx.strip('\n') 
        #  = f.read().strip('\n')
        img_path = image_dir.joinpath(f"{misc_idx}.png")
        xml_path = xml_dir.joinpath(f"{misc_idx}.xml")
        bbox = get_bbox(xml_path)
        print(misc_idx, bbox)
        length.append(bbox[1][0] - bbox[0][0])
        height.append(bbox[1][1] - bbox[0][1])
        diag.append(((bbox[1][0] - bbox[0][0])**2 + (bbox[1][1] - bbox[0][1])**2)**0.5)
        # if diag[-1] > 50:
        #     # print(idx)
        #     imshow_bboxed(img_path, bbox, out_file=viz_dir.joinpath(f"{misc_idx}.png"))
        #     a += 1
        imshow_bboxed(img_path, bbox, out_file=viz_dir.joinpath(f"{misc_idx}.png"))
# print(a)
# plt.hist(np.array(diag), bins=200)
# plt.savefig('height.png')
# plt.show()