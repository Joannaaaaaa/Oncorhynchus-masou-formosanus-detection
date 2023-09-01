import os.path
import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    #    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    #    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    #    cv2.putText(
    #        img,
    #        text=class_name,
    #        org=(x_min, y_min - int(0.3 * text_height)),
    #        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #        fontScale=0.35,
    #        color=TEXT_COLOR,
    #        lineType=cv2.LINE_AA,
    #    )
    return img

def visualize(image, bboxes, category_ids):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        #        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


# 讀取images、labels名稱->str
img_path = "data_n/images"
lab_path = "data_n/labels"

img_names = os.listdir(img_path)
lab_names = os.listdir(lab_path)
x = []
# print(img_name, lab_name)
# for i in range(len(img_name)):
#     # if img_name[i][0:5] == lab_name[i][0:5]:
#         # continue
#     print(img_name[i])
#     print(lab_name[i])
for i in range(len(img_names)):

    img_name = img_names[i]  # images name
    lab_name = lab_names[i]  # labels name

    image = cv2.imread(img_path + '/' + img_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(lab_name.split('.')[0])
    # print(img_name.split('.')[0])
    print(i)
    lab_file = lab_path + '/' + lab_name
    tree = ET.parse(lab_file)
    root = tree.getroot()  # get root object

    width = int(root.find("size")[0].text)
    height = int(root.find("size")[1].text)
    channels = int(root.find("size")[2].text)

    bboxes = []
    for Object in root.findall('object'):
        bndbox = Object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # store data in list
        bboxes.append([xmin, ymin, xmax, ymax])

    category_ids = [[0]]
    for d in range(len(bboxes) - 1):
        category_ids.append([0])

    #  visualize(image, bboxes, category_ids)

    transform = A.Compose([
        # A.Resize(width=768, height=512),
        A.SafeRotate(p=0.8, limit=(-45, 45), border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.6),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.1),
        A.RandomBrightnessContrast(p=0.25)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    )

    new_bboxes = []

    for j in range(5):
        trans = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        new_bboxes.append(trans['bboxes'])

        #
        n_img = "data_n/n_img/" + img_name.split('.')[0] + "_" + str(j) + ".jpg"
        cv2.imwrite(n_img, trans['image'])

        # create pascal voc writer (image_path, width, height)
        writer = Writer(n_img, width, height)

        for k in range(len(new_bboxes[j])):
            # add objects (class, xmin, ymin, xmax, ymax)
            writer.addObject('fish', int(new_bboxes[j][k][0]), int(new_bboxes[j][k][1]), int(new_bboxes[j][k][2]),
                             int(new_bboxes[j][k][3]))

        # write to file
        writer.save("data_n/n_lab/" + lab_name.split('.')[0] + "_" + str(j) + ".xml")
