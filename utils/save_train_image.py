from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 设置事件文件的路径
event_file = 'path/to/your/event/file'

# 创建一个事件累加器对象并加载数据
event_acc = EventAccumulator(event_file, size_guidance={'images': 0})
event_acc.Reload()

# 获取所有图像的标签
image_tags = event_acc.Tags()['images']

# 对于每个标签，提取所有图像并保存
for tag in image_tags:
    images = event_acc.Images(tag)
    for index, image in enumerate(images):
        img = np.frombuffer(image.encoded_image_string, dtype=np.uint8)
        img = Image.open(io.BytesIO(img))
        img.save(f'{tag}_{index}.png')
