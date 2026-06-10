from mmengine import MODELS as MMENGINE_MODELS
from mmengine import TASK_UTILS as MMENGINE_TASK_UTILS

from mmengine import Registry

MODELS = Registry('model',
                  parent=MMENGINE_MODELS,
                  locations=['llava.model.multimodal_encoder.embodiedscan.models'])
TASK_UTILS = Registry('task util',
                      parent=MMENGINE_TASK_UTILS,
                      locations=['llava.model.multimodal_encoder.embodiedscan.models'])
