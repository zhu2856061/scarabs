## scarabs: a universal training framework

#### todo
use visdom 

#### core:
  - Training of tabular data, For example, CTR used in recommendation systems
  - Training of text data, For example, text classification
  - Training of image data, For example, image classification
  - Training of LLM, For example, llm pretrain

#### very easy to use
``` shell
pip install scarabs
```

#### In detail

1. Tabular Data
You can refer to tabular_ctr in the examples folder

2. Text Data
You can refer to llm_classification in the examples folder

3. LLM
You can refer to llm_pretrain in the examples folder

4. refer to github https://github.com/zhu2856061/scarabs



#### update
[2025-04-25] 新增tabular ctr 的增量训练功能
如何进行增量训练
1. 已 example/tabular_ctr/fm 为例

- 先正常训练一版模型 main.py 文件中加入如下代码
```python
import os
from transformers import HfArgumentParser
from scarabs.args_factory import (
    DataArguments,
    ModelArguments,
    TaskArguments,
    TrainArguments,
)
from scarabs.mora.models.ctr_with_fm import CtrWithFM, CtrWithFMConfig
from scarabs.task_factory import TaskFactoryWithTabularCtr
parser = HfArgumentParser(
    (TaskArguments, DataArguments, ModelArguments, TrainArguments)  # type: ignore
)
task_args, data_args, model_args, training_args = parser.parse_json_file(
    "arguments.json"
)
# # 基于config.json文件的特征配制生产特征映射信息
config = CtrWithFMConfig.from_pretrained("config.json")
task = TaskFactoryWithTabularCtr(task_args, data_args, None, None, config)
task.create_feature2transformer_and_config()

# # 加上新的特征映射信息，并进行模型训练
config = CtrWithFMConfig.from_pretrained(
    os.path.join(
        task_args.task_name_or_path, data_args.dataset_cache, "meta/config.json"
    )
)
task = TaskFactoryWithTabularCtr(
    task_args, data_args, model_args, training_args, config
)
task.train(model=CtrWithFM(config))
```

- 再基于训练好的一版模型进行增量训练

-- 修改arguments.json文件，加入 "incremental_resume_from_checkpoint": "./encode/model/checkpoint-xxx", 其中为历史训练好的模型checkpoint路径

-- 修改main.py 文件，对其中的部分代码进行修改，如下：

```python
# # 基于config.json文件的特征配制生产特征映射信息
config = CtrWithFMConfig.from_pretrained("./encode/model/checkpoint-xxx/config.json")
task = TaskFactoryWithTabularCtr(task_args, data_args, None, None, config)
task.create_feature2transformer_and_config()

# # 加上新的特征映射信息，并进行模型训练
config = CtrWithFMConfig.from_pretrained(
    os.path.join(
        task_args.task_name_or_path, data_args.dataset_cache, "meta/config.json"
    )
)
task = TaskFactoryWithTabularCtr(
    task_args, data_args, model_args, training_args, config
)
task.train(model=CtrWithFM(config))
```
即可进行增量训练，在训练的日志部分会有增量模型部分矩阵改变的日志打印，请留意

```
logger.warning(f"{v} shape mismatched, current: {model_dict[v].shape} != history:{state_dict[k].shape}")
```
给出当前模型矩阵和历史模型矩阵的形状不一致，请留意

```
logger.warning(f"{key} is updated from history:{history_size} to current:{current_size}")
```
给出历史模型矩阵已经修正成新的矩阵大小