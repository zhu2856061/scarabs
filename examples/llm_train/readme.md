# 如何从0到1训练一个自己的LLM模型-以 LLama 模型 和 医疗数据 为例

## 1 前期准备

### 1.1 准备数据
从这个链接下载数据：https://github.com/Toyhom/Chinese-medical-dialogue-data/tree/master/Data_%E6%95%B0%E6%8D%AE/Pediatric_%E5%84%BF%E7%A7%91 获得 儿科5-14000.csv 文件

查看文件内容如下
```
department,title,ask,answer
营养保健科,小儿肥胖超重该如何治疗,女宝宝，刚7岁，这一年，察觉到，我家孩子身上肉很多，而且，食量非常的大，平时都不喜欢吃去玩，请问：小儿肥胖超重该如何治疗。,孩子出现肥胖症的情况。家长要通过孩子运功和健康的饮食来缓解他的症状，可以先让他做一些有氧运动，比如慢跑，爬坡，游泳等，并且饮食上孩子多吃黄瓜，胡萝卜，菠菜等，禁止孩子吃一些油炸食品和干果类食物，这些都是干热量高脂肪的食物，而且不要让孩子总是吃完就躺在床上不动，家长在治疗小儿肥胖期间如果孩子情况严重就要及时去医院在医生的指导下给孩子治疗。
营养保健科,小儿肥胖超重该怎样医治,男孩子，刚4岁，最近，发现，我家孩子体重要比别的孩子重很多，而且，最近越来越能吃了，还特别的懒，请问：小儿肥胖超重该怎样医治。,孩子一旦患上肥胖症家长要先通过运动和饮食来改变孩子的情况，要让孩子做一些他这个年龄段能做的运动，如游泳，慢跑等，要给孩子多吃一些像苹果，猕猴桃，胡萝卜等食物，禁止孩子吃高热量，高脂肪的食物，像蛋糕，干果，曲奇饼干等，严格的控制孩子的饮食，不要让他暴饮暴食，多运动对改变孩子肥胖都是有好处的，在治疗小儿肥胖期间如果情况严重，建议家长先带孩子去医院检查一下孩子肥胖症的原因在针对性的治疗。
```
需要将数据准备成 用于预训练的样式，样式如下

```
{"text": "总是拉稀吃油腻会拉肚子怎么回事，三四天前就已经开始肚子疼，然后拉稀，啊吃东西就不痛不拉，吃东西就已经开始痛已经开始拉，全是拉稀。请问总是拉稀吃油腻会拉肚>子怎么回事，你好通过上述病情叙述考虑由于消化不良引来的胃肠功能紊乱建议你清淡饮食，禁术辛辣油腻生凉刺激性食物，多喝热水，防止操劳受寒加班，在治疗期间患者需要做的饮>食有节，规律作息积极参加体育锻炼，调节身心健康，保持自身卫生，避免因抵抗力下降而导致细菌入侵。"}
{"text": "连续四五天拉稀伴随腹痛怎么回事，三四天前就已经开始肚子疼，然后拉稀，啊吃东西就不痛不拉，吃东西就已经开始痛已经开始拉，全是拉稀。请问连续四五天拉稀伴随腹>痛怎么回事，你好，根据你叙述的情况。如果恶心呕吐次数比较多，是需要有到医院仔细检查血常规及电解质。避免出现失水及电解质紊乱等情况。如果恶心呕吐不是特别严重，留意腹>部防寒。平时留意饮食，防止排便极冷辛辣等刺激性食物。在治疗期间患者需要做的饮食有节，规律作息积极参加体育锻炼，调节身心健康，保持自身卫生，避免因抵抗力下降而导致细>菌入侵。"}
```

打开jupyter notebook，在notebook中运行如下代码，即可完成数据处理

``` python
import json
import pandas as pd

ds = pd.read_csv("儿科5-14000.csv", encoding="utf-8")
lines = ds.to_dict(orient="records")
wfile = open("data.txt", "w")
for line in lines:
    new_line = line["title"] + "，" + line["ask"] + "，" + line["answer"]
    new_line = {"text": new_line}
    wfile.write(json.dumps(new_line, ensure_ascii=False) + "\n")
```

### 1.2 准备模型配置
由于选择llama模型，会采用 transformer 的架构中定义的模型结构，我们需要提供一个config.json 文件对模型进行初始化，该文件内容如下：
``` json
{
  "_name_or_path": "llama",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 2560,
  "max_position_embeddings": 1024,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 14,
  "num_hidden_layers": 12,
  "num_key_value_heads": 2,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.0",
  "use_cache": false,
  "vocab_size": 10000
}
```
config.json 文件中主要是模型的结构参数，重点的参数：
hidden_size，intermediate_size，max_position_embeddings，num_attention_heads，num_hidden_layers，vocab_size 这几个参数的设置将影响模型的参数量，越大越耗显存（若按上述数值设置，将是一个160M的llama模型）


## 2 生成tokenizer

这里我们采用BPE分词器，进行生成llama tokenizer

``` python
import os
import sys
import torch
from torchinfo import summary
from transformers import AutoConfig, HfArgumentParser, LlamaTokenizerFast

from scarabs.tokenizer_factory import TokenizerFactoryWithBPE
twsp = TokenizerFactoryWithBPE(
    unk_token="<unk>",
    vocab_size=10000,
    special_tokens=["<unk>", "<s>", "</s>"],
    TokenizerFastFunc=LlamaTokenizerFast,
    output_dir="./llama-160m",
)
data_files = "./data.txt"
twsp.create_tokenizer([data_files])
```

其中 unk_token="`<unk>`" 是必填项，vocab_size 是必填项 词表最大量，数据越多，该值应该设置越大，special_tokens 是特殊类的token值，用于填充，bos，eos，end， TokenizerFastFunc 是生成那种tokenizer，这里由于是llama，故选择LlamaTokenizerFast， output_dir 是保存路径

运行上述代码，即可生成tokenizer，生成tokenizer需要时间，请耐心等待。


## 3 开始训练
首先，我们需要准备好模型训练的配置文件 arguments.json，该文件内容如下：

``` json
{
    "do_train": true,
    "output_dir": "./encode/llm_train",
    "overwrite_output_dir": true,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0,
    "optim": "adamw_torch",
    "logging_dir": "./encode/llm_train",
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 1,
    "save_safetensors": true,
    "lr_scheduler_type": "constant_with_warmup",
    "report_to": "tensorboard",
    "label_names": [
        "labels"
    ],
    "metric_for_best_model": "loss",
    "greater_is_better": false,
    "use_cpu": false,
    "warmup_steps": 10,
    "seed": 2024,
    "remove_unused_columns": false,
    "gradient_checkpointing": true,
    "load_best_model_at_end": false,
    "early_stopping_patience": 20,
    "early_stopping_threshold": 1e-7,
    "model_name_or_path": "./llama-160m",
    "dataset_cache": "./encode/data",
    "overwrite_cache": false,
    "preprocessing_num_workers": 4,
    "max_seq_length": 1024,
    "train_file": "./data",
    "extension": "json"
}
```
相关参数说明如下：
train_file： 训练数据文件夹，里面应该为json格式，包含text字段，该字段为模型输入的文本，
extension: 表名我们将以 json格式读入数据
dataset_cache： 数据缓存路径，
model_name_or_path： 为刚生成的tokenizer文件夹
output_dir： 模型训练输出路径
logging_dir： 训练日志输出路径
其他相关参数可以参考 transformers 官方文档，这里不再赘述。


准备好上述文件后，接下来即可进行训练

文件结构如下
```
├── arguments.json
├── config.json
├── llama-160m
├── main.py
```
其中 main.py 为训练主程序
``` python
import os
import sys
import torch
from torchinfo import summary
from transformers import AutoConfig, HfArgumentParser, LlamaTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.task_factory import TaskFactoryWithPreTrain

# Params
parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, train_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1])
    )
else:
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

# define model
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "token": model_args.token,
    "trust_remote_code": model_args.trust_remote_code,
}
config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
model = LlamaForCausalLM(config)

# check model
input_ids = torch.randint(0, 10, (2, 512))
attention_mask = torch.ones_like(input_ids)
summary(
    model,
    input_data={"input_ids": input_ids, "attention_mask": attention_mask},
    depth=5,
)

# train
task = TaskFactoryWithPreTrain()
task.train(model_args, data_args, train_args, model=model, config=config)

```

启动训练
``` shell
torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py arguments.json
```
训练完成后，会在模型输出路径./encode/llm_train下生成一个文件夹，里面包含模型参数，日志等文件，

### 4 简单测试下模型的生成能力

构建预估代码
``` python

import os
import sys
import torch

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from scarabs.task_factory import TaskFactoryWithPreTrain
# predict
task = TaskFactoryWithPreTrain()
task.inference_with_load_model(
    "../data/llama-160m",
    "../data/llama-160m",
    LlamaForCausalLM,
)
res = task.inference("小儿肥胖超重该如何治疗", max_tokens=100)
print(res)
```

到此，我们完成了从0到1训练出来一个基础模型，但是可以看到，还不能进行问答，因此，我们还需要在这个基座模型的基础上进行 sft 微调，而且是对话方式的微调，故接下来将介绍如何进行对话微调。


