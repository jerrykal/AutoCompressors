import json
import os
import pathlib
import re
import string
import sys
from datetime import datetime

import numpy as np
import pytz
import torch
from transformers import AutoTokenizer

from args import ModelArguments


def get_model_and_tokenizer(
    model_args: ModelArguments,
    torch_dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if "llama" in (model_args.model_name_or_path or model_args.config_name).lower():
        from auto_compressor import LlamaAutoCompressorModel

        AutoCompressorModel = LlamaAutoCompressorModel
    else:
        from auto_compressor import AutoCompressorModel

    model = AutoCompressorModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer


def get_last_checkpoint_or_last_model(folder):
    """modification of get_last_checkpoint from transformer.trainer_utils.
    This function will return the main folder if it contains files of the form "pytorch_model*". The default HF function ignores those and only looks
    for "checkpoint-*" folders."""
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    _re_model = re.compile("pytorch_model" + r"*")
    content = os.listdir(folder)
    models = [path for path in content if _re_model.search(path) is not None]
    if models != []:
        return folder
    else:
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(folder, path))
        ]
        if len(checkpoints) == 0:
            return
        return os.path.join(
            folder,
            max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
        )


def parse_checkpoint_step(checkpoint):
    if checkpoint.split("-")[0] != "checkpoint":
        return -1
    else:
        try:
            return int(checkpoint.split("-")[-1])
        except:
            print(f"got checkpoint name {checkpoint}, couldn't parse step")
            return -1


def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(
    text,
    ignore_case=True,
    ignore_punctuation=True,
    ignore_space=True,
    ignore_number=False,
):
    if isinstance(text, str):
        text = [text]
        unpack = True
    else:
        unpack = False
    if ignore_case:
        text = np.char.lower(text)
    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        text = np.char.translate(text, table=repl_table)
    if ignore_number:
        repl_table = string.digits.maketrans("", "", string.digits)
        text = np.char.translate(text, table=repl_table)
    if ignore_space:
        for i, words in enumerate(np.char.split(text)):
            text[i] = " ".join(words)
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if unpack:
        text = text[0]
    return text


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file

    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone("Asia/Taipei")
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")
