import logging
import os
import random
import re
from functools import partial
from typing import Any, Mapping

import datasets
import torch

from chat_utils import apply_chat_template

logger = logging.getLogger(__name__)


def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(
        inputs["input_ids"], list
    ), "Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "attention_mask":
                v = v + [1]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k == "token_type_ids":
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs


class Data:
    def _process_pretrain_data(data, indices):
        outputs = {"labels": [], "index": [], "length": []}
        for input_ids, labels, index in zip(data["input_ids"], data["labels"], indices):
            outputs["index"].append(index)
            outputs["length"].append(len(input_ids))
            # NOTE: the labels will be automatically generated in Trainer._prepare_inputs
            print(labels)
            while True:
                pass
            outputs["labels"].append(labels)
        return outputs

    def _process_language_modeling(data, indices, tokenizer, min_length, max_length):
        outputs = {"input_ids": [], "labels": [], "length": [], "index": []}

        for i, text in enumerate(data["text"]):
            # truncate text for faster processing
            encoded = tokenizer(text)
            if len(encoded["input_ids"]) < min_length:
                continue
            elif len(encoded["input_ids"]) < max_length:
                encoded = add_eos(encoded, tokenizer.eos_token_id)
            else:
                for k, v in encoded.items():
                    encoded[k] = v[:max_length]

            # NOTE: the labels will be automatically generated in Trainer._prepare_inputs
            encoded["labels"] = encoded["input_ids"].copy()

            for k, v in encoded.items():
                if k in outputs:
                    outputs[k].append(v)
            # length is required for grouping
            outputs["length"].append(len(encoded["input_ids"]))
            outputs["index"].append(indices[i])

        return outputs

    def _process_instruction_tuning(
        data, indices, tokenizer, chat_template, min_length, max_length, eval_mode=False
    ):
        outputs = {"input_ids": [], "labels": [], "length": [], "index": []}

        for i, source in enumerate(data["conversations"]):
            if source[0]["role"] != "user":
                # Skip the first one if it is not from user
                source = source[1:]

            # NOTE: in evaluation, we only use the first turn in the conversation
            if eval_mode:
                # a string (the expected output from the assistant)
                if len(source) > 1:
                    labels = source[1]["content"]
                else:
                    labels = None
                source = source[:1]

            encoded = apply_chat_template(
                chat_template,
                source,
                tokenizer=tokenizer,
                # only return labels in evaluation mode
                return_labels=not eval_mode,
                add_generation_prompt=eval_mode,
            ).encoded

            # NOTE: shift the labels in advance
            # labels = encoded["labels"][1:]
            # labels.append(-100)
            # encoded["labels"] = labels

            # skip data that not fall in between min_length and max_length
            if min_length is not None and len(encoded["input_ids"]) < min_length:
                continue
            if max_length is not None and len(encoded["input_ids"]) > max_length:
                continue

            if eval_mode:
                encoded["labels"] = labels

            for k, v in encoded.items():
                if k in outputs:
                    outputs[k].append(v)
            outputs["length"].append(len(encoded["input_ids"]))
            outputs["index"].append(indices[i])

        return outputs

    def prepare_train_data(
        data_files=None,
        tokenizer=None,
        max_length=4096,
        min_length=512,
        chat_template="vicuna",
        seed=42,
        cache_dir=None,
        load_from_cache_file=None,
        ignore_index=False,
        ignore_length=False,
    ):
        if data_files is None:
            return None

        if isinstance(data_files, list):
            logger.info(f"Loading training data from {data_files}...")
        elif isinstance(data_files, str):
            logger.info(f"Loading training data from {data_files}...")
            data_files = [data_files]
        else:
            raise ValueError(f"Invalid training data {data_files}!")

        data_2_num_sample = {}
        for data_file in data_files:
            match = re.search("\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub("\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            data_2_num_sample[data_file] = max_sample_num

        random.seed(seed)

        train_datasets = []
        for data_file, max_sample_num in data_2_num_sample.items():
            if os.path.isdir(data_file) and os.path.exists(
                os.path.join(data_file, "dataset_info.json")
            ):
                # the dataset may be save_to_disk in advance
                dataset = datasets.load_from_disk(data_file)
                dataset = dataset.map(
                    Data._process_pretrain_data,
                    batched=True,
                    num_proc=32,
                    batch_size=32,
                    with_indices=True,
                )

            else:
                # the dataset is a json file
                dataset = datasets.load_dataset(
                    "json", data_files=data_file, split="train", cache_dir=cache_dir
                )

                column_names = dataset.column_names
                if "text" in column_names:
                    process_fn = partial(
                        Data._process_language_modeling,
                        tokenizer=tokenizer,
                        min_length=min_length,
                        max_length=max_length,
                    )
                elif "conversations" in column_names:
                    process_fn = partial(
                        Data._process_instruction_tuning,
                        tokenizer=tokenizer,
                        chat_template=chat_template,
                        min_length=min_length,
                        max_length=max_length,
                    )
                else:
                    raise ValueError(
                        "Found neither 'text' nor 'conversations' in the training data!"
                    )

                dataset = dataset.map(
                    process_fn,
                    batched=True,
                    num_proc=32,
                    remove_columns=dataset.column_names,
                    batch_size=32,
                    with_indices=True,
                    load_from_cache_file=load_from_cache_file,
                )

            if max_sample_num is not None and len(dataset) > max_sample_num:
                dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

            # index column is useless in training
            if "index" in dataset.column_names and ignore_index:
                dataset = dataset.remove_columns(["index"])
            if "length" in dataset.column_names and ignore_length:
                dataset = dataset.remove_columns(["length"])

            # NOTE: Temporary remove attention_mask from dataset, it will cause error when not removed,
            #       but since the dataset preprocessing takes a long time, we will manually remove it here during instruction finetuning
            if "attention_mask" in dataset.column_names:
                dataset = dataset.remove_columns(["attention_mask"])

            train_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(train_datasets)

        return dataset

    def prepare_eval_data(
        data_files=None,
        tokenizer=None,
        max_length=4096,
        min_length=512,
        chat_template="vicuna",
        max_eval_num=None,
        cache_dir=None,
        seed=42,
        load_from_cache_file=None,
        ignore_index=False,
        ignore_length=False,
    ):
        if data_files is None:
            return None

        random.seed(seed)

        if max_eval_num is not None:
            dataset = datasets.load_dataset(
                "json",
                data_files=data_files,
                split=f"train[:{max_eval_num}]",
                cache_dir=cache_dir,
            )
        else:
            dataset = datasets.load_dataset(
                "json", data_files=data_files, split="train", cache_dir=cache_dir
            )

        column_names = dataset.column_names
        if "text" in column_names:
            process_fn = partial(
                Data._process_language_modeling,
                tokenizer=tokenizer,
                min_length=min_length,
                max_length=max_length,
            )
        elif "conversations" in column_names:
            process_fn = partial(
                Data._process_instruction_tuning,
                tokenizer=tokenizer,
                chat_template=chat_template,
                min_length=min_length,
                max_length=max_length,
                eval_mode=True,
            )
        else:
            raise ValueError(
                "Found neither 'text' nor 'conversations' in the training data!"
            )

        dataset = dataset.map(
            process_fn,
            batched=True,
            num_proc=32,
            remove_columns=dataset.column_names,
            with_indices=True,
            load_from_cache_file=load_from_cache_file,
        )
        if "index" in dataset.column_names and ignore_index:
            dataset = dataset.remove_columns(["index"])
        if "length" in dataset.column_names and ignore_length:
            dataset = dataset.remove_columns(["length"])

        return dataset


class DataCollatorWithDynamicPadding:
    def __init__(self, pad_token_id: int, padding_side: str) -> None:
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.keys_to_tensorize = {
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "token_type_ids",
            "length",
            "depth",
            "index",
        }

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, list[int]]:
        first_example = examples[0]
        return_batch = {}

        for key in first_example.keys():
            if "attention_mask" in key:
                pad_token_id = 0
            elif "label" in key:
                pad_token_id = -100
            else:
                pad_token_id = self.pad_token_id

            batch_values = [example[key] for example in examples]
            if isinstance(batch_values[0], list):
                return_batch[key] = self.add_padding(batch_values, pad_token_id)
            else:
                return_batch[key] = batch_values

            if key in self.keys_to_tensorize:
                return_batch[key] = torch.tensor(return_batch[key])

        return return_batch

    def add_padding(
        self,
        sequences: list[torch.LongTensor],
        fill_value: int,
    ) -> list:
        max_len = max([len(seq) for seq in sequences])
        if self.padding_side == "right":
            return [seq + [fill_value] * (max_len - len(seq)) for seq in sequences]
        else:
            return [[fill_value] * (max_len - len(seq)) + seq for seq in sequences]
