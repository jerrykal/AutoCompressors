import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser

from args import DataTrainingArguments, ModelArguments, TrainingArguments
from data import DataCollatorWithDynamicPadding
from longbench_utils import qa_f1_score
from utils import FileLogger, get_model_and_tokenizer, makedirs

logger = logging.getLogger(__name__)


@dataclass
class EvalArgs(TrainingArguments):
    num_topics_list: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50, 60, 70],
        metadata={"help": "How many topics to in the conversation?"},
    )
    target_topic: str = field(
        default="first", metadata={"help": "Which topic to evaluate?"}
    )
    segment_length: int = field(
        default=1024,
        metadata={"help": "The length of each segment to compress."},
    )

    do_sample: bool = False
    max_new_tokens: int = 50


def process_topic_retrieval(data, tokenizer, num_topics_list, target_topic):
    outputs = {
        "input_ids": [],
        "attention_mask": [],
        "target": [],
        "length": [],
        "num": [],
    }

    for context, question, topics, num in zip(
        data["context"], data["question"], data["topics"], data["num_topics"]
    ):
        # filter out samples that do not have proper number of topics/lines
        if num not in num_topics_list:
            continue

        if num == 1:
            context = context.split(
                " \n USER: Great, this is the end of our discussion"
            )[0]
            context = context + " Now the record ends."

        if target_topic == "first":
            question = "What is the first topic we have discussed? Only give me the topic name. Do not summarize yourself."
            target = topics[0]
        elif target_topic == "random":
            target_idx = np.random.randint(0, num)
            question = f"What is the No.{target_idx} topic we have discussed? Only give me the topic name. Do not summarize yourself."
            target = topics[target_idx]
        else:
            raise NotImplementedError

        prompt = " ".join([context, question])
        # the question always asks for the first topic

        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
        )

        encoded["target"] = target
        encoded["length"] = len(encoded.input_ids)
        encoded["num"] = num

        for k, v in encoded.items():
            if k in outputs:
                outputs[k].append(v)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([ModelArguments, DataTrainingArguments, EvalArgs])
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_args,
        torch_dtype=(
            torch.bfloat16
            if eval_args.bf16
            else (torch.float16 if eval_args.fp16 else torch.float32)
        ),
    )

    # Load data
    with accelerator.main_process_first():
        dataset = load_dataset(
            "json",
            data_files=data_args.eval_data,
            cache_dir=data_args.dataset_cache_dir,
            split="train",
        )
        dataset = dataset.map(
            process_topic_retrieval,
            batched=True,
            num_proc=32,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "num_topics_list": eval_args.num_topics_list,
                "target_topic": eval_args.target_topic,
            },
        )

        # group instances of the same number of topics together, so that their lengths are approximately equal
        groupby_dataset = dataset.to_pandas().groupby("num")

    data_collator = DataCollatorWithDynamicPadding(
        pad_token_id=tokenizer.pad_token_id, padding_side="left"
    )

    accuracy = {}
    f1_score = {}
    results = defaultdict(list)
    for num, dataset in groupby_dataset:
        dataset = Dataset.from_pandas(
            groupby_dataset.get_group(num), preserve_index=False
        )
        all_targets = dataset["target"]
        # remove unnecessary columns
        dataset = dataset.remove_columns(["target", "num"])

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=data_collator,
            # only pin memory when no gpu
            pin_memory=not eval_args.use_cpu,
        )

        # NOTE: prepare model only once
        if len(accelerator._models) == 0:
            model, dataloader = accelerator.prepare(model, dataloader)
            model = accelerator.unwrap_model(model)
        else:
            dataloader = accelerator.prepare(dataloader)

        all_lengths = []
        all_outputs = []
        for _, x in enumerate(tqdm(dataloader, desc=f"Evaluating {num} Topics")):
            length = x.pop("length")

            num_segments = length // eval_args.segment_length

            tokens_to_compress = x["input_ids"][
                :, : num_segments * eval_args.segment_length
            ]
            raw_tokens = x["input_ids"][:, num_segments * eval_args.segment_length :]
            summary_vectors = model(
                tokens_to_compress,
                output_softprompt=True,
                segment_lengths=[eval_args.segment_length] * num_segments,
            ).softprompt

            outputs = model.generate(
                raw_tokens,
                softprompt=summary_vectors,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
            )

            start_idx = raw_tokens.size(1)
            outputs = outputs[:, start_idx:]

            if accelerator.num_processes > 1:
                outputs = accelerator.pad_across_processes(
                    outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1
                )
                outputs = accelerator.gather_for_metrics(outputs)
                length = accelerator.gather_for_metrics(length)

            outputs = outputs.tolist()
            length = length.tolist()

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(outputs)
            all_lengths.extend(length)

        length = int(sum(all_lengths) / len(all_lengths))

        acc = 0
        f1 = 0
        for output, target in zip(all_outputs, all_targets):
            if target.lower() in output.lower():
                acc += 1
            else:
                acc += 0
            f1 += qa_f1_score(output, target)
            results[length].append({"target": target, "prediction": output})

        acc /= len(all_outputs)
        f1 /= len(all_outputs)

        accuracy[length] = acc
        f1_score[length] = round(f1, 4)

    if accelerator.process_index == 0:
        with open(
            makedirs(os.path.join(eval_args.output_dir, "results.json")),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f)
        # also save config
        with open(
            os.path.join(eval_args.output_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(eval_args.to_dict(), f)

        file_logger = FileLogger(
            makedirs(os.path.join(eval_args.output_dir, "metrics.log"))
        )
        file_logger.log({"accuracy": accuracy, "f1": f1_score}, Args=asdict(eval_args))


if __name__ == "__main__":
    main()
