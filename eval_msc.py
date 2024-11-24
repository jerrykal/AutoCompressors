# Based on: https://github.com/namespace-Pt/UltraGist/blob/main/main/eval_msc.py

import json
import logging
import os
from dataclasses import asdict, dataclass, field

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, PreTrainedTokenizer

# Based on: https://github.com/namespace-Pt/UltraGist/blob/main/main/eval_msc.py
from args import DataTrainingArguments, ModelArguments, TrainingArguments
from chat_utils import apply_chat_template
from data import DataCollatorWithDynamicPadding
from utils import FileLogger, get_model_and_tokenizer, makedirs, normalize_text

logger = logging.getLogger(__name__)


@dataclass
class EvalArgs(TrainingArguments):
    segment_length: int = field(
        default=1024,
        metadata={"help": "The length of each segment to compress."},
    )


def process_msc(data: Dataset, tokenizer: PreTrainedTokenizer, max_length: int):
    outputs = {"input_ids": [], "attention_mask": [], "target": []}

    for context, input_, output in zip(data["context"], data["input"], data["output"]):
        prompt = context + "\n" + input_

        if max_length is not None:
            prompt = tokenizer.decode(
                tokenizer.encode(prompt, add_special_tokens=False)[-max_length:]
            )

        encoded = apply_chat_template(
            template="no",
            messages=[{"role": "user", "content": prompt}],
            tokenizer=tokenizer,
            add_generation_prompt=True,
        ).encoded
        encoded["target"] = output

        for k, v in encoded.items():
            outputs[k].append(v)
    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([ModelArguments, DataTrainingArguments, EvalArgs])
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(cpu=eval_args.use_cpu)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    model, tokenizer = get_model_and_tokenizer(
        model_args=model_args,
        torch_dtype=(
            torch.bfloat16
            if eval_args.bf16
            else (torch.float16 if eval_args.fp16 else torch.float32)
        ),
    )
    model.eval()

    # Load data
    with accelerator.main_process_first():
        dataset = load_dataset(
            "json",
            data_files=data_args.eval_data,
            cache_dir=data_args.dataset_cache_dir,
            split="train",
        )
        dataset = dataset.map(
            process_msc,
            batched=True,
            num_proc=32,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": None,
            },
        )

    results = []

    all_targets = dataset["target"]
    dataset = dataset.remove_columns(["target"])
    data_collator = DataCollatorWithDynamicPadding(
        pad_token_id=tokenizer.pad_token_id, padding_side="left"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=data_collator,
        pin_memory=not eval_args.use_cpu,
    )

    if len(accelerator._models) == 0:
        model, dataloader = accelerator.prepare(model, dataloader)
        model = accelerator.unwrap_model(model)
    else:
        dataloader = accelerator.prepare(dataloader)

    all_outputs = []
    for x in tqdm(dataloader):
        num_segments = x["input_ids"].size(1) // eval_args.segment_length

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
            max_new_tokens=20,
            do_sample=False,
            top_p=None,
            temperature=None,
        )

        start_idx = raw_tokens.size(1)
        outputs = outputs[:, start_idx:]

        if accelerator.num_processes > 1:
            outputs = accelerator.pad_across_processes(
                outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1
            )
            outputs = accelerator.gather_for_metrics(outputs)

        outputs = outputs.tolist()

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(outputs)

    if accelerator.process_index == 0:
        rouge = Rouge()
        score = rouge.get_scores(
            normalize_text(all_outputs),
            normalize_text(all_targets),
            avg=True,
            ignore_empty=True,
        )["rouge-l"]["r"]

        for output, target in zip(all_outputs, all_targets):
            results.append({"target": target, "prediction": output})

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
        file_logger.log({"rouge": score}, Args=asdict(eval_args))


if __name__ == "__main__":
    main()
