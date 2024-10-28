from dataclasses import dataclass, field
from itertools import chain

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class Args:
    output_path: str = field(
        metadata={"help": "Where to save the preprocessed dataset."}
    )
    block_size: int = field(metadata={"help": "The block size to use for the dataset."})
    tokenizer_name: str = field(metadata={"help": "The name of the tokenizer to use."})
    num_proc: int = field(
        default=64, metadata={"help": "The number of processes to use."}
    )


def tokenize_function(examples, tokenizer, text_column_name):
    texts = []
    for text in examples[text_column_name]:
        while text.startswith("</s>"):
            text = text[len("</s>") :]
        texts.append(text)
    output = tokenizer(texts, add_special_tokens=False)
    return output


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]

    raw_dataset = load_dataset("data/redpajama_1b.py")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    preprocessed_datasets = []
    for split in raw_dataset.keys():
        column_names = raw_dataset[split].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenized_data = raw_dataset[split].map(
            lambda x: tokenize_function(x, tokenizer, text_column_name),
            batched=True,
            num_proc=args.num_proc,
            remove_columns=column_names,
            desc=f"Tokenizing {split} split",
        )

        # Add the preprocessed data to the preprocessed dataset train split
        preprocessed_datasets.append(
            tokenized_data.map(
                group_texts,
                batched=True,
                num_proc=args.num_proc,
                desc=f"Grouping texts in {split} split in chunks of {args.block_size}",
                fn_kwargs={"block_size": args.block_size},
            )
        )

    print(f"Saving preprocessed dataset to {args.output_path}")
    preprocessed_datasets = concatenate_datasets(preprocessed_datasets)
    preprocessed_datasets.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
