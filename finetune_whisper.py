from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

from transformers import WhisperProcessor
import torchaudio
dataset = load_dataset(
    "csv",
    data_files="/home4/khanhnd/self-condition/data/valset_bilingual.tsv",
    split="train",
    cache_dir="cache",
    sep="\t",
)
eval_dataset=load_dataset(
    "csv",
    data_files="/home4/khanhnd/self-condition/data/valset_bilingual.tsv",
    split="train",
    cache_dir="cache",
    sep="\t",
)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")
input_str = dataset[0]["transcript"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
print(dataset[0])
import numpy as np
def prepare_dataset(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_filepath"])
    assert sampling_rate==16000
    batch["audio"] = speech_array[0]
    # load and resample audio data from 48 to 16kHz
    batch.pop("audio_filepath")
    # print(batch["audio"].shape)
    # print(type(batch["audio"]))
    return batch

def extract_feature(batch):
    audio=[np.array(i) for i in batch["audio"]]
    # print(len(batch["audio"][0]))
    # compute log-Mel input features from input audio array
    features=feature_extractor(audio, sampling_rate=16000).input_features
    batch["input_features"] = features
    tokenizer.set_prefix_tokens(language="Vietnamese", task="transcribe") 
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, num_proc=4)

dataset = dataset.map(extract_feature, num_proc=4,batched=True,batch_size=2)

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
import evaluate

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=10,
    logging_steps=1,
    report_to=["tensorboard"],
    # load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    # push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train() 