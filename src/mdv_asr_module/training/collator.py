from __future__ import annotations

import torch


class QwenTrainingCollator:
    def __init__(self, processor, language: str = "Vietnamese"):
        self.processor = processor
        self.language = language

    def build_target(self, record) -> str:
        return f"language {self.language}<asr_text>{record.text.strip()}"

    def build_messages(self, items):
        conversations = []
        for item in items:
            record, audio = item["record"], item["audio"]
            messages = [
                {"role": "user", "content": [{"type": "audio", "audio": audio.waveform}]},
                {"role": "assistant", "content": [{"type": "text", "text": self.build_target(record)}]},
            ]
            conversations.append(messages)
        return conversations

    def __call__(self, items):
        if not items:
            raise ValueError("Cannot collate an empty training batch")
        conversations = self.build_messages(items)
        batch = self.processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True,
            processor_kwargs={"output_labels": True, "sampling_rate": 16000,
                              "padding": "longest", "truncation": False},
        )
        self.validate_labels(batch)
        batch["records"] = [item["record"] for item in items]
        return batch

    @staticmethod
    def validate_labels(batch) -> None:
        labels = batch.get("labels")
        if not torch.is_tensor(labels) or labels.ndim != 2:
            raise ValueError("Processor did not return a two-dimensional labels tensor")
        supervised = labels != -100
        if not torch.all(supervised.any(dim=-1)):
            raise ValueError("At least one sample has no supervised assistant tokens")
        attention = batch.get("attention_mask")
        if torch.is_tensor(attention) and attention.shape == labels.shape:
            if torch.any(supervised & (attention == 0)):
                raise ValueError("Padded positions must be masked from assistant labels")

