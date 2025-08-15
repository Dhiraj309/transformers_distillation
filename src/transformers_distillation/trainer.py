from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .utils import TaskType, detect_task_type

try:
    from transformers.integrations.accelerate import AcceleratorConfig
except ImportError:
    AcceleratorConfig = None  # Older Transformers versions won't have this

class DistillationTrainer(Trainer):
    def __init__(
        self,
        model, 
        args: TrainingArguments,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        teacher_model=None,
        is_pretrained = False,
        kd_alpha=0.5,
        temperature=2.0,
        **kwargs
    ):
        # # Get TrainingArguments from args or kwargs
        # training_args = None
        # if args and hasattr(args[0], "__dict__"):
        #     training_args = args[0]
        # elif "args" in kwargs and hasattr(kwargs["args"], "__dict__"):
        #     training_args = kwargs["args"]

        # Patch accelerator_config if supported and missing
        # if AcceleratorConfig is not None and training_args is not None:
        #     if not hasattr(training_args, "accelerator_config") or training_args.accelerator_config is None:
        #         training_args.accelerator_config = AcceleratorConfig()

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )

        self.teacher_model = teacher_model
        self.kd_alpha = kd_alpha
        self.temperature = temperature


        # Decide how to detect task type
        if is_pretrained:
            # Pretrained → detect using name/path (config preserved)
            self.task_type = detect_task_type(model.name_or_path)
        else:
            # From scratch → detect directly from model instance
            self.task_type = detect_task_type(model)

        if self.teacher_model is not None:
            self.teacher_model.to(self.model.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False


        if self.teacher_model is not None:
            self.teacher_model.to(self.model.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def shift_tokens_right(
            self,
            input_ids,
            pad_tokens_id,
            decoder_start_token_id
    ):
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted.masked_fill_(shifted == -100, pad_tokens_id)

        return shifted

    def prepare_labels(self, inputs):
        """
        Dynamically prepare labels based on task type
        """
        if self.task_type == TaskType.CAUSAL_LM:
            return inputs.get("labels", inputs["input_ids"])
        elif self.task_type == TaskType.MLM:
            # Assume labels are already masked (MLM dataset)
            return inputs.get("labels", inputs["input_ids"])
        elif self.task_type == TaskType.SEQ2SEQ_LM:
            labels = inputs.get("labels", inputs["input_ids"])
            if "decoder_input_ids" not in inputs:
                inputs["decoder_input_ids"] = self.shift_tokens_right(
                    labels,
                    self.model.config.pad_token_id,
                    self.model.config.decoder_start_token_id
                )
            return labels
        else:
            # fallback
            return inputs.get("labels", inputs["input_ids"])


def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = self.prepare_labels(inputs)

    student_outputs = model(**inputs)
    student_logits = student_outputs.logits

    loss_fct = torch.nn.CrossEntropyLoss()
    lm_loss = loss_fct(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1)
    )

    # KD loss only during training
    if self.teacher_model is not None and model.training:
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        kd_loss = F.kl_div(
            input=F.log_softmax(student_logits / self.temperature, dim=-1),
            target=F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        loss = self.kd_alpha * kd_loss + (1.0 - self.kd_alpha) * lm_loss
    else:
        loss = lm_loss

    return (loss, student_outputs) if return_outputs else loss
    

def DistillTrainer(
    teacher_model,
    student_model,
    train_dataset,
    tokenizer,
    training_args: TrainingArguments,
    is_pretrained = False,
    kd_alpha=0.5,
    temperature=2.0
):
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        kd_alpha=kd_alpha,
        temperature=temperature
    )
    return trainer
