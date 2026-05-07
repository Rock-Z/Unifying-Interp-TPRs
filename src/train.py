import gin
import torch
import optuna
import os
import numpy as np
import json
from typing import Literal
from digits import load_digits, tokenize_function
from utils import parse_args_for_gin, gin_config_to_readable_dictionary
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    EncoderDecoderConfig,
)
from model import (
    RecurrentEncoderDecoderModel,
    RecurrentDecoderConfig,
    RecurrentEncoderConfig,
    TensorProductEncoderForPretraining,
    TensorProductEncoderConfig,
)


def compute_metrics(eval_pred, tokenizer):
    """
    Compute accuracy for autoregressive evaluation (predictions are
    token ids given by `.generate()` instead of logits); assumes that
    the first token is the bos token
    """
    predictions, labels = eval_pred
    # discard first column of predictions (bos token)
    predictions = predictions[:, 1:]
    # discard last column of labels (no predictions corresponding to them)
    # also, predictions reach eos, so last col is likely padding
    labels = labels[:, :-1]
    # compute accuracy, ignoring pad tokens
    accuracy = np.sum(
        (predictions == labels) & (labels != tokenizer.pad_token_id)
    ) / np.sum(labels != tokenizer.pad_token_id)
    # find all sequences where all tokens are correct
    correct_sequences = (
        np.sum((predictions != labels) & (labels != tokenizer.pad_token_id), axis=1)
        == 0
    )
    return {"token_accuracy": accuracy, "sequence_accuracy": np.mean(correct_sequences)}


def sanity_check(trainer, tokenizer, num_examples=3):
    """Print correct and incorrect examples from evaluation"""
    eval_dataloader = trainer.get_eval_dataloader()
    correct = []
    incorrect = []

    for batch in eval_dataloader:
        inputs = {k: v.to(trainer.args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = trainer.prediction_step(
                trainer.model, inputs, prediction_loss_only=False
            )
        preds = outputs[1].cpu().numpy()
        labels = inputs["labels"].cpu().numpy()
        filler_ids = inputs["filler_ids"].cpu().numpy()

        for i in range(len(preds)):
            input_str = tokenizer.decode(filler_ids[i], skip_special_tokens=True)
            label_str = tokenizer.decode(labels[i], skip_special_tokens=True)
            pred_str = tokenizer.decode(preds[i], skip_special_tokens=True)

            if pred_str == label_str:
                if len(correct) < num_examples:
                    correct.append((input_str, label_str, pred_str))
            else:
                if len(incorrect) < num_examples:
                    incorrect.append((input_str, label_str, pred_str))

        if len(correct) >= num_examples and len(incorrect) >= num_examples:
            break

    print("\n=== Correct Examples ===")
    for i, (inp, lbl, pred) in enumerate(correct):
        print(f"Example {i+1}:")
        print("Input:", inp)
        print("   GT:", lbl)
        print(" Pred:", pred)
        print()

    print("\n=== Incorrect Examples ===")
    for i, (inp, lbl, pred) in enumerate(incorrect):
        print(f"Example {i+1}:")
        print("Input:", inp)
        print("   GT:", lbl)
        print(" Pred:", pred)
        print()


def save_metrics(metrics, output_dir, filename="eval_results.json"):
    """Save evaluation metrics to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Convert any non-serializable values to strings
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.float32) or isinstance(v, np.float64):
            serializable_metrics[k] = float(v)
        elif isinstance(v, np.int32) or isinstance(v, np.int64):
            serializable_metrics[k] = int(v)
        else:
            serializable_metrics[k] = v

    with open(output_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"Evaluation results saved to {output_path}")


@gin.configurable(denylist=["tokenizer"])
def seq2seq_init(
    tokenizer,
    encoder_config,
    decoder_config,
) -> RecurrentEncoderDecoderModel:

    # build seq2seq model from encoder and decoder configs
    # some configs (e.g. vocab_size) are inferred from tokenizer
    encoder_config = RecurrentEncoderConfig(vocab_size=len(tokenizer), **encoder_config)

    decoder_config = RecurrentDecoderConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **decoder_config,
    )

    # build model by combining encoder and decoder configs
    model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    model = RecurrentEncoderDecoderModel(model_config)

    return model


def train_seq2seq(
    dataset,
    tokenizer,
    seq2seq_training_args: Seq2SeqTrainingArguments,
):
    training_arguments = seq2seq_training_args

    trainer = Seq2SeqTrainer(
        model_init=lambda: seq2seq_init(tokenizer=tokenizer),
        processing_class=tokenizer,
        # no need for padding in collator, as it is handled by tokenizer
        args=training_arguments,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=lambda examples: tokenize_function(examples, tokenizer),
    )

    trainer.train()
    trainer.model.save_pretrained(training_arguments.output_dir + "/best_model")
    model = RecurrentEncoderDecoderModel.from_pretrained(
        training_arguments.output_dir + "/best_model"
    )
    seq2seq_eval_stats = trainer.evaluate()

    # Save evaluation metrics
    save_metrics(seq2seq_eval_stats, training_arguments.output_dir, "eval_results.json")

    return trainer, model, seq2seq_eval_stats


@gin.configurable(denylist=["tokenizer", "model"])
def train_tpe(
    dataset,
    tokenizer,
    model,
    tpe_config,
    tpe_training_args: TrainingArguments,
):
    """
    Stage 2: Train the Tensor Product Encoder to align with encoder embeddings
    """
    # Tensor Product Encoder setup
    tpe_config_obj = TensorProductEncoderConfig(
        n_fillers=len(tokenizer),
        filler_pad_token_id=tokenizer.pad_token_id,
        **tpe_config,
    )
    tpe = TensorProductEncoderForPretraining(tpe_config_obj, embedding_model=model.encoder)

    # Train TPE to align with encoder embeddings
    # Modify the arguments object directly
    tpe_training_args.label_names = ["filler_ids", "role_ids"]
    tpe_training_args.prediction_loss_only = True
    tpe_training_arguments = tpe_training_args

    # Extract role_scheme from TPE config for data collator
    role_scheme = tpe_config_obj.role_scheme
    if role_scheme is None:
        raise ValueError("role_scheme must be specified in tpe_config")

    tpe_trainer = Trainer(
        model=tpe,
        args=tpe_training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=lambda examples: tokenize_function(
            examples, tokenizer, format="tpe", role_scheme=role_scheme
        ),
    )

    tpe_trainer.train()
    tpe.save_pretrained(tpe_training_args.output_dir + "/best_model")

    # Save TPE training evaluation metrics
    tpe_train_eval_stats = tpe_trainer.evaluate()
    save_metrics(
        tpe_train_eval_stats, tpe_training_args.output_dir, "eval_results.json"
    )

    return tpe_trainer, tpe


@gin.configurable(denylist=["tokenizer", "model", "tpe"])
def evaluate_tpe(
    dataset,
    tokenizer,
    model,
    tpe,
    tpe_training_args: Seq2SeqTrainingArguments,
):
    """
    Stage 3: Evaluate TP Encoder in terms of substitution accuracy
    """
    tpr_encoder_decoder = RecurrentEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder=tpe.encoder, decoder=model.decoder
    )

    # Extract role_scheme from TPE config for data collator
    role_scheme = tpe.config.role_scheme
    if role_scheme is None:
        raise ValueError("role_scheme must be specified in TPE config")

    eval_trainer = Seq2SeqTrainer(
        model=tpr_encoder_decoder,
        args=tpe_training_args,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        eval_dataset=dataset["test"],
        data_collator=lambda examples: tokenize_function(
            examples, tokenizer, format="tpe_eval", role_scheme=role_scheme
        ),
    )

    eval_stats = eval_trainer.evaluate()
    print("Sanity check for TPR Substitution Accuracy:")
    sanity_check(eval_trainer, tokenizer, 5)

    # Save TPE substitution accuracy evaluation metrics
    save_metrics(
        eval_stats, tpe_training_args.output_dir, "substitution_eval_results.json"
    )

    return eval_stats, tpr_encoder_decoder, eval_trainer


@gin.configurable
def main(
    data_paths_dict,
    seq2seq_training_args: Seq2SeqTrainingArguments,
    tpe_config,
    tpe_training_args: TrainingArguments,
    wandb_configs: dict = {},
    do_hyperparameter_search: bool = False,
    skip_seq2seq: bool = False,
):

    # load dataset and tokenizer
    dataset, tokenizer = load_digits(file_paths=data_paths_dict)

    # ----------------------------------------------------------------
    # STAGE 1: Train Seq2Seq model on digits task
    # ----------------------------------------------------------------
    if do_hyperparameter_search:
        raise NotImplementedError("Hyperparameter search not implemented yet.")

    if not skip_seq2seq:
        trainer, model, seq2seq_eval_stats = train_seq2seq(
            dataset=dataset,
            tokenizer=tokenizer,
            seq2seq_training_args=seq2seq_training_args,
        )
    else:
        model = RecurrentEncoderDecoderModel.from_pretrained(
            seq2seq_training_args.output_dir + "/best_model"
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=seq2seq_training_args,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
            eval_dataset=dataset["valid"],
            data_collator=lambda examples: tokenize_function(examples, tokenizer),
        )
        seq2seq_eval_stats = trainer.evaluate()

    # ----------------------------------------------------------------
    # STAGE 2: Train TPR Encoder to align with encoder embeddings
    # ----------------------------------------------------------------
    tpe_trainer, tpe = train_tpe(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        tpe_config=tpe_config,
        tpe_training_args=tpe_training_args,
    )

    # ----------------------------------------------------------------
    # STAGE 3: Evaluate TP Encoder in terms of substitution accuracy
    # ----------------------------------------------------------------
    eval_stats, tpr_encoder_decoder, eval_trainer = evaluate_tpe(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        tpe=tpe,
    )

    print("Seq2Seq Accuracy: ", seq2seq_eval_stats)
    print("TPR Model Substitution Accuracy (autoregressive):\n", eval_stats)

    return {
        "Seq2Seq": seq2seq_eval_stats,
        "TPE": eval_stats,
        "Seq2Seq_model": model,
        "TPE_model": tpr_encoder_decoder,
        "Seq2Seq_trainer": trainer,
        "TPE_trainer": tpe_trainer,
        "gin_config": gin.operative_config_str(),
    }


if __name__ == "__main__":
    gin.external_configurable(Seq2SeqTrainingArguments, module="transformers")
    gin.external_configurable(TrainingArguments, module="transformers")
    parse_args_for_gin()
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "inverting-tpr"
    main()
