import torch
import numpy as np
import gin
from typing import Optional
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from digits import load_digits, tokenize_function
from model import RecurrentEncoderDecoderModel, TensorProductEncoderForPretraining
from probing import (
    LinearProbe,
    LinearProbeConfig,
    auto_select_role_pinv_l2_lambda,
    auto_select_tpe_output_l2_lambda,
)
from utils import parse_args_for_gin
from analogy_utils import digits_without_special_tokens
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# see configs/digit_invert_tpr_all_pos.gin for example usage


def _is_content_only_role_scheme(role_scheme: Optional[str]) -> bool:
    return bool(role_scheme and role_scheme.endswith("_content"))


def _map_probe_position_to_tpe_role_id(position: int, role_scheme: Optional[str]) -> int:
    if position <= 0:
        return -position
    return position if _is_content_only_role_scheme(role_scheme) else position + 1

def compute_metrics(pred):
    # Handle both tuple and array predictions
    logits = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
    preds = np.asarray(logits).argmax(-1)
    labels = np.asarray(pred.label_ids)
    # Flatten possible extra dims
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    accuracy = (labels == preds).mean()
    return {'accuracy': accuracy}

def print_examples(bucket_name: str, rows: list[tuple], tokenizer, role_id: int) -> None:
    if not rows:
        print(f"{bucket_name}: (none found)")
        return
    print(f"{bucket_name}:")
    for _, _, y, yhat, input_ids in rows:
        text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        print(f"  input: {text}")
        print(f"  target@{role_id}: {y} ({tokenizer.decode([y])})")
        print(f"  pred  @{role_id}: {yhat} ({tokenizer.decode([yhat])})")

def san_check(trainer: Trainer, tokenizer, role_id: int, n_samples: int = 3) -> None:
    model = trainer.model
    model.eval()

    dataloader = trainer.get_eval_dataloader()
    correct = []
    incorrect = []

    for batch_idx, batch in enumerate(dataloader):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        labels = labels.to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(-1)

        for i in range(int(labels.shape[0])):
            ok = bool(preds[i].item() == labels[i].item())
            bucket = correct if ok else incorrect
            if len(bucket) < n_samples:
                bucket.append((batch_idx, i, int(labels[i].item()), int(preds[i].item()), batch["input_ids"][i]))
        if len(correct) >= n_samples and len(incorrect) >= n_samples:
            break

    print(f"\n[SAN_CHECK] role_id={role_id}")
    print_examples("correct", correct, tokenizer=tokenizer, role_id=role_id)
    print_examples("incorrect", incorrect, tokenizer=tokenizer, role_id=role_id)

def collate_fn(examples, position, model, tokenizer):
    if model.encoder.config.model_type == 'recurrent_encoder':
        tokenized = tokenize_function(examples, tokenizer, 'seq2seq')
        if position >= 0:
            tokenized['labels'] = tokenized['input_ids'][:, position]
        else:
            B = tokenized['input_lengths'].shape[0]
            ids = tokenized['input_lengths'] + position
            tokenized['labels'] = tokenized['input_ids'][torch.arange(B), ids]
    else:
        assert model.encoder.config.model_type == 'tensor_product_encoder'
        role_scheme = model.encoder.config.role_scheme
        if role_scheme is None:
            raise ValueError("role_scheme must be specified in TPE config")
        tokenized = tokenize_function(examples, tokenizer, 'tpe', role_scheme=role_scheme)
        role = _map_probe_position_to_tpe_role_id(position, role_scheme)
        B = tokenized['filler_ids'].shape[0]
        tokenized['labels'] = tokenized['filler_ids'][tokenized['role_ids']==role]
    return tokenized

def filter_min_len(dataset, tokenizer, min_len):
    """
    Filter dataset to only include examples with input length >= min_len.
    """
    def _filter(ex):
        return len(tokenizer(ex['input'])['input_ids']) >= min_len
    return dataset.filter(_filter)


def filter_examples_for_filler_role_pair(dataset, filler: int, position: int):
    """Filter a digits split to examples containing a given filler-position pair."""
    if position <= 0:
        raise ValueError("Held-out positions must be positive")

    def _filter(ex):
        digits = digits_without_special_tokens(ex["input"]).split()
        return position <= len(digits) and int(digits[position - 1]) == filler

    return dataset.filter(_filter)

@gin.configurable
def evaluate_probes_for_role(
    dataset,
    tokenizer,
    seq2seq_model,
    tpe,
    role_id,
    eval_dataset,
    generalization_eval_dataset=None,
    heldout_pair=None,
    *,
    analytic_training_args,
    trainable_training_args,
    regularization='l2',
    reg_param=0.1,
    role_unbinding: str = "pinv",
    role_pinv_regularization: str = "l2",
    role_pinv_l2_lambda: Optional[float] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    sanity_check_samples: int = 0,
):
    """
    Evaluate analytic and trainable probes for a given role.
    analytic_training_args and trainable_training_args must be gin-configured TrainingArguments instances.
    """
    # Why +2:
    # The input is typically wrapped with <BOS> and <SEP>. adding 2 makes sure the
    # len of the input is always >= min_len.
    min_len = role_id + 2 if role_id >= 0 else (-role_id + 1)
    if regularization not in ('l2', 'atol', 'topk'):
        regularization = 'l2'
    role_scheme = getattr(tpe.config, "role_scheme", None)
    probe_model = LinearProbe.from_tpencoder(
        tpencoder=tpe,
        encoder=seq2seq_model.encoder,
        role_id=_map_probe_position_to_tpe_role_id(role_id, role_scheme),
        regularization=regularization,
        l2_lambda=reg_param if regularization == 'l2' else None,
        atol=reg_param if regularization == 'atol' else None,
        topk=int(reg_param) if regularization == 'topk' else None,
        role_unbinding=role_unbinding,
        role_pinv_regularization=role_pinv_regularization,
        role_pinv_l2_lambda=role_pinv_l2_lambda,
        role_pinv_atol=role_pinv_atol,
        role_pinv_topk=role_pinv_topk,
    )
    analytic_trainer = Trainer(
        model=probe_model,
        args=analytic_training_args,
        eval_dataset=filter_min_len(eval_dataset, tokenizer, min_len),
        data_collator=lambda ex: collate_fn(ex, role_id, probe_model, tokenizer),
        compute_metrics=compute_metrics
    )
    analytic_results = analytic_trainer.evaluate()
    if sanity_check_samples and sanity_check_samples > 0:
        print("\n[Analytic probe]")
        san_check(analytic_trainer, tokenizer, role_id=role_id, n_samples=int(sanity_check_samples))
    analytic_trainer.save_model(os.path.join(analytic_training_args.output_dir, f"role_{role_id}"))
    encoder_hidden_size = None
    if getattr(seq2seq_model.encoder.config, "architecture", None) == "LSTM":
        if hasattr(tpe, "output_layer") and tpe.output_layer is not None:
            encoder_hidden_size = int(tpe.output_layer.out_features)
        else:
            encoder_hidden_size = int(seq2seq_model.encoder.config.hidden_size) * 2
    config = LinearProbeConfig(
        encoder_model_type=seq2seq_model.encoder.config.model_type,
        encoder_hidden_size=encoder_hidden_size,
        num_labels=int(tpe.filler_embedding.num_embeddings)
    )
    trainable_probe = LinearProbe(config, seq2seq_model.encoder)
    trainable_trainer = Trainer(
        model=trainable_probe,
        args=trainable_training_args,
        train_dataset=filter_min_len(dataset['train'], tokenizer, min_len),
        eval_dataset=filter_min_len(eval_dataset, tokenizer, min_len),
        data_collator=lambda ex: collate_fn(ex, role_id, trainable_probe, tokenizer),
        compute_metrics=compute_metrics
    )
    trainable_trainer.train()
    trained_results = trainable_trainer.evaluate()
    if sanity_check_samples and sanity_check_samples > 0:
        print("\n[Trained probe]")
        san_check(trainable_trainer, tokenizer, role_id=role_id, n_samples=int(sanity_check_samples))
    result = {
        'role_id': role_id,
        'analytic_accuracy': analytic_results['eval_accuracy'],
        'trained_accuracy': trained_results['eval_accuracy'],
        'test': {
            'analytic_accuracy': analytic_results['eval_accuracy'],
            'trained_accuracy': trained_results['eval_accuracy'],
            'num_examples': len(filter_min_len(eval_dataset, tokenizer, min_len)),
        },
    }
    if generalization_eval_dataset is not None:
        filtered_generalization_dataset = filter_min_len(generalization_eval_dataset, tokenizer, min_len)
        analytic_generalization_results = analytic_trainer.evaluate(
            eval_dataset=filtered_generalization_dataset,
            metric_key_prefix="generalization_eval",
        )
        trained_generalization_results = trainable_trainer.evaluate(
            eval_dataset=filtered_generalization_dataset,
            metric_key_prefix="generalization_eval",
        )
        result['generalization_matched'] = {
            'analytic_accuracy': analytic_generalization_results['generalization_eval_accuracy'],
            'trained_accuracy': trained_generalization_results['generalization_eval_accuracy'],
            'num_examples': len(filtered_generalization_dataset),
        }
        if heldout_pair is not None:
            result['heldout_pair'] = {
                'filler': int(heldout_pair[0]),
                'position': int(heldout_pair[1]),
            }
    return result

def get_optimal_reg_param(tpe_path):
    """
    Compute the optimal tiknohov regularization parameter for the given TPE path, which 
    is the square root of MSE loss, cached in eval_results.json of the TPE checkpoint.
    """
    eval_results_path = os.path.join(tpe_path, 'eval_results.json')
    if not os.path.exists(eval_results_path):
        raise FileNotFoundError(f"Could not find eval_results.json at {eval_results_path}. Please provide reg_param explicitly or ensure the file exists.")
    with open(eval_results_path, 'r') as f:
        eval_results = json.load(f)
    eval_loss = eval_results.get('eval_loss', None)
    if eval_loss is None:
        raise ValueError(f"eval_loss not found in {eval_results_path}.")
    return float(np.sqrt(eval_loss))

def auto_select_reg_param_from_samples(dataset, tokenizer, seq2seq_model, tpe, sample_size: int = 128):
    role_scheme = getattr(tpe.config, "role_scheme", None)
    if role_scheme is None:
        raise ValueError("role_scheme must be specified in TPE config")

    split = dataset["train"]
    n = min(int(sample_size), len(split))
    examples = [split[i] for i in range(n)]
    tokenized = tokenize_function(examples, tokenizer, "tpe", role_scheme=role_scheme)

    device = tpe.filler_embedding.weight.device
    with torch.no_grad():
        target_hidden = seq2seq_model.encoder(
            input_ids=tokenized["embedding_model_input_ids"].to(device),
            input_lengths=tokenized["embedding_model_input_lengths"].to(device),
        ).last_hidden_state
        if isinstance(target_hidden, tuple):
            target_hidden = torch.cat([target_hidden[0], target_hidden[1]], dim=-1)

    reg_lambda, best_value, (log_lo, log_hi) = auto_select_tpe_output_l2_lambda(
        tpe,
        target_hidden,
        tokenized["filler_ids"],
        tokenized["role_ids"],
        device=device,
    )
    return float(reg_lambda), float(best_value), (float(log_lo), float(log_hi))

def save_probe_results_plot_and_config(results, timestamp=None, results_dir="results"):
    """
    Plot analytic and trained probe accuracy vs role_id, save as PNG/PDF, and save gin config.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)
    analytic_acc = [r['analytic_accuracy'] for r in results]
    trained_acc = [r['trained_accuracy'] for r in results]
    role_ids = [r['role_id'] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(role_ids, analytic_acc, 'o-', label='Analytic Probe')
    plt.plot(role_ids, trained_acc, 'x-', label='Trained Probe')
    plt.xlabel('Role ID')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.title('Probe Accuracy by Role Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(role_ids)
    plt.savefig(f"{results_dir}/probe_compare_{timestamp}.png")
    plt.savefig(f"{results_dir}/probe_compare_{timestamp}.pdf")
    print(f"Saved plot to {results_dir}/probe_compare_{timestamp}.png and {results_dir}/probe_compare_{timestamp}.pdf")
    # Save raw results to JSON (both timestamped and stable filenames)
    results_json = {
        'results': results,
        'timestamp': timestamp,
    }
    with open(f"{results_dir}/probe_compare_{timestamp}.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    with open(f"{results_dir}/probe_compare_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results JSON to {results_dir}/probe_compare_{timestamp}.json and {results_dir}/probe_compare_results.json")
    # Save gin operative config
    with open(f"{results_dir}/probe_compare_{timestamp}.gin", 'w') as f:
        f.write(gin.operative_config_str())
    print(f"Saved gin config to {results_dir}/probe_compare_{timestamp}.gin")

@gin.configurable
def main(
    data_paths_dict,
    seq2seq_path=None,
    tpe_path=None,
    regularization='l2',
    reg_param=None,
    role_unbinding: str = "pinv",
    role_pinv_regularization: str = "l2",
    role_pinv_l2_lambda: Optional[float] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    role_sweep_range=range(-7, 0),
    heldout_pairs=None,
    generalization_eval_split_name: Optional[str] = None,
    results_dir='results',
    reg_param_sample_size: int = 128,
):
    """
    Run probe comparison for all positions, matching probe_compare_all_pos logic.
    TrainingArguments are injected into evaluate_probes_for_role via gin.
    """
    if seq2seq_path is None or tpe_path is None:
        raise ValueError("You must provide seq2seq_path and tpe_path via gin config or arguments.")

    dataset, tokenizer = load_digits(file_paths=data_paths_dict)
    seq2seq_model = RecurrentEncoderDecoderModel.from_pretrained(seq2seq_path + '/best_model')
    tpr_model = TensorProductEncoderForPretraining.from_pretrained(tpe_path + '/best_model')
    try:
        tpe = tpr_model.encoder
    except Exception:
        tpe = tpr_model

    if role_unbinding == "pinv" and role_pinv_regularization == "l2" and role_pinv_l2_lambda is None:
        role_scheme = getattr(tpe.config, "role_scheme", None)
        if role_scheme is None:
            raise ValueError("role_scheme must be specified in TPE config")
        split = dataset["train"]
        n = min(128, len(split))
        examples = [split[i] for i in range(n)]
        tokenized = tokenize_function(examples, tokenizer, "tpe", role_scheme=role_scheme)
        device = tpe.filler_embedding.weight.device
        reg_lambda, best_value, (log_lo, log_hi) = auto_select_role_pinv_l2_lambda(
            tpe,
            filler_ids=tokenized["filler_ids"],
            role_ids=tokenized["role_ids"],
            device=device,
        )
        role_pinv_l2_lambda = float(reg_lambda)
        print(
            f"[INFO] Auto-selected role unbinding l2 ≈ {role_pinv_l2_lambda:.5g} "
            f"(val_mse≈{best_value:.4e}; window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}])"
        )

    if reg_param is None:
        if regularization == "l2":
            reg_param_val, best_value, (log_lo, log_hi) = auto_select_reg_param_from_samples(
                dataset,
                tokenizer,
                seq2seq_model,
                tpe,
                sample_size=reg_param_sample_size,
            )
            print(
                f"[INFO] Auto-selected reg_param ≈ {reg_param_val:.5g} "
                f"(objective=mse; val_mse≈{best_value:.4e}; search window "
                f"[{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}])"
            )
        else:
            reg_param_val = get_optimal_reg_param(tpe_path)
            print(f"[INFO] Using fallback reg_param = {reg_param_val} from {tpe_path} (regularization={regularization})")
    else:
        reg_param_val = reg_param
        print(f"[INFO] Using provided reg_param = {reg_param_val}")

    results = []
    normalized_holdout_pairs = None
    if heldout_pairs is not None:
        normalized_holdout_pairs = [(int(filler), int(position)) for filler, position in heldout_pairs]
    for role_id in role_sweep_range:
        generalization_eval_dataset = None
        heldout_pair = None
        if generalization_eval_split_name is not None:
            if normalized_holdout_pairs is None:
                raise ValueError("heldout_pairs must be provided when generalization_eval_split_name is set")
            if role_id <= 0:
                raise ValueError("Heldout probe evaluation currently expects positive raw positions")
            heldout_pair = next(
                ((filler, position) for filler, position in normalized_holdout_pairs if position == int(role_id)),
                None,
            )
            if heldout_pair is None:
                raise ValueError(f"No held-out pair configured for role_id={role_id}")
            generalization_eval_dataset = filter_examples_for_filler_role_pair(
                dataset[generalization_eval_split_name],
                filler=int(heldout_pair[0]),
                position=int(heldout_pair[1]),
            )
        result = evaluate_probes_for_role(
            dataset,
            tokenizer,
            seq2seq_model,
            tpe,
            role_id,
            eval_dataset=dataset['test'],
            generalization_eval_dataset=generalization_eval_dataset,
            heldout_pair=heldout_pair,
            regularization=regularization,
            reg_param=reg_param_val,
            role_unbinding=role_unbinding,
            role_pinv_regularization=role_pinv_regularization,
            role_pinv_l2_lambda=role_pinv_l2_lambda,
            role_pinv_atol=role_pinv_atol,
            role_pinv_topk=role_pinv_topk,
        )
        if 'generalization_matched' in result:
            print(
                f"Role {role_id}: Test Analytic={result['test']['analytic_accuracy']:.4f}, "
                f"Test Trained={result['test']['trained_accuracy']:.4f}, "
                f"Generalization Analytic={result['generalization_matched']['analytic_accuracy']:.4f}, "
                f"Generalization Trained={result['generalization_matched']['trained_accuracy']:.4f}"
            )
        else:
            print(f"Role {role_id}: Analytic={result['analytic_accuracy']:.4f}, Trained={result['trained_accuracy']:.4f}")
        results.append(result)
    save_probe_results_plot_and_config(results, results_dir=results_dir)
    return results

if __name__ == "__main__":
    gin.external_configurable(TrainingArguments)
    parse_args_for_gin()
    main()
