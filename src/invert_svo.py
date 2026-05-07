import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import gin
import numpy as np
import torch
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from sentences import load_sentences, SVORoleAssigner
from model import TensorProductEncoderForPretraining
from probing import (
    LinearProbe,
    LinearProbeConfig,
    auto_select_role_pinv_l2_lambda,
    auto_select_tpe_output_l2_lambda,
)
from utils import parse_args_for_gin, load_dataset_with_embeddings


def compute_metrics(pred):
    labels = pred.label_ids
    preds_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = np.asarray(preds_logits).argmax(-1)
    labels = np.asarray(labels)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    accuracy = (labels == preds).mean()
    return {'accuracy': float(accuracy)}


def add_labels_for_role(dataset_split, role_id: int):
    """Add a labels column with the filler at role_id, or -1 if missing."""
    import numpy as np
    ds = dataset_split
    if 'labels' in ds.column_names:
        ds = ds.remove_columns('labels')
    role_ids_data = ds['role_ids']
    filler_ids_data = ds['filler_ids']
    labels_list: List[int] = []
    for i in range(len(role_ids_data)):
        current_roles = role_ids_data[i]
        current_fillers = filler_ids_data[i]
        label = -1
        for j in range(len(current_roles)):
            if current_roles[j] == role_id:
                label = current_fillers[j]
                break
        labels_list.append(label)
    ds = ds.add_column('labels', np.array(labels_list))
    return ds


def collate_embeddings(batch):
    """Collate hidden_states and labels into torch tensors."""
    hidden = torch.tensor([b['hidden_states'] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
    return {'hidden_states': hidden, 'labels': labels}


def save_results(results: List[Dict], results_dir: str, tag: str):
    """Write probe results and gin config to stable and timestamped files."""
    os.makedirs(results_dir, exist_ok=True)
    # Save stable and timestamped files
    with open(os.path.join(results_dir, f"probe_compare_svo_{tag}.json"), 'w') as f:
        json.dump({'results': results, 'timestamp': tag}, f, indent=2)
    with open(os.path.join(results_dir, "probe_compare_results_svo.json"), 'w') as f:
        json.dump({'results': results, 'timestamp': tag}, f, indent=2)
    with open(os.path.join(results_dir, f"probe_compare_svo_{tag}.gin"), 'w') as f:
        f.write(gin.operative_config_str())


def format_filler(label_id: int, role_assigner: SVORoleAssigner) -> str:
    """Format a filler id as noun/verb token string, with fallback markers."""
    if label_id < 0:
        return "<missing>"
    noun_count = len(role_assigner.noun_filler2idx)
    if label_id < noun_count:
        return role_assigner.noun_idx2filler[label_id]
    verb_idx = label_id - noun_count
    if verb_idx in role_assigner.verb_idx2filler:
        return role_assigner.verb_idx2filler[verb_idx]
    return f"<unk:{label_id}>"


def remap_labels_for_role(dataset_split, label_offset: int, label_count: int):
    """Filter labels to a role slice and remap them to a contiguous range."""
    max_label = label_offset + label_count
    # Keep only labels in the role slice, then shift to a contiguous range.
    dataset_split = dataset_split.filter(lambda x: label_offset <= x['labels'] < max_label)
    labels = np.asarray(dataset_split['labels']) - label_offset
    dataset_split = dataset_split.remove_columns('labels')
    dataset_split = dataset_split.add_column('labels', labels)
    return dataset_split


def log_sanity_examples(
    trainer: Trainer,
    dataset_split,
    role_name: str,
    role_assigner: SVORoleAssigner,
    title: str,
    label_offset: int,
    max_examples: int = 5,
):
    """Print a few correct/incorrect examples for a probe over a dataset split."""
    if dataset_split is None or len(dataset_split) == 0:
        print(f"[SAN_CHECK] {title} ({role_name}): no examples available")
        return

    def _print_bucket(bucket_name: str, rows: List[Tuple[int, int]]) -> None:
        if not rows:
            print(f"{bucket_name}: (none found)")
            return
        print(f"{bucket_name}:")
        for i, (gold_id, pred_id) in enumerate(rows):
            gold = format_filler(gold_id, role_assigner)
            guess = format_filler(pred_id, role_assigner)
            print(f"  ex{i}: gold={gold} pred={guess}")

    model = trainer.model
    model.eval()
    correct: List[Tuple[int, int]] = []
    incorrect: List[Tuple[int, int]] = []
    dataloader = trainer.get_eval_dataloader(dataset_split)
    for batch in dataloader:
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        labels = labels.to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = logits.argmax(-1)

        for i in range(int(labels.shape[0])):
            gold_id = int(labels[i].item()) + label_offset
            pred_id = int(pred_ids[i].item()) + label_offset
            bucket = correct if pred_id == gold_id else incorrect
            if len(bucket) < max_examples:
                bucket.append((gold_id, pred_id))
        if len(correct) >= max_examples and len(incorrect) >= max_examples:
            break

    print(f"\n[SAN_CHECK] {title} ({role_name})")
    _print_bucket("correct", correct)
    _print_bucket("incorrect", incorrect)


@gin.configurable
def main(
    sentences_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str] = None,
    tpe_path: Optional[str] = None,
    regularization: str = 'l2',
    reg_param: Optional[float] = None,
    reg_param_sample_size: int = 128,
    role_unbinding: str = "pinv",
    role_pinv_regularization: str = "l2",
    role_pinv_l2_lambda: Optional[float] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    role_scheme: str = 'svo',
    results_dir: str = 'results',
    roles_to_eval: Tuple[str, str] = ('subj', 'obj'),
    skip_trainable_probe: bool = False,
    cache_trained_results_path: Optional[str] = None,
    *,
    analytic_training_args: TrainingArguments,
    trainable_training_args: TrainingArguments,
):
    if tpe_path is None:
        raise ValueError("tpe_path must be provided via gin or CLI override")

    # Load dataset and embeddings
    dataset, role_assigner = load_sentences(sentences_path, role_scheme=role_scheme)
    dataset, embedding_dim = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=sentences_path,
        embedding_model_name=embedding_model_name,
        embedding_cache_path=embedding_cache_path,
        embedding_column_name="target_embeddings",
        add_prefix="search_query: " if embedding_model_name.startswith("nomic-ai") else "",
    )
    # rename embeddings column for probing API
    for split in dataset:
        if 'target_embeddings' in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column('target_embeddings', 'hidden_states')

    # Prepare evaluation datasets for subject/object
    eval_datasets = {}
    name_to_roleidx = {}
    if hasattr(role_assigner, 'role2idx'):
        if 'subject' in role_assigner.role2idx:
            name_to_roleidx['subj'] = role_assigner.role2idx['subject']
        if 'verb' in role_assigner.role2idx:
            name_to_roleidx['verb'] = role_assigner.role2idx['verb']
        if 'object' in role_assigner.role2idx:
            name_to_roleidx['obj'] = role_assigner.role2idx['object']
    for name in roles_to_eval:
        if name not in name_to_roleidx:
            continue
        rid = name_to_roleidx[name]
        test_ds = add_labels_for_role(dataset['test'], rid)
        test_ds = test_ds.filter(lambda x: x['labels'] != -1)
        train_ds = add_labels_for_role(dataset['train'], rid)
        train_ds = train_ds.filter(lambda x: x['labels'] != -1)
        eval_datasets[name] = (train_ds, test_ds)

    # Load TPE
    tpr_model = TensorProductEncoderForPretraining.from_pretrained(os.path.join(tpe_path, 'best_model'))
    try:
        tpe = tpr_model.encoder
    except Exception:
        tpe = tpr_model

    # Determine reg_param if None. For l2 output inversion, use the same
    # sample-based objective as SAE construction; eval_loss is a training MSE
    # scale and is not a reliable Tikhonov lambda.
    if reg_param is None:
        if regularization == "l2":
            probe_batch_size = min(int(reg_param_sample_size), len(dataset["train"]))
            if probe_batch_size <= 0:
                raise ValueError("Cannot auto-select reg_param from an empty train split.")
            batch_subset = dataset["train"].select(range(probe_batch_size))
            hidden_states_batch = torch.tensor(batch_subset["hidden_states"], dtype=torch.float32)
            filler_ids_batch = torch.tensor(batch_subset["filler_ids"], dtype=torch.long)
            role_ids_batch = torch.tensor(batch_subset["role_ids"], dtype=torch.long)
            device = tpe.filler_embedding.weight.device
            reg_lambda, best_value, (log_lo, log_hi) = auto_select_tpe_output_l2_lambda(
                tpe,
                hidden_states_batch.to(device),
                filler_ids_batch.to(device),
                role_ids_batch.to(device),
                device=device,
            )
            reg_param = float(reg_lambda)
            detail_tokens: List[str] = []
            if np.isfinite(best_value):
                detail_tokens.append(f"val_mse≈{best_value:.4e}")
            detail_tokens.append(
                f"window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}]"
            )
            print(f"[INFO] Auto-selected reg_param ≈ {reg_param:.5g} ({', '.join(detail_tokens)})")
        else:
            eval_json_candidates = [
                os.path.join(tpe_path, 'eval_results_tpe.json'),
                os.path.join(tpe_path, 'eval_results.json'),
            ]
            eval_loss_val = None
            for p in eval_json_candidates:
                if os.path.exists(p):
                    with open(p, 'r') as f:
                        data = json.load(f)
                    if 'eval_loss' in data:
                        eval_loss_val = float(data['eval_loss'])
                        break
            if eval_loss_val is None:
                raise FileNotFoundError(
                    "Could not infer reg_param: eval_results_tpe.json or eval_results.json "
                    "is missing eval_loss, and sample-based auto-selection currently supports only l2."
                )
            reg_param = float(np.sqrt(max(0.0, eval_loss_val)))
            print(f"[INFO] Inferred reg_param from eval_loss: {reg_param:.5g}")

    if role_unbinding == "pinv" and role_pinv_regularization == "l2" and role_pinv_l2_lambda is None:
        device = tpe.filler_embedding.weight.device
        batch_subset = dataset["train"].select(range(min(128, len(dataset["train"]))))
        filler_ids_tensor = torch.tensor(batch_subset["filler_ids"], dtype=torch.long).to(device)
        role_ids_tensor = torch.tensor(batch_subset["role_ids"], dtype=torch.long).to(device)
        reg_lambda, best_value, (log_lo, log_hi) = auto_select_role_pinv_l2_lambda(
            tpe,
            filler_ids=filler_ids_tensor,
            role_ids=role_ids_tensor,
            device=device,
        )
        role_pinv_l2_lambda = float(reg_lambda)
        print(
            f"[INFO] Auto-selected role unbinding l2 ≈ {role_pinv_l2_lambda:.5g} "
            f"(val_mse≈{best_value:.4e}; window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}])"
        )

    results: List[Dict] = []
    # Prepare trained results cache
    if cache_trained_results_path is None:
        cache_trained_results_path = os.path.join(tpe_path, 'trained_probe_results_svo.json')
    trained_cache: Dict[str, float] = {}
    if os.path.exists(cache_trained_results_path):
        try:
            with open(cache_trained_results_path, 'r') as f:
                trained_cache = json.load(f)
        except Exception:
            trained_cache = {}

    # For each role name, evaluate analytic and trainable probes independently
    for name, (train_ds, test_ds) in eval_datasets.items():
        noun_count = len(role_assigner.noun_filler2idx)
        verb_count = len(role_assigner.verb_filler2idx)
        label_offset = noun_count if name == "verb" else 0
        label_count = verb_count if name == "verb" else noun_count
        train_ds = remap_labels_for_role(train_ds, label_offset, label_count)
        test_ds = remap_labels_for_role(test_ds, label_offset, label_count)

        # Analytic probe
        analytic_probe = LinearProbe.from_tpencoder(
            tpencoder=tpe,
            encoder=None,
            role_id=name_to_roleidx[name],
            regularization=regularization if regularization in ('l2', 'atol', 'topk') else 'l2',
            l2_lambda=reg_param if regularization == 'l2' else None,
            atol=reg_param if regularization == 'atol' else None,
            topk=int(reg_param) if regularization == 'topk' and reg_param is not None and not np.isnan(reg_param) else None,
            role_unbinding=role_unbinding,
            role_pinv_regularization=role_pinv_regularization,
            role_pinv_l2_lambda=role_pinv_l2_lambda,
            role_pinv_atol=role_pinv_atol,
            role_pinv_topk=role_pinv_topk,
            embedding_model_name=embedding_model_name,
        )
        classifier = analytic_probe.classifier[-1]
        max_label = label_offset + label_count
        if max_label > classifier.out_features:
            raise ValueError("Requested label slice exceeds probe output dimension.")
        # Shrink the analytic head to the role-specific label slice.
        new_layer = torch.nn.Linear(
            classifier.in_features,
            label_count,
            dtype=classifier.weight.dtype,
        )
        with torch.no_grad():
            new_layer.weight.copy_(classifier.weight[label_offset:max_label, :])
            new_layer.bias.copy_(classifier.bias[label_offset:max_label])
        analytic_probe.classifier[-1] = new_layer
        analytic_probe.config.num_labels = label_count
        analytic_trainer = Trainer(
            model=analytic_probe,
            args=analytic_training_args,
            eval_dataset=test_ds,
            data_collator=collate_embeddings,
            compute_metrics=compute_metrics,
        )
        analytic_metrics = analytic_trainer.evaluate()
        log_sanity_examples(
            analytic_trainer,
            test_ds,
            name,
            role_assigner,
            title="Analytic probe examples",
            label_offset=label_offset,
        )

        # Trainable probe (use cache if available or skip flag set)
        trained_acc_value: Optional[float] = None
        if name in trained_cache:
            trained_acc_value = float(trained_cache[name])
        elif not skip_trainable_probe:
            probe_config = LinearProbeConfig(
                encoder_model_type='sentence-transformers',
                encoder_hidden_size=embedding_dim,
                num_labels=label_count,
            )
            trainable_probe = LinearProbe(probe_config, None)
            trainable_trainer = Trainer(
                model=trainable_probe,
                args=trainable_training_args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                data_collator=collate_embeddings,
                compute_metrics=compute_metrics,
            )
            trainable_trainer.train()
            trained_metrics = trainable_trainer.evaluate()
            log_sanity_examples(
                trainable_trainer,
                test_ds,
                name,
                role_assigner,
                title="Trainable probe examples",
                label_offset=label_offset,
            )
            trained_acc_value = float(trained_metrics.get('eval_accuracy', float('nan')))
            # Update cache on disk
            trained_cache[name] = trained_acc_value
            try:
                with open(cache_trained_results_path, 'w') as f:
                    json.dump(trained_cache, f, indent=2)
            except Exception:
                pass
        else:
            # skip_trainable_probe True and no cache -> leave as NaN
            trained_acc_value = float('nan')

        results.append({
            'role_name': name,
            'analytic_accuracy': float(analytic_metrics.get('eval_accuracy', float('nan'))),
            'trained_accuracy': trained_acc_value,
        })

    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results, results_dir, tag)
    # Also print in readable order
    for r in results:
        print(f"Role {r['role_name']}: Analytic={r['analytic_accuracy']:.4f}, Trained={r['trained_accuracy']:.4f}")
    return results


if __name__ == '__main__':
    gin.external_configurable(TrainingArguments)
    parse_args_for_gin()
    main()  # type: ignore[arg-type, call-arg]
