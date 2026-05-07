"""Configuration generators for integration tests."""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional


class ConfigGenerator:
    """Generates test configurations for different scripts and parameter combinations."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = Path(temp_dir)

    def generate_digit_config(
        self,
        task: str = "reverse",
        vocab_size: int = 5,
        seq_length: Tuple[int, int] = (1, 3),
        prefix: Optional[str] = None,
        sample_sizes: Tuple[int, int, int] = (10, 5, 5),
    ) -> Path:
        """Generate a digit generation config."""
        if prefix is None:
            prefix = f"toy_{task}"

        min_seq, max_seq = seq_length
        n_train, n_valid, n_test = sample_sizes

        config_content = f"""
generate_examples.min_seq_length = {min_seq}
generate_examples.max_seq_length = {max_seq}
generate_examples.vocab_size = {vocab_size}

transform.task = "{task}"

main.random_seed = 42
main.data_dir = "{self.temp_dir}/digits/data/"
main.prefix = "{prefix}"
main.n_train = {n_train}
main.n_valid = {n_valid}
main.n_test = {n_test}
"""
        config_path = self.temp_dir / f"{prefix}_gen.gin"
        config_path.write_text(config_content)
        return config_path

    def generate_training_config(
        self,
        architecture: str = "GRU",
        role_scheme: str = "r2l",
        data_prefix: str = "toy_reverse",
        model_sizes: Optional[Dict[str, int]] = None,
    ) -> Path:
        """Generate a training config with specified architecture and role scheme."""
        if model_sizes is None:
            model_sizes = {
                "embedding_size": 8,
                "hidden_size": 16,
                "filler_dim": 8,
                "role_dim": 8,
            }

        # Calculate n_roles based on role scheme
        if role_scheme == "bow":
            n_roles = 3  # 2 + 1 for padding
        else:
            n_roles = 6  # max_seq_length(3) + 2 + 1 for padding

        # For LSTM architecture, hidden and cell states are concatenated, so TPE needs 2x hidden_size
        tpe_hidden_size = (
            model_sizes["hidden_size"] * 2
            if architecture == "LSTM"
            else model_sizes["hidden_size"]
        )

        config_content = f"""
seq2seq_init.encoder_config = {{
    'architecture': '{architecture}',
    'embedding_size': {model_sizes['embedding_size']},
    'hidden_size': {model_sizes['hidden_size']},
    'n_layers': 1,
    'dropout': 0
}}

seq2seq_init.decoder_config = {{
    'architecture': '{architecture}',
    'embedding_size': {model_sizes['embedding_size']},
    'hidden_size': {model_sizes['hidden_size']},
    'n_layers': 1,
    'dropout': 0
}}

main.seq2seq_training_args = @seq2seq/Seq2SeqTrainingArguments()
seq2seq/Seq2SeqTrainingArguments.output_dir = '{self.temp_dir}/checkpoints/seq2seq'
seq2seq/Seq2SeqTrainingArguments.save_total_limit = 1
seq2seq/Seq2SeqTrainingArguments.num_train_epochs = 1
seq2seq/Seq2SeqTrainingArguments.learning_rate = 0.01
seq2seq/Seq2SeqTrainingArguments.lr_scheduler_type = 'constant'
seq2seq/Seq2SeqTrainingArguments.per_device_train_batch_size = 2
seq2seq/Seq2SeqTrainingArguments.per_device_eval_batch_size = 2
seq2seq/Seq2SeqTrainingArguments.warmup_steps = 0
seq2seq/Seq2SeqTrainingArguments.weight_decay = 0.01
seq2seq/Seq2SeqTrainingArguments.eval_strategy = 'no'
seq2seq/Seq2SeqTrainingArguments.save_strategy = 'no'
seq2seq/Seq2SeqTrainingArguments.predict_with_generate = True
seq2seq/Seq2SeqTrainingArguments.generation_max_length = 5
seq2seq/Seq2SeqTrainingArguments.remove_unused_columns = False
seq2seq/Seq2SeqTrainingArguments.report_to = 'none'

main.tpe_config = {{
    'hidden_size': {tpe_hidden_size},
    'n_roles': {n_roles},
    'filler_dim': {model_sizes['filler_dim']},
    'role_dim': {model_sizes['role_dim']},
    'role_pad_token_id': 0,
    'role_scheme': '{role_scheme}',
}}

main.tpe_training_args = @tpe/TrainingArguments()
tpe/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/tpe'
tpe/TrainingArguments.save_total_limit = 1
tpe/TrainingArguments.num_train_epochs = 1
tpe/TrainingArguments.learning_rate = 0.01
tpe/TrainingArguments.lr_scheduler_type = 'constant'
tpe/TrainingArguments.per_device_train_batch_size = 2
tpe/TrainingArguments.per_device_eval_batch_size = 2
tpe/TrainingArguments.warmup_steps = 0
tpe/TrainingArguments.weight_decay = 0.01
tpe/TrainingArguments.eval_strategy = 'no'
tpe/TrainingArguments.save_strategy = 'no'
tpe/TrainingArguments.remove_unused_columns = False
tpe/TrainingArguments.metric_for_best_model = 'eval_loss'
tpe/TrainingArguments.greater_is_better = False
tpe/TrainingArguments.load_best_model_at_end = False
tpe/TrainingArguments.report_to = 'none'

# Evaluation configuration for TPE substitution accuracy
evaluate_tpe.tpe_training_args = @tpe_eval/Seq2SeqTrainingArguments()
tpe_eval/Seq2SeqTrainingArguments.output_dir = '{self.temp_dir}/checkpoints/tpe_eval'
tpe_eval/Seq2SeqTrainingArguments.per_device_eval_batch_size = 2
tpe_eval/Seq2SeqTrainingArguments.remove_unused_columns = False
tpe_eval/Seq2SeqTrainingArguments.predict_with_generate = True
tpe_eval/Seq2SeqTrainingArguments.generation_max_length = 5
tpe_eval/Seq2SeqTrainingArguments.report_to = 'none'

main.data_paths_dict = {{
    'train': '{self.temp_dir}/digits/data/{data_prefix}.train',
    'valid': '{self.temp_dir}/digits/data/{data_prefix}.valid',
    'test': '{self.temp_dir}/digits/data/{data_prefix}.test'
}}
"""
        config_path = self.temp_dir / f"train_{architecture.lower()}_{role_scheme}.gin"
        config_path.write_text(config_content)
        return config_path

    def generate_invert_config(
        self,
        regularization: str = "l2",
        reg_param: float = 0.1,
        role_sweep_range: Tuple[int, int] = (-3, 0),
        data_prefix: str = "toy_reverse",
    ) -> Path:
        """Generate a TPR inversion config."""
        start_role, end_role = role_sweep_range
        role_list = list(range(start_role, end_role))

        config_content = f"""
main.data_paths_dict = {{
    'train': '{self.temp_dir}/digits/data/{data_prefix}.train',
    'valid': '{self.temp_dir}/digits/data/{data_prefix}.valid',
    'test': '{self.temp_dir}/digits/data/{data_prefix}.test'
}}
main.seq2seq_path = '{self.temp_dir}/checkpoints/seq2seq'
main.tpe_path = '{self.temp_dir}/checkpoints/tpe'
main.regularization = '{regularization}'
main.reg_param = {reg_param}
main.role_sweep_range = {role_list}
main.results_dir = '{self.temp_dir}/tmp'

evaluate_probes_for_role.analytic_training_args = @analytic/TrainingArguments()

analytic/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/analytic'
analytic/TrainingArguments.per_device_eval_batch_size = 2
analytic/TrainingArguments.remove_unused_columns = False
analytic/TrainingArguments.report_to = 'none'

evaluate_probes_for_role.trainable_training_args = @trainable/TrainingArguments()

trainable/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/trainable_probe'
trainable/TrainingArguments.save_total_limit = 0
trainable/TrainingArguments.per_device_train_batch_size = 2
trainable/TrainingArguments.per_device_eval_batch_size = 2
trainable/TrainingArguments.num_train_epochs = 1
trainable/TrainingArguments.learning_rate = 1e-2
trainable/TrainingArguments.save_strategy = 'no'
trainable/TrainingArguments.remove_unused_columns = False
trainable/TrainingArguments.report_to = 'none'
"""
        config_path = self.temp_dir / f"invert_{regularization}.gin"
        config_path.write_text(config_content)
        return config_path

    def generate_sentence_config(
        self,
        role_scheme: str = "svo",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        skip_components: Tuple[bool, bool, bool] = (False, True, True),
        random_seed: int | None = None,
    ) -> Path:
        """Generate a sentence training config."""
        skip_tpe, skip_trainable, skip_analytic = skip_components

        # Determine n_roles based on role scheme
        if role_scheme == "svo":
            n_roles = 3
        elif role_scheme == "bow":
            n_roles = 2
        else:  # pair
            n_roles = 5

        config_content = f"""
main.sentences_path = '{self.temp_dir}/sentences/'
main.embedding_model_name = '{embedding_model}'
main.embedding_cache_path = '{self.temp_dir}/sentences/'
main.role_scheme = '{role_scheme}'
main.skip_tpe = {str(skip_tpe)}
main.skip_trainable_probe = {str(skip_trainable)}
main.skip_analytic_probe = {str(skip_analytic)}

"""
        if random_seed is not None:
            config_content += f"\nmain.random_seed = {random_seed}\n"
        config_content += f"""

main.tpe_config = {{
    'filler_dim': 4,
    'role_dim': 4,
    'n_roles': {n_roles},
    'hidden_size': 8,
}}

main.tpe_training_args = @tpe/TrainingArguments()

tpe/TrainingArguments.num_train_epochs = 1
tpe/TrainingArguments.per_device_train_batch_size = 2
tpe/TrainingArguments.per_device_eval_batch_size = 2
tpe/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/tpe'
tpe/TrainingArguments.save_total_limit = 0
tpe/TrainingArguments.eval_strategy = 'no'
tpe/TrainingArguments.save_strategy = 'no'
tpe/TrainingArguments.report_to = 'none'
"""

        # Add probe configs if needed
        if not skip_trainable:
            config_content += f"""
main.probe_training_args = @trainable_probe/TrainingArguments()
trainable_probe/TrainingArguments.num_train_epochs = 1
trainable_probe/TrainingArguments.per_device_train_batch_size = 2
trainable_probe/TrainingArguments.per_device_eval_batch_size = 2
trainable_probe/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/probe'
trainable_probe/TrainingArguments.save_total_limit = 0
trainable_probe/TrainingArguments.save_strategy = 'no'
trainable_probe/TrainingArguments.report_to = 'none'
"""

        if not skip_analytic:
            config_content += f"""
main.analytic_training_args = @analytic_probe_eval/TrainingArguments()
analytic_probe_eval/TrainingArguments.per_device_eval_batch_size = 2
analytic_probe_eval/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/analytic'
analytic_probe_eval/TrainingArguments.report_to = 'none'
"""

        config_path = self.temp_dir / f"sentences_{role_scheme}.gin"
        config_path.write_text(config_content)
        return config_path

    def generate_paired_sentence_config(
        self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", role_for_probe: int = 0
    ) -> Path:
        """Generate a paired sentence training config."""
        config_content = f"""
main.sentences_path = '{self.temp_dir}/pair_sentences/'
main.dataset_loader = @load_paired_sentences
main.role_scheme = 'pair'
main.embedding_model_name = '{embedding_model}'
main.embedding_cache_path = '{self.temp_dir}/pair_sentences/'
main.skip_tpe = False
main.skip_trainable_probe = True
main.skip_analytic_probe = True
main.role_for_probe = {role_for_probe}

main.tpe_config = {{
    'filler_dim': 4,
    'role_dim': 4,
    'n_roles': 5,
    'hidden_size': 8,
}}

main.tpe_training_args = @tpe/TrainingArguments()

tpe/TrainingArguments.num_train_epochs = 1
tpe/TrainingArguments.per_device_train_batch_size = 2
tpe/TrainingArguments.per_device_eval_batch_size = 2
tpe/TrainingArguments.output_dir = '{self.temp_dir}/checkpoints/tpe'
tpe/TrainingArguments.save_total_limit = 0
tpe/TrainingArguments.eval_strategy = 'no'
tpe/TrainingArguments.save_strategy = 'no'
tpe/TrainingArguments.report_to = 'none'
"""
        config_path = self.temp_dir / "paired_sentences.gin"
        config_path.write_text(config_content)
        return config_path


# Configuration parameter sets for comprehensive testing
COMPREHENSIVE_TEST_PARAMS = {
    "digit_tasks": ["copy", "reverse", "sort_ascending", "interleave"],
    "architectures": ["GRU", "LSTM", "RNN"],
    "role_schemes": ["l2r", "r2l", "bow"],
    "regularization_methods": [("l2", 0.1), ("atol", 1e-3), ("topk", 5)],
    "sentence_role_schemes": ["svo", "bow"],
    "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"],  # Keep minimal for tests
    "skip_combinations": [
        (False, True, True),  # Only TPE
        (False, False, True),  # TPE + trainable probe
        (False, True, False),  # TPE + analytic probe
    ],
}
