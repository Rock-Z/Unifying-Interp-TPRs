"""Integration tests for important scripts with toy data and configs."""

import pytest
import tempfile
import os
import sys
import subprocess
from pathlib import Path
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import our config generator
from config_generator import ConfigGenerator, COMPREHENSIVE_TEST_PARAMS

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def config_gen(temp_dir):
    """Create a config generator instance."""
    return ConfigGenerator(temp_dir)


def run_script(script_path, config_path, cwd=None):
    """Helper function to run a script with uv run."""
    if cwd is None:
        cwd = Path(__file__).resolve().parents[1]
    
    cmd = ["uv", "run", str(script_path), str(config_path)]
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180  # 3 minutes timeout
    )
    
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result


# ===== DEFAULT/BASIC TESTS (run by default) =====

class TestBasicDigitsWorkflow:
    """Basic tests for digits workflow - runs by default."""

    def test_generate_digits_basic(self, temp_dir, config_gen):
        """Test basic digit generation functionality."""
        config_path = config_gen.generate_digit_config()
        
        # Run the script
        result = run_script("src/generate_digits.py", config_path)
        
        # Check that data files were created
        data_dir = temp_dir / "digits" / "data"
        assert (data_dir / "toy_reverse.train").exists()
        assert (data_dir / "toy_reverse.valid").exists()
        assert (data_dir / "toy_reverse.test").exists()
        assert (data_dir / "toy_reverse.dataset_creation_args.gin").exists()
        
        # Verify basic content structure
        train_file = data_dir / "toy_reverse.train"
        assert train_file.stat().st_size > 0
        
        content = train_file.read_text()
        lines = content.strip().split('\n')
        assert lines[0] == "input_seq\ttarget_seq"
        assert len(lines) >= 10  # Should have expected number of examples

    @pytest.mark.slow
    def test_train_basic(self, temp_dir, config_gen):
        """Test basic training functionality."""
        # First generate data
        self.test_generate_digits_basic(temp_dir, config_gen)
        
        # Generate training config
        train_config = config_gen.generate_training_config()
        
        # Run training script
        result = run_script("src/train.py", train_config)
        
        # Check that model checkpoints were created
        seq2seq_dir = temp_dir / "checkpoints" / "seq2seq"
        tpe_dir = temp_dir / "checkpoints" / "tpe"
        
        assert seq2seq_dir.exists()
        assert tpe_dir.exists()

    @pytest.mark.slow
    def test_invert_tpr_basic(self, temp_dir, config_gen):
        """Test basic TPR inversion functionality."""
        # First generate data and train models
        self.test_train_basic(temp_dir, config_gen)
        
        # Generate invert config
        invert_config = config_gen.generate_invert_config()
        
        # Run inversion script
        result = run_script("src/invert_tpr.py", invert_config)
        
        # Should complete without error
        assert result.returncode == 0


class TestBasicSentenceWorkflow:
    """Basic tests for sentence workflow - runs by default."""

    def test_generate_sentences_basic(self, temp_dir):
        """Test basic sentence generation."""
        script_path = "src/generate_sentences.py"
        
        # Create output directory
        sentences_dir = temp_dir / "sentences"
        sentences_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the script with custom arguments
        cwd = Path(__file__).resolve().parents[1]
        cmd = ["uv", "run", str(script_path), "--prefix", str(sentences_dir / "data")]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0
        
        # Check that data files were created
        assert (sentences_dir / "data.train").exists()
        assert (sentences_dir / "data.valid").exists()
        assert (sentences_dir / "data.test").exists()
        assert (sentences_dir / "data.nouns").exists()
        assert (sentences_dir / "data.verbs").exists()

    def test_generate_pair_sentences_basic(self, temp_dir):
        """Test basic paired sentence generation."""
        script_path = "src/generate_pair_sentences.py"
        
        # Create output directory
        pair_sentences_dir = temp_dir / "pair_sentences"
        pair_sentences_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the script with custom arguments
        cwd = Path(__file__).resolve().parents[1]
        cmd = ["uv", "run", str(script_path), "--prefix", str(pair_sentences_dir / "data"), "--max_rows", "50"]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0
        
        # Check that data files were created
        assert (pair_sentences_dir / "data.train").exists()
        assert (pair_sentences_dir / "data.valid").exists()
        assert (pair_sentences_dir / "data.test").exists()

    @pytest.mark.slow
    def test_train_sentences_basic(self, temp_dir, config_gen):
        """Test basic sentence training."""
        # First generate sentence data
        self.test_generate_sentences_basic(temp_dir)
        
        # Generate sentence training config
        sentence_config = config_gen.generate_sentence_config()
        
        # Run training script
        result = run_script("src/train_sentences.py", sentence_config)
        
        # Check that checkpoint directory was created
        checkpoint_dir = temp_dir / "checkpoints" / "tpe"
        assert checkpoint_dir.exists()

    @pytest.mark.slow  
    def test_train_paired_sentences_basic(self, temp_dir, config_gen):
        """Test basic paired sentence training."""
        # First generate paired sentence data
        self.test_generate_pair_sentences_basic(temp_dir)
        
        # Generate paired sentence config
        paired_config = config_gen.generate_paired_sentence_config()
        
        # Run training script
        result = run_script("src/train_sentences.py", paired_config)
        
        # Check that checkpoint directory was created
        checkpoint_dir = temp_dir / "checkpoints" / "tpe"
        assert checkpoint_dir.exists()


class TestBasicIntegrationWorkflows:
    """Basic end-to-end workflow tests - runs by default."""

    @pytest.mark.slow
    def test_digits_full_workflow_basic(self, temp_dir, config_gen):
        """Test the complete basic digits workflow: generate -> train -> invert."""
        # Step 1: Generate data
        digit_config = config_gen.generate_digit_config()
        run_script("src/generate_digits.py", digit_config)
        
        # Step 2: Train models
        train_config = config_gen.generate_training_config()
        run_script("src/train.py", train_config)
        
        # Step 3: Invert TPR
        invert_config = config_gen.generate_invert_config()
        run_script("src/invert_tpr.py", invert_config)

    @pytest.mark.slow
    def test_sentences_full_workflow_basic(self, temp_dir, config_gen):
        """Test the complete basic sentences workflow: generate -> train."""
        # Step 1: Generate sentence data
        sentences_dir = temp_dir / "sentences"
        sentences_dir.mkdir(parents=True, exist_ok=True)
        
        cwd = Path(__file__).resolve().parents[1]
        cmd = ["uv", "run", "src/generate_sentences.py", "--prefix", str(sentences_dir / "data")]
        subprocess.run(cmd, cwd=cwd, check=True, timeout=60)
        
        # Step 2: Train models
        sentence_config = config_gen.generate_sentence_config()
        run_script("src/train_sentences.py", sentence_config)


# ===== COMPREHENSIVE TESTS (run only with uv run -m pytest -m comprehensive) =====

@pytest.mark.comprehensive
@pytest.mark.slow
class TestDigitTaskVariations:
    """Test different digit transformation tasks - comprehensive mode only."""

    @pytest.mark.parametrize("task", COMPREHENSIVE_TEST_PARAMS['digit_tasks'])
    def test_different_digit_tasks(self, temp_dir, config_gen, task):
        """Test digit generation with different transformation tasks."""
        config_path = config_gen.generate_digit_config(task=task, sample_sizes=(8, 4, 4))
        
        # Run the script
        result = run_script("src/generate_digits.py", config_path)
        
        # Check that data files were created with correct naming
        data_dir = temp_dir / "digits" / "data"
        assert (data_dir / f"toy_{task}.train").exists()
        assert (data_dir / f"toy_{task}.valid").exists()
        assert (data_dir / f"toy_{task}.test").exists()
        
        # Verify content structure
        train_file = data_dir / f"toy_{task}.train"
        assert train_file.stat().st_size > 0
        
        content = train_file.read_text()
        lines = content.strip().split('\n')
        assert lines[0] == "input_seq\ttarget_seq"
        assert len(lines) >= 8


@pytest.mark.comprehensive
@pytest.mark.slow
class TestArchitectureRoleVariations:
    """Test different architecture and role scheme combinations - comprehensive mode only."""

    @pytest.mark.parametrize("architecture", COMPREHENSIVE_TEST_PARAMS['architectures'])
    @pytest.mark.parametrize("role_scheme", COMPREHENSIVE_TEST_PARAMS['role_schemes'])
    def test_architecture_role_combinations(self, temp_dir, config_gen, architecture, role_scheme):
        """Test training with different architecture and role scheme combinations."""
        # Generate basic data first
        digit_config = config_gen.generate_digit_config()
        run_script("src/generate_digits.py", digit_config)
        
        # Generate training config with specific architecture and role scheme
        train_config = config_gen.generate_training_config(
            architecture=architecture, 
            role_scheme=role_scheme
        )
        
        # Run training script
        result = run_script("src/train.py", train_config)
        
        # Verify checkpoints created
        seq2seq_dir = temp_dir / "checkpoints" / "seq2seq"
        tpe_dir = temp_dir / "checkpoints" / "tpe"
        
        assert seq2seq_dir.exists(), f"Seq2seq checkpoint missing for {architecture}-{role_scheme}"
        assert tpe_dir.exists(), f"TPE checkpoint missing for {architecture}-{role_scheme}"


@pytest.mark.comprehensive
@pytest.mark.slow
class TestRegularizationVariations:
    """Test different regularization methods - comprehensive mode only."""

    @pytest.mark.parametrize("reg_type,reg_param", COMPREHENSIVE_TEST_PARAMS['regularization_methods'])
    def test_different_regularization_methods(self, temp_dir, config_gen, reg_type, reg_param):
        """Test TPR inversion with different regularization methods."""
        # Generate data and train models first
        digit_config = config_gen.generate_digit_config()
        run_script("src/generate_digits.py", digit_config)
        
        train_config = config_gen.generate_training_config()
        run_script("src/train.py", train_config)
        
        # Generate invert config with specific regularization
        invert_config = config_gen.generate_invert_config(
            regularization=reg_type, 
            reg_param=reg_param,
            role_sweep_range=(-2, 0)  # Smaller range for faster testing
        )
        
        # Run inversion script
        result = run_script("src/invert_tpr.py", invert_config)
        
        # Should complete without error for all regularization types
        assert result.returncode == 0, f"Inversion failed for regularization {reg_type}"


@pytest.mark.comprehensive
@pytest.mark.slow
class TestSentenceRoleSchemeVariations:
    """Test different sentence role schemes - comprehensive mode only."""

    @pytest.mark.parametrize("role_scheme", COMPREHENSIVE_TEST_PARAMS['sentence_role_schemes'])
    def test_sentence_role_schemes(self, temp_dir, config_gen, role_scheme):
        """Test sentence training with different role schemes."""
        # Generate sentence data
        sentences_dir = temp_dir / "sentences"
        sentences_dir.mkdir(parents=True, exist_ok=True)
        
        cwd = Path(__file__).resolve().parents[1]
        cmd = ["uv", "run", "src/generate_sentences.py", "--prefix", str(sentences_dir / "data")]
        subprocess.run(cmd, cwd=cwd, check=True, timeout=60)
        
        # Generate sentence config with specific role scheme
        sentence_config = config_gen.generate_sentence_config(role_scheme=role_scheme)
        
        # Run training script
        result = run_script("src/train_sentences.py", sentence_config)
        
        # Check that checkpoint directory was created
        checkpoint_dir = temp_dir / "checkpoints" / "tpe"
        assert checkpoint_dir.exists(), f"TPE checkpoint missing for role scheme {role_scheme}"


@pytest.mark.comprehensive
@pytest.mark.slow
class TestProbeComponentVariations:
    """Test different probe component combinations - comprehensive mode only."""

    @pytest.mark.parametrize("skip_tpe,skip_trainable,skip_analytic", COMPREHENSIVE_TEST_PARAMS['skip_combinations'])
    def test_probe_skip_combinations(self, temp_dir, config_gen, skip_tpe, skip_trainable, skip_analytic):
        """Test different combinations of skipping probe components."""
        # Generate sentence data
        sentences_dir = temp_dir / "sentences"
        sentences_dir.mkdir(parents=True, exist_ok=True)
        
        cwd = Path(__file__).resolve().parents[1]
        cmd = ["uv", "run", "src/generate_sentences.py", "--prefix", str(sentences_dir / "data")]
        subprocess.run(cmd, cwd=cwd, check=True, timeout=60)
        
        # Generate config with specific skip combination
        sentence_config = config_gen.generate_sentence_config(
            skip_components=(skip_tpe, skip_trainable, skip_analytic)
        )
        
        # Run the script
        result = run_script("src/train_sentences.py", sentence_config)
        
        # Should complete successfully regardless of skip combination
        assert result.returncode == 0, f"Failed with skip combination: TPE={skip_tpe}, trainable={skip_trainable}, analytic={skip_analytic}"


@pytest.mark.comprehensive
@pytest.mark.slow
class TestConfigurationRobustness:
    """Test edge cases and robustness - comprehensive mode only."""

    def test_minimal_data_sizes(self, temp_dir, config_gen):
        """Test scripts work with very minimal data sizes."""
        # Test with minimal sequence lengths and vocabulary
        config_path = config_gen.generate_digit_config(
            task="copy",
            vocab_size=3,
            seq_length=(1, 2),
            prefix="minimal",
            sample_sizes=(4, 2, 2)
        )
        
        # Should work with minimal data
        result = run_script("src/generate_digits.py", config_path)
        
        # Verify files created
        data_dir = temp_dir / "digits" / "data"
        assert (data_dir / "minimal.train").exists()
        
        # Check content is reasonable
        with open(data_dir / "minimal.train", "r") as f:
            lines = f.readlines()
            assert len(lines) >= 2  # Header + at least some data

    def test_different_data_sizes_paired_sentences(self, temp_dir):
        """Test paired sentence generation with different sizes."""
        script_path = "src/generate_pair_sentences.py"
        
        for max_rows in [20, 50]:
            pair_sentences_dir = temp_dir / f"pair_sentences_{max_rows}"
            pair_sentences_dir.mkdir(parents=True, exist_ok=True)
            
            # Run the script with different max_rows
            cwd = Path(__file__).resolve().parents[1]
            cmd = ["uv", "run", str(script_path), 
                   "--prefix", str(pair_sentences_dir / "data"), 
                   "--max_rows", str(max_rows)]
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            assert result.returncode == 0, f"Failed with max_rows={max_rows}"
            
            # Check files were created
            assert (pair_sentences_dir / "data.train").exists()
            assert (pair_sentences_dir / "data.valid").exists()
            assert (pair_sentences_dir / "data.test").exists() 