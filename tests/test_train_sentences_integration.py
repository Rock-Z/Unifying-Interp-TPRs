"""
Integration tests for train_sentences.py script with TensorProductEncoderWithBackProjection.

Tests cover:
- Script execution with different model types
- Configuration validation
- Model type selection logic
- Dataset compatibility
- Error handling for invalid configurations
"""

import pytest
import torch
import tempfile
import shutil
import os
import sys
import json
import gin
from unittest.mock import patch, Mock, MagicMock
from transformers import TrainingArguments

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_sentences import main, add_labels_for_role, tpe_data_collator
from model import (
    TensorProductEncoderForPretraining,
    TensorProductEncoderWithDecodingLoss,
    TensorProductEncoderWithBackProjection,
    TensorProductEncoderConfig
)


class TestTrainSentencesIntegration:
    """Integration test suite for train_sentences.py with back projection support."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        # Create a minimal mock dataset
        mock_split = Mock()
        mock_split.column_names = ['filler_ids', 'role_ids', 'sentence']
        mock_split.__len__ = lambda self: 10
        
        # Create sample data for the dataset
        sample_data = {
            'filler_ids': [[1, 2, 0], [2, 1, 0], [0, 1, 2]] * 4,  # 12 samples
            'role_ids': [[0, 1, 2], [1, 0, 2], [2, 0, 1]] * 4,
            'sentence': [f'test sentence {i}' for i in range(12)],
            'target_embeddings': [torch.randn(1, 64).tolist() for _ in range(12)]
        }
        
        # Mock both integer indexing and column name indexing
        def mock_getitem(self, key):
            if isinstance(key, str):
                # Column access like dataset['role_ids']
                return sample_data[key]
            elif isinstance(key, int):
                # Row access like dataset[0]
                return {k: v[key] for k, v in sample_data.items()}
            else:
                return sample_data[key]
        
        mock_split.__getitem__ = mock_getitem
        mock_split.add_column = lambda name, data: mock_split
        mock_split.filter = lambda func: mock_split
        mock_split.rename_column = lambda old, new: mock_split
        mock_split.remove_columns = lambda cols: mock_split
        
        mock_dataset = {
            'train': mock_split,
            'valid': mock_split,
            'test': mock_split
        }
        
        return mock_dataset
    
    @pytest.fixture
    def mock_role_assigner(self):
        """Mock role assigner for testing."""
        # Use spec to limit what attributes the mock has
        mock_assigner = Mock(spec=['noun_filler2idx', 'verb_filler2idx', 'role2idx'])
        mock_assigner.noun_filler2idx = {'noun1': 1, 'noun2': 2}
        mock_assigner.verb_filler2idx = {'verb1': 3, 'verb2': 4}
        mock_assigner.role2idx = {'subject': 0, 'verb': 1, 'object': 2}
        return mock_assigner
    
    @pytest.fixture
    def basic_training_args(self, temp_dir):
        """Basic training arguments for testing."""
        return TrainingArguments(
            output_dir=temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy='no',
            save_strategy='no',
            remove_unused_columns=False,
            label_names=['target_embeddings']
        )
    
    def test_model_type_selection_back_projection(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test that TensorProductEncoderWithBackProjection is selected correctly."""
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings, \
             patch('train_sentences.Trainer') as mock_trainer_class:
            
            # Setup mocks
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            mock_trainer = Mock()
            mock_trainer.train.return_value = None
            mock_trainer.evaluate.return_value = {'eval_loss': 0.5}
            mock_trainer.model = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Test back projection model selection
            main(
                sentences_path='dummy_path',
                embedding_model_name='test_model',
                embedding_cache_path='dummy_cache',
                tpe_config={
                    'filler_dim': 32,
                    'role_dim': 4,
                    'has_linear_layer': True,
                    'back_projection_loss_weight': 1.0
                },
                tpe_training_args=TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    eval_strategy='no',
                    save_strategy='no',
                    remove_unused_columns=False,
                    label_names=['target_embeddings']
                ),
                skip_trainable_probe=True,
                skip_analytic_probe=True,
                tpe_back_projection=True,  # Enable back projection
                dataset_loader=mock_load
            )
            
            # Verify model was created and trained
            mock_trainer_class.assert_called()
            mock_trainer.train.assert_called_once()
            
            # Check that the model instance is of correct type
            trainer_call_args = mock_trainer_class.call_args
            model_instance = trainer_call_args[1]['model']
            assert isinstance(model_instance, TensorProductEncoderWithBackProjection)
    
    def test_model_type_selection_decoding_loss(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test that TensorProductEncoderWithDecodingLoss is selected correctly."""
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings, \
             patch('train_sentences.Trainer') as mock_trainer_class:
            
            # Setup mocks
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            mock_trainer = Mock()
            mock_trainer.train.return_value = None
            mock_trainer.evaluate.return_value = {'eval_loss': 0.5}
            mock_trainer.model = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Test decoding loss model selection
            main(
                sentences_path='dummy_path',
                embedding_model_name='test_model',
                embedding_cache_path='dummy_cache',
                tpe_config={
                    'filler_dim': 32,
                    'role_dim': 4,
                    'role_id': 0
                },
                tpe_training_args=TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    eval_strategy='no',
                    save_strategy='no',
                    remove_unused_columns=False,
                    label_names=['target_embeddings']
                ),
                skip_trainable_probe=True,
                skip_analytic_probe=True,
                tpe_decoding_loss=True,  # Enable decoding loss
                dataset_loader=mock_load
            )
            
            # Check that the model instance is of correct type
            trainer_call_args = mock_trainer_class.call_args
            model_instance = trainer_call_args[1]['model']
            assert isinstance(model_instance, TensorProductEncoderWithDecodingLoss)
    
    def test_model_type_selection_regular_pretraining(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test that TensorProductEncoderForPretraining is selected by default."""
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings, \
             patch('train_sentences.Trainer') as mock_trainer_class:
            
            # Setup mocks
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            mock_trainer = Mock()
            mock_trainer.train.return_value = None
            mock_trainer.evaluate.return_value = {'eval_loss': 0.5}
            mock_trainer.model = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Test regular pretraining model selection (default)
            main(
                sentences_path='dummy_path',
                embedding_model_name='test_model',
                embedding_cache_path='dummy_cache',
                tpe_config={
                    'filler_dim': 32,
                    'role_dim': 4
                },
                tpe_training_args=TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    eval_strategy='no',
                    save_strategy='no',
                    remove_unused_columns=False,
                    label_names=['target_embeddings']
                ),
                skip_trainable_probe=True,
                skip_analytic_probe=True,
                tpe_decoding_loss=False,
                tpe_back_projection=False,
                dataset_loader=mock_load
            )
            
            # Check that the model instance is of correct type
            trainer_call_args = mock_trainer_class.call_args
            model_instance = trainer_call_args[1]['model']
            assert isinstance(model_instance, TensorProductEncoderForPretraining)
    
    def test_mutual_exclusion_error(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test that using both decoding loss and back projection raises error."""
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings:
            
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            # Test that mutual exclusion is enforced
            with pytest.raises(ValueError, match="Cannot use both tpe_decoding_loss and tpe_back_projection"):
                main(
                    sentences_path='dummy_path',
                    embedding_model_name='test_model',
                    embedding_cache_path='dummy_cache',
                    tpe_config={'filler_dim': 32, 'role_dim': 4},
                    tpe_training_args=TrainingArguments(
                        output_dir=temp_dir,
                        num_train_epochs=1,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        eval_strategy='no',
                        save_strategy='no'
                    ),
                    tpe_decoding_loss=True,
                    tpe_back_projection=True,  # Both enabled - should raise error
                    dataset_loader=mock_load
                )
    
    def test_back_projection_requires_linear_layer(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test that back projection automatically enables linear layer."""
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings, \
             patch('train_sentences.Trainer') as mock_trainer_class:
            
            # Setup mocks
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            mock_trainer = Mock()
            mock_trainer.train.return_value = None
            mock_trainer.evaluate.return_value = {'eval_loss': 0.5}
            mock_trainer.model = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Test with has_linear_layer=False initially
            main(
                sentences_path='dummy_path',
                embedding_model_name='test_model',
                embedding_cache_path='dummy_cache',
                tpe_config={
                    'filler_dim': 32,
                    'role_dim': 4,
                    'has_linear_layer': False  # This should be overridden
                },
                tpe_training_args=TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    eval_strategy='no',
                    save_strategy='no',
                    remove_unused_columns=False,
                    label_names=['target_embeddings']
                ),
                skip_trainable_probe=True,
                skip_analytic_probe=True,
                tpe_back_projection=True,
                dataset_loader=mock_load
            )
            
            # Check that the model was created successfully (should not raise error)
            trainer_call_args = mock_trainer_class.call_args
            model_instance = trainer_call_args[1]['model']
            assert isinstance(model_instance, TensorProductEncoderWithBackProjection)
            assert model_instance.encoder.output_layer is not None
    
    def test_skip_tpe_loading_back_projection(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test loading saved TensorProductEncoderWithBackProjection model."""
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings, \
             patch('train_sentences.TensorProductEncoderWithBackProjection.from_pretrained') as mock_load_model:
            
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            mock_model = Mock(spec=TensorProductEncoderWithBackProjection)
            mock_load_model.return_value = mock_model
            
            # Create dummy eval results file
            eval_results_path = os.path.join(temp_dir, 'eval_results_tpe.json')
            with open(eval_results_path, 'w') as f:
                json.dump({'eval_loss': 0.5}, f)
            
            main(
                sentences_path='dummy_path',
                embedding_model_name='test_model',
                embedding_cache_path='dummy_cache',
                tpe_config={'filler_dim': 32, 'role_dim': 4},
                tpe_training_args=TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    eval_strategy='no',
                    save_strategy='no'
                ),
                skip_tpe=True,  # Skip training, load model
                skip_trainable_probe=True,
                skip_analytic_probe=True,
                tpe_back_projection=True,
                dataset_loader=mock_load
            )
            
            # Verify correct model type was loaded
            expected_path = os.path.join(temp_dir, 'best_model')
            mock_load_model.assert_called_once_with(expected_path)
    
    def test_data_collator_for_decoding_loss(self):
        """Test custom data collator for decoding loss."""
        # Sample batch data
        features = [
            {
                'filler_ids': [1, 2, 0],
                'role_ids': [0, 1, 2],
                'sentence': 'test sentence 1',
                'labels': 5,
                'other_field': 0.5
            },
            {
                'filler_ids': [2, 1, 0],
                'role_ids': [1, 0, 2],
                'sentence': 'test sentence 2',
                'labels': 3,
                'other_field': 0.7
            }
        ]
        
        batch = tpe_data_collator(features)
        
        # Check that tensors are properly created
        assert torch.is_tensor(batch['filler_ids'])
        assert torch.is_tensor(batch['role_ids'])
        assert torch.is_tensor(batch['labels'])
        assert torch.is_tensor(batch['probe_labels'])
        assert torch.is_tensor(batch['other_field'])
        
        # Check that sentences remain as strings
        assert isinstance(batch['sentence'], list)
        assert batch['sentence'][0] == 'test sentence 1'
        
        # Check that probe_labels is a copy of labels
        assert torch.equal(batch['labels'], batch['probe_labels'])
    
    def test_data_collator_missing_labels_error(self):
        """Test that data collator raises error when labels are missing."""
        features = [
            {
                'filler_ids': [1, 2, 0],
                'role_ids': [0, 1, 2],
                'sentence': 'test sentence 1'
                # Missing 'labels'
            }
        ]
        
        with pytest.raises(ValueError, match="Expected 'labels' in batch"):
            tpe_data_collator(features)
    
    def test_add_labels_for_role_function(self):
        """Test add_labels_for_role function."""
        # Test case 1: No existing labels column
        mock_split = Mock()
        mock_split.column_names = ['role_ids', 'filler_ids']
        mock_split.remove_columns.return_value = mock_split
        mock_split.add_column.return_value = mock_split
        
        # Sample data with ragged arrays (different lengths)
        role_ids_data = [[0, 1, 2], [1, 2, 0], [0, 2]]
        filler_ids_data = [[10, 20, 30], [15, 25, 35], [12, 22]]
        
        # Mock indexing
        mock_split.__getitem__ = lambda self, key: {
            'role_ids': role_ids_data,
            'filler_ids': filler_ids_data
        }[key]
        mock_split.__len__ = lambda: len(role_ids_data)
        
        # Test function
        result = add_labels_for_role(mock_split, role_id=1, allow_missing=True)
        
        # Verify add_column was called (remove_columns should not be called since no "labels" column)
        mock_split.remove_columns.assert_not_called()
        mock_split.add_column.assert_called_once()
        
        # Check the labels that would be added
        call_args = mock_split.add_column.call_args
        assert call_args[0][0] == "labels"  # First argument should be "labels"
        expected_labels = [20, 15, -1]  # [filler for role 1 in seq 0, filler for role 1 in seq 1, -1 for missing role 1 in seq 2]
        assert list(call_args[0][1]) == expected_labels
        
        # Test case 2: Existing labels column
        mock_split_with_labels = Mock()
        mock_split_with_labels.column_names = ['role_ids', 'filler_ids', 'labels']
        mock_split_with_labels.remove_columns.return_value = mock_split_with_labels
        mock_split_with_labels.add_column.return_value = mock_split_with_labels
        mock_split_with_labels.__getitem__ = lambda self, key: {
            'role_ids': role_ids_data,
            'filler_ids': filler_ids_data
        }[key]
        mock_split_with_labels.__len__ = lambda: len(role_ids_data)
        
        result = add_labels_for_role(mock_split_with_labels, role_id=0, allow_missing=True)
        
        # Verify both remove_columns and add_column were called
        mock_split_with_labels.remove_columns.assert_called_once_with("labels")
        mock_split_with_labels.add_column.assert_called_once()
    
    def test_config_validation_back_projection_variants(self, temp_dir, mock_dataset, mock_role_assigner):
        """Test that different back projection variants work correctly."""
        variants = ['tpr_to_gt', 'gt_to_projected']
        
        with patch('sentences.load_sentences') as mock_load, \
             patch('train_sentences.load_dataset_with_embeddings') as mock_load_embeddings, \
             patch('train_sentences.Trainer') as mock_trainer_class:
            
            mock_load.return_value = (mock_dataset, mock_role_assigner)
            mock_load_embeddings.return_value = (mock_dataset, 64)
            
            mock_trainer = Mock()
            mock_trainer.train.return_value = None
            mock_trainer.evaluate.return_value = {'eval_loss': 0.5}
            mock_trainer.model = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            for variant in variants:
                # Test each variant
                main(
                    sentences_path='dummy_path',
                    embedding_model_name='test_model',
                    embedding_cache_path='dummy_cache',
                    tpe_config={
                        'filler_dim': 32,
                        'role_dim': 4,
                        'has_linear_layer': True,
                        'back_projection_loss_variant': variant
                    },
                    tpe_training_args=TrainingArguments(
                        output_dir=os.path.join(temp_dir, variant),
                        num_train_epochs=1,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        eval_strategy='no',
                        save_strategy='no',
                        remove_unused_columns=False,
                        label_names=['target_embeddings']
                    ),
                    skip_trainable_probe=True,
                    skip_analytic_probe=True,
                    tpe_back_projection=True,
                    dataset_loader=mock_load
                )
                
                # Check model was created with correct variant
                trainer_call_args = mock_trainer_class.call_args
                model_instance = trainer_call_args[1]['model']
                assert model_instance.back_projection_loss_variant == variant


if __name__ == "__main__":
    pytest.main([__file__]) 