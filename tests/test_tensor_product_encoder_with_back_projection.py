"""
Unit tests for TensorProductEncoderWithBackProjection class.

Tests cover:
- Initialization and configuration
- Forward pass functionality
- Back-projection loss computation
- Error handling for invalid configurations
- Different loss variants
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    TensorProductEncoderWithBackProjection,
    TensorProductEncoderConfig,
    TensorProductEncoder
)


class TestTensorProductEncoderWithBackProjection:
    """Test suite for TensorProductEncoderWithBackProjection."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return TensorProductEncoderConfig(
            hidden_size=64,
            n_fillers=10,
            n_roles=3,
            filler_dim=32,
            role_dim=16,
            has_linear_layer=True,
            back_projection_loss_weight=1.0,
            back_projection_loss_variant='tpr_to_gt'
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        batch_size = 4
        seq_len = 5
        return {
            'filler_ids': torch.randint(0, 10, (batch_size, seq_len)),
            'role_ids': torch.randint(0, 3, (batch_size, seq_len)),
            'target_embeddings': torch.randn(batch_size, 1, 64)
        }
    
    def test_initialization_valid_config(self, basic_config):
        """Test successful initialization with valid config."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        
        assert model.back_projection_loss_weight == 1.0
        assert model.back_projection_loss_variant == 'tpr_to_gt'
        assert model.back_projection is not None
        assert model.encoder.output_layer is not None
        
        # Check back-projection layer dimensions
        assert model.back_projection.in_features == basic_config.hidden_size
        assert model.back_projection.out_features == basic_config.filler_dim * basic_config.role_dim
    
    def test_initialization_no_linear_layer_raises_error(self):
        """Test that initialization fails when has_linear_layer=False."""
        config = TensorProductEncoderConfig(
            hidden_size=64,
            n_fillers=10,
            n_roles=3,
            filler_dim=32,
            role_dim=16,
            has_linear_layer=False  # This should cause error
        )
        
        with pytest.raises(ValueError, match="requires the encoder to have an output_layer"):
            TensorProductEncoderWithBackProjection(config)
    
    def test_initialization_with_custom_config_values(self):
        """Test initialization with custom back-projection config values."""
        config = TensorProductEncoderConfig(
            hidden_size=64,
            n_fillers=10,
            n_roles=3,
            filler_dim=32,
            role_dim=16,
            has_linear_layer=True,
            back_projection_loss_weight=2.5,
            back_projection_loss_variant='gt_to_projected'
        )
        
        model = TensorProductEncoderWithBackProjection(config)
        assert model.back_projection_loss_weight == 2.5
        assert model.back_projection_loss_variant == 'gt_to_projected'
    
    def test_forward_training_mode_with_target_embeddings(self, basic_config, sample_data):
        """Test forward pass in training mode with target embeddings."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        model.train()
        
        outputs = model.forward(
            filler_ids=sample_data['filler_ids'],
            role_ids=sample_data['role_ids'],
            target_embeddings=sample_data['target_embeddings']
        )
        
        assert outputs.loss is not None
        assert outputs.encoder_hidden_states is not None
        assert outputs.encoder_hidden_states.shape == (4, 1, 64)
        
        # Loss should be a combination of reconstruction + back-projection
        assert torch.is_tensor(outputs.loss)
        assert outputs.loss.requires_grad
    
    def test_forward_training_mode_missing_inputs_raises_error(self, basic_config, sample_data):
        """Test that forward pass raises error when required inputs are missing in training mode."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        model.train()
        
        with pytest.raises(ValueError, match="either 'target_embeddings' or 'embedding_model_input_ids' must be provided"):
            model.forward(
                filler_ids=sample_data['filler_ids'],
                role_ids=sample_data['role_ids']
                # Missing target_embeddings and embedding_model_input_ids
            )
    
    def test_forward_eval_mode_no_back_projection_loss(self, basic_config, sample_data):
        """Test that back-projection loss is not computed in eval mode."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        model.eval()
        
        with patch.object(model, '_compute_back_projection_loss') as mock_compute:
            outputs = model.forward(
                filler_ids=sample_data['filler_ids'],
                role_ids=sample_data['role_ids'],
                target_embeddings=sample_data['target_embeddings']
            )
            
            # Back-projection loss should not be computed in eval mode
            mock_compute.assert_not_called()
            assert outputs.encoder_hidden_states is not None
    
    def test_back_projection_loss_tpr_to_gt_variant(self, basic_config, sample_data):
        """Test back-projection loss computation with 'tpr_to_gt' variant."""
        basic_config.back_projection_loss_variant = 'tpr_to_gt'
        model = TensorProductEncoderWithBackProjection(basic_config)
        model.train()
        
        # Set up cached embeddings
        target_embeddings = sample_data['target_embeddings']
        raw_tpr_binding = torch.randn(4, 32 * 16)  # filler_dim * role_dim
        
        model._cached_target_embeddings = target_embeddings
        model._cached_raw_tpr_binding = raw_tpr_binding.unsqueeze(1)  # Add seq_len dim
        
        loss = model._compute_back_projection_loss(target_embeddings)
        
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        assert loss.item() >= 0  # MSE loss should be non-negative
    
    def test_back_projection_loss_gt_to_projected_variant(self, basic_config, sample_data):
        """Test back-projection loss computation with 'gt_to_projected' variant."""
        basic_config.back_projection_loss_variant = 'gt_to_projected'
        model = TensorProductEncoderWithBackProjection(basic_config)
        model.train()
        
        # Set up cached embeddings
        target_embeddings = sample_data['target_embeddings']
        raw_tpr_binding = torch.randn(4, 32 * 16)  # filler_dim * role_dim
        
        model._cached_target_embeddings = target_embeddings
        model._cached_raw_tpr_binding = raw_tpr_binding.unsqueeze(1)  # Add seq_len dim
        
        loss = model._compute_back_projection_loss(target_embeddings)
        
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        assert loss.item() >= 0  # MSE loss should be non-negative
    
    def test_back_projection_loss_invalid_variant_raises_error(self, basic_config, sample_data):
        """Test that invalid loss variant raises error."""
        basic_config.back_projection_loss_variant = 'invalid_variant'
        model = TensorProductEncoderWithBackProjection(basic_config)
        
        target_embeddings = sample_data['target_embeddings']
        model._cached_raw_tpr_binding = torch.randn(4, 1, 32 * 16)
        
        with pytest.raises(ValueError, match="Unknown back_projection_loss_variant"):
            model._compute_back_projection_loss(target_embeddings)
    
    def test_flatten_tensor(self, basic_config):
        """Test _flatten_tensor helper method."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        
        # Test 3D tensor flattening
        tensor_3d = torch.randn(4, 1, 64)
        flattened = model._flatten_tensor(tensor_3d)
        assert flattened.shape == (4, 64)
        
        # Test 2D tensor unchanged
        tensor_2d = torch.randn(4, 64)
        unchanged = model._flatten_tensor(tensor_2d)
        assert torch.equal(unchanged, tensor_2d)
    
    def test_prepare_target_embeddings_tuple_input(self, basic_config):
        """Test _prepare_target_embeddings with LSTM tuple input."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        
        # Simulate LSTM output (hidden_state, cell_state)
        hidden = torch.randn(4, 1, 32)
        cell = torch.randn(4, 1, 32)
        lstm_output = (hidden, cell)
        
        prepared = model._prepare_target_embeddings(lstm_output)
        
        # Should concatenate and flatten
        expected_shape = (4, 64)  # 32 + 32 = 64
        assert prepared.shape == expected_shape
    
    def test_prepare_target_embeddings_regular_tensor(self, basic_config):
        """Test _prepare_target_embeddings with regular tensor input."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        
        target_embeddings = torch.randn(4, 1, 64)
        prepared = model._prepare_target_embeddings(target_embeddings)
        
        # Should just flatten
        expected_shape = (4, 64)
        assert prepared.shape == expected_shape
    
    def test_should_compute_back_projection_loss(self, basic_config):
        """Test _should_compute_back_projection_loss logic."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        
        # Training mode with target embeddings should return True
        model.train()
        target_embeddings = torch.randn(4, 1, 64)
        assert model._should_compute_back_projection_loss(target_embeddings) is True
        
        # Eval mode should return False
        model.eval()
        assert model._should_compute_back_projection_loss(target_embeddings) is False
        
        # Training mode with None target should return False
        model.train()
        assert model._should_compute_back_projection_loss(None) is False
    
    def test_gradient_flow(self, basic_config, sample_data):
        """Test that gradients flow properly through back-projection loss."""
        model = TensorProductEncoderWithBackProjection(basic_config)
        model.train()
        
        # Enable gradients for parameters
        for param in model.parameters():
            param.requires_grad_(True)
        
        outputs = model.forward(
            filler_ids=sample_data['filler_ids'],
            role_ids=sample_data['role_ids'],
            target_embeddings=sample_data['target_embeddings']
        )
        
        # Perform backward pass
        outputs.loss.backward()
        
        # Check that gradients exist for key parameters
        assert model.encoder.filler_embedding.weight.grad is not None
        assert model.encoder.role_embedding.weight.grad is not None
        assert model.encoder.output_layer.weight.grad is not None
        assert model.back_projection.weight.grad is not None
    
    def test_loss_weight_scaling(self, basic_config, sample_data):
        """Test that back-projection loss is properly weighted."""
        # Test with different weights
        weights = [0.1, 1.0, 2.0]
        losses = []
        
        for weight in weights:
            config = TensorProductEncoderConfig(
                hidden_size=64,
                n_fillers=10,
                n_roles=3,
                filler_dim=32,
                role_dim=16,
                has_linear_layer=True,
                back_projection_loss_weight=weight
            )
            
            model = TensorProductEncoderWithBackProjection(config)
            model.train()
            
            outputs = model.forward(
                filler_ids=sample_data['filler_ids'],
                role_ids=sample_data['role_ids'],
                target_embeddings=sample_data['target_embeddings']
            )
            
            losses.append(outputs.loss.item())
        
        # Higher weights should generally lead to different total losses
        # (exact relationships depend on the relative magnitudes of losses)
        assert len(set(losses)) >= 2  # At least some variation in total loss
    
    def test_integration_with_embedding_model(self, basic_config):
        """Test integration with embedding model for target computation."""
        # Create a mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.return_value.last_hidden_state = torch.randn(4, 1, 64)
        
        model = TensorProductEncoderWithBackProjection(basic_config, embedding_model=mock_embedding_model)
        model.train()
        
        batch_size = 4
        seq_len = 5
        
        outputs = model.forward(
            filler_ids=torch.randint(0, 10, (batch_size, seq_len)),
            role_ids=torch.randint(0, 3, (batch_size, seq_len)),
            embedding_model_input_ids=torch.randint(0, 1000, (batch_size, 10)),
            embedding_model_input_lengths=torch.tensor([10, 8, 9, 7])
        )
        
        assert outputs.loss is not None
        assert outputs.encoder_hidden_states is not None
        mock_embedding_model.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__]) 