from typing import Optional, Dict, Any, Tuple, Literal, Union
import os
import json
import torch
import torch.nn as nn
import gin
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

from probing import invert_output_layer, regularized_pinv


def _covariance_sqrt(
    embeddings: torch.Tensor,
    *,
    center: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return SPD covariance square root used for whitening-aware decoding."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be rank-2, got shape={tuple(embeddings.shape)}")
    if center is None:
        center = embeddings.mean(dim=0)
    x_centered = embeddings - center
    denom = max(int(x_centered.shape[0]) - 1, 1)
    cov = (x_centered.T @ x_centered) / float(denom)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov = cov + eps * eye
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(eps)
    cov_sqrt = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T
    return cov_sqrt


def _apply_feature_rescaling(
    *,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    calibration_embeddings: torch.Tensor,
    strategy: Literal["inv_std", "inv_mad"],
    eps: float,
) -> None:
    """Rescale features while preserving decode output under positive scaling."""
    pre_activations = calibration_embeddings @ encoder_weight.T + encoder_bias
    if strategy == "inv_std":
        scale = 1.0 / pre_activations.std(dim=0, unbiased=False).clamp_min(eps)
    elif strategy == "inv_mad":
        med = pre_activations.median(dim=0).values
        mad = (pre_activations - med).abs().median(dim=0).values
        scale = 1.0 / mad.clamp_min(eps)
    else:
        raise ValueError(f"Unknown feature rescaling strategy: {strategy}")

    scale = scale.clamp_min(eps)
    encoder_weight.mul_(scale.unsqueeze(1))
    encoder_bias.mul_(scale)
    decoder_weight.div_(scale.unsqueeze(0))


def _ridge_refine_decoder(
    *,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    refinement_embeddings: torch.Tensor,
    l2_lambda: float,
) -> None:
    """Closed-form ridge fit of decoder/bias with fixed encoder."""
    hidden = torch.relu(refinement_embeddings @ encoder_weight.T + encoder_bias)
    ones = torch.ones(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
    design = torch.cat([hidden, ones], dim=1)  # [N, H+1]

    n_latents = hidden.shape[1]
    gram = design.T @ design
    reg = torch.eye(n_latents + 1, device=gram.device, dtype=gram.dtype)
    reg[-1, -1] = 0.0  # do not regularize bias term
    gram = gram + float(l2_lambda) * reg
    rhs = design.T @ refinement_embeddings
    solution = torch.linalg.solve(gram, rhs)  # [H+1, D]

    decoder_weight.copy_(solution[:-1, :].T)
    decoder_bias.copy_(solution[-1, :])

class SAEConfig(PretrainedConfig):
    """Configuration class for Sparse Autoencoder.
    
    This configuration class stores all hyperparameters and metadata for a Sparse
    Autoencoder (SAE) model. It extends Hugging Face's PretrainedConfig to enable
    model saving/loading compatibility.
    
    Args:
        input_dim: Dimension of the input embeddings (default: 768).
        hidden_dim: Number of hidden features/latents in the SAE (default: 2048).
        sparsity_penalty: L1 regularization coefficient for sparsity loss. Higher values
            encourage sparser activations (default: 0.0).
        dead_latent_threshold: Average activation threshold below which a latent is
            considered "dead" and may be resampled during training (default: 1e-6).
        activation_threshold: Optional threshold for feature activation during eval/inference.
            If set, features with activation below this threshold are masked to zero.
            Used by both L1 SAE and TopK SAE (default: None).
        supervision_weight: Weight for supervised feature activation loss, used by
            SupervisedSparseAutoencoder (default: 1.0).
        feature_map: Optional mapping from feature index to metadata (e.g., filler_id,
            role_id). Used for interpretability and feature quality metrics. Keys are
            string representations of feature indices, values are dicts with metadata.
        embedding_model_name: Optional name/identifier of the embedding model used to
            generate target embeddings during training or evaluation (e.g., 
            "nomic-ai/modernbert-embed-base").
        role_scheme: Optional role scheme identifier used for TPR-constructed SAEs.
            For sentences: typically "svo". For digits: "l2r", "r2l", "bow", or
            "bidirectional".
        feature_map_scheme: Optional scheme indicating how features are organized.
            "filler" means one feature per filler (role-invariant), "filler_role"
            means one feature per filler-role pair.
        **kwargs: Additional arguments passed to PretrainedConfig.

    
    Example:
        >>> config = SAEConfig(
        ...     input_dim=768,
        ...     hidden_dim=2048,
        ...     sparsity_penalty=0.01,
        ...     embedding_model_name="nomic-ai/modernbert-embed-base",
        ...     role_scheme="svo"
        ... )
        >>> sae = SparseAutoencoder(config)
    """
    
    model_type = "sae"
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        sparsity_penalty: float = 0.0,
        dead_latent_threshold: float = 1e-6,
        activation_threshold: Optional[float] = None,
        supervision_weight: float = 1.0,
        feature_map: Optional[Dict[str, Any]] = None,
        embedding_model_name: Optional[str] = None,
        role_scheme: Optional[str] = None,
        feature_map_scheme: Optional[Literal["filler", "filler_role"]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_penalty = sparsity_penalty
        self.dead_latent_threshold = dead_latent_threshold
        self.activation_threshold = activation_threshold
        # Optional: weight for supervised feature activation loss (used by SupervisedSparseAutoencoder)
        self.supervision_weight = supervision_weight
        # Optional: mapping from feature index to metadata (persisted in config)
        self.feature_map = feature_map
        # Optional: embedding model name used for training/evaluation
        self.embedding_model_name = embedding_model_name
        # Optional: role scheme used (mainly relevant for TPR-constructed SAEs)
        self.role_scheme = role_scheme
        # Optional: feature map scheme indicating whether features are per filler or per filler-role pair
        self.feature_map_scheme = feature_map_scheme
        # Avoid returning huge tensors during eval predictions.
        self.keys_to_ignore_at_inference = ["reconstructed", "encoded", "dead_mask", "dead_ratio"]


class SparseAutoencoder(PreTrainedModel):
    """Sparse Autoencoder compatible with Hugging Face Trainer."""
    
    config_class = SAEConfig
    
    def __init__(self, config: SAEConfig):
        super().__init__(config)
        self.config = config
        
        # Encoder: input -> hidden (with sparsity)
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Decoder: hidden -> input (reconstruction)
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim)
        
        # Initialize weights with tied encoder/decoder
        self._init_tied_weights()

        # Track dead latent statistics from the most recent forward pass
        self.last_dead_mask: Optional[torch.Tensor] = None
        self.last_dead_ratio: Optional[torch.Tensor] = None
        
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _init_tied_weights(self):
        """Initialize encoder weights as transpose of decoder weights."""
        # Initialize decoder weights first
        nn.init.xavier_uniform_(self.decoder.weight)
        # Set encoder weights as transpose of decoder weights
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        
        # Initialize biases to zero
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse hidden representation."""
        encoded = self.encoder(x)
        
        # Apply ReLU for sparsity
        encoded = torch.relu(encoded)
        
        # Apply activation thresholding in eval/inference mode if threshold is set
        if not self.training and self.config.activation_threshold is not None:
            threshold = torch.tensor(float(self.config.activation_threshold), device=encoded.device, dtype=encoded.dtype)
            mask = (encoded >= threshold).to(encoded.dtype)
            encoded = encoded * mask
            
        return encoded
    
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation back to input."""
        return self.decoder(hidden)
    
    def forward(
        self, 
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        
        # Handle different input formats
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Either inputs_embeds must be provided")
        
        # Flatten if needed (handle batch_size, seq_len, hidden_dim)
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        
        # Forward pass
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        
        # Calculate losses
        if labels is not None:
            target = labels.view(-1, labels.shape[-1]) if len(labels.shape) > 2 else labels
        else:
            target = x
            
        # Reconstruction loss (MSE)
        reconstruction_loss = nn.functional.mse_loss(reconstructed, target, reduction='mean')
        
        # Sparsity penalty (L1 regularization on encoded activations)
        sparsity_loss = torch.mean(torch.abs(encoded))
        
        # Total loss
        total_loss = reconstruction_loss + self.config.sparsity_penalty * sparsity_loss

        # Dead latent statistics
        avg_activation = encoded.abs().mean(0)
        dead_mask = avg_activation < self.config.dead_latent_threshold
        dead_ratio = dead_mask.float().mean()
        # keep the most recent mask and ratio for callbacks
        self.last_dead_mask = dead_mask.detach()
        self.last_dead_ratio = dead_ratio.detach()
        
        # Reshape back to original dimensions if needed
        if len(original_shape) > 2:
            reconstructed = reconstructed.view(original_shape)
            encoded = encoded.view(*original_shape[:-1], -1)
        
        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'reconstructed': reconstructed,
            'encoded': encoded,
            'dead_ratio': dead_ratio,
            'dead_mask': dead_mask
        }
    
    def get_sparse_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse encoded features for input."""
        return self.encode(x)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse representation."""
        encoded = self.encode(x)
        return self.decode(encoded)

    def resample_features(self, mask: torch.Tensor) -> None:
        """Reinitialize encoder/decoder weights for latents indicated by mask."""
        for idx in mask.nonzero(as_tuple=True)[0]:
            nn.init.xavier_uniform_(self.encoder.weight[idx : idx + 1])
            nn.init.xavier_uniform_(self.decoder.weight[:, idx : idx + 1])
            if self.encoder.bias is not None:
                nn.init.zeros_(self.encoder.bias[idx : idx + 1])

    @classmethod
    def from_tensor_product_encoder(
        cls,
        tpencoder,
        sae_config: Optional[Dict[str, Any]] = None,
        tpe_output_layer_regularization: Literal['l2', 'atol', 'topk'] = 'l2',
        tpe_output_layer_regularization_value: Optional[Union[float, int]] = 1e-12,
        filler_unbinding: Literal['pinv', 'norm'] = 'pinv',
        role_unbinding: Literal['pinv', 'norm'] = 'pinv',
        role_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'none',
        role_pinv_l2_lambda: Optional[float] = None,
        role_pinv_atol: Optional[float] = None,
        role_pinv_topk: Optional[int] = None,
        role_invariant: bool = True,
        first_layer_construction: Literal["unbinding", "pinv-tpencoding"] = "unbinding",
        second_layer_construction: Literal["transpose-unbinding", "pinv-unbinding", "tpencoding"] = "pinv-unbinding",
        allowed_filler_role_pairs: Optional[list[tuple[int, int]]] = None,
        bias_anchor: Optional[torch.Tensor] = None,
        construction_calibration_embeddings: Optional[torch.Tensor] = None,
        decoder_pinv_whiten: bool = False,
        decoder_pinv_regularization: Literal["none", "atol", "l2", "topk"] = "none",
        decoder_pinv_l2_lambda: Optional[float] = None,
        decoder_pinv_atol: Optional[float] = None,
        decoder_pinv_topk: Optional[int] = None,
        feature_rescale_strategy: Literal["none", "inv_std", "inv_mad"] = "none",
        feature_rescale_eps: float = 1e-6,
        decoder_refinement: Literal["none", "ridge"] = "none",
        decoder_refinement_l2: float = 1e-4,
        gating_calibration_embeddings: Optional[torch.Tensor] = None,
        gating_strategy: Literal["none", "quantile", "mad"] = "none",
        gating_target_sparsity: float = 0.02,
        gating_mad_scale: float = 3.0,
    ) -> "SparseAutoencoder":
        """Construct an SAE whose features correspond to TPR bindings.

        Features are created analytically from a trained Tensor Product Encoder
        (TPE). For each filler/role pair (fᵢ, rⱼ) we set decoder column
        d = (W⁺)ᵀ (fᵢ ⊗ uⱼ) and encoder row dᵀ with bias c = −dᵀ b, where W and b
        are the TPE output layer weights and bias, uⱼ are role unbinding vectors
        (computed with a pseudoinverse), and W⁺ is the pseudoinverse of W. When
        role_invariant is True an additional feature is created for each filler
        that fires regardless of role via fᵢ ⊗ (∑ⱼ uⱼ).

        Args:
            tpencoder: Trained TensorProductEncoder instance.
            sae_config: Optional configuration for the SAE. input_dim and
                hidden_dim will be overridden to match the TPE and number of
                constructed features.
            tpe_output_layer_regularization: Regularization method for inverting
                the TPE output layer ('l2', 'atol', or 'topk').
            tpe_output_layer_regularization_value: Regularization parameter value.
                For 'l2': lambda value (float), for 'atol': tolerance (float),
                for 'topk': number of singular values to keep (int).
            filler_unbinding: Method for computing filler unbinding vectors.
            role_unbinding: Method for computing role unbinding vectors.
            role_pinv_regularization: Regularization method for role embedding pseudoinverse.
            role_pinv_l2_lambda: Lambda for role embedding Tikhonov regularization.
            role_pinv_atol: Atol cutoff for role embedding pseudoinverse.
            role_pinv_topk: Top-k singular values for role embedding pseudoinverse.
            role_invariant: If True, features are constructed for each filler regardless of role.
                If False, features are constructed for each filler × role pair.
            allowed_filler_role_pairs: Optional ordered list of legal (filler_id, role_id)
                pairs to construct when role_invariant is False. If provided, overrides
                exhaustive filler × role construction.
            bias_anchor: Optional reconstruction anchor. If set, encoder biases are
                derived using this anchor and decoder bias is set to it.
            construction_calibration_embeddings: Optional embeddings used for
                whitening-aware pseudoinverse, feature rescaling, and decoder
                ridge refinement.
            decoder_pinv_whiten: If True and using ``pinv-unbinding``, compute
                decoder pseudoinverse in whitened coordinates.
            decoder_pinv_regularization: Regularization method for decoder
                pseudoinverse when ``second_layer_construction='pinv-unbinding'``.
            decoder_pinv_l2_lambda: Lambda for decoder Tikhonov regularization.
            decoder_pinv_atol: Atol cutoff for decoder pseudoinverse.
            decoder_pinv_topk: Top-k singular values for decoder pseudoinverse.
            feature_rescale_strategy: Optional per-feature positive rescaling.
                ``inv_std`` and ``inv_mad`` use calibration activations.
            feature_rescale_eps: Numerical floor for scaling statistics.
            decoder_refinement: Optional closed-form decoder refinement mode.
            decoder_refinement_l2: Ridge strength used when
                ``decoder_refinement='ridge'``.
            gating_calibration_embeddings: Tensor of shape [N, input_dim] for calibrating
                per-feature gating thresholds. Required if gating_strategy != "none".
            gating_strategy: How to set per-feature activation thresholds.
                "none": no gating (default).
                "quantile": per-feature threshold at (1 - gating_target_sparsity) quantile.
                "mad": per-feature threshold at median + gating_mad_scale * MAD.
            gating_target_sparsity: Target fraction of examples active per feature (for quantile).
            gating_mad_scale: Multiplier for MAD in robust thresholding.

        Returns:
            SparseAutoencoder with analytically initialized weights.
        """

        if first_layer_construction not in ["unbinding", "pinv-tpencoding"]:
            raise ValueError(f"Invalid first layer construction: {first_layer_construction}")
        if second_layer_construction not in ["transpose-unbinding", "pinv-unbinding", "tpencoding"]:
            raise ValueError(f"Invalid second layer construction: {second_layer_construction}")

        # Gather dimensions and embeddings from the TPE
        filler_embs = tpencoder.filler_embedding.weight.detach()
        role_embs = tpencoder.role_embedding.weight.detach()
        n_fillers, filler_dim = filler_embs.shape
        n_roles, role_dim = role_embs.shape

        compute_device = filler_embs.device
        if tpencoder.output_layer is not None:
            compute_device = tpencoder.output_layer.weight.device

        filler_embs = filler_embs.to(compute_device)
        role_embs = role_embs.to(compute_device)

        # Compute role unbinding vectors per requested method
        if role_unbinding == 'pinv':
            role_unembed = regularized_pinv(
                role_embs,
                regularization=role_pinv_regularization,
                l2_lambda=role_pinv_l2_lambda,
                atol=role_pinv_atol,
                topk=role_pinv_topk,
            )
        else:
            # normalized unbinding: one vector per role scaled by its squared norm
            role_norm_sq = role_embs.norm(dim=1, keepdim=True).pow(2).clamp_min(1e-12)
            role_unembed = (role_embs / role_norm_sq).T.clone()
        role_unembed = role_unembed.to(compute_device)

        # Compute filler unbinding vectors if classification-style unbinding requested
        if filler_unbinding == 'pinv':
            # Require full row rank (num_fillers <= filler_dim)
            assert (
                filler_embs.shape[0] <= filler_embs.shape[1]
            ), "Filler Unbinding only works for full rank fillers"
            filler_unembed = torch.pinverse(filler_embs).T.clone()
        else:
            filler_norm_sq = filler_embs.norm(dim=1, keepdim=True).pow(2).clamp_min(1e-12)
            filler_unembed = filler_embs / filler_norm_sq
        filler_unembed = filler_unembed.to(compute_device)

        # Invert the output layer to obtain W^+
        W_pinv, _ = invert_output_layer(
            tpencoder,
            regularization=tpe_output_layer_regularization,
            l2_lambda=tpe_output_layer_regularization_value if tpe_output_layer_regularization == 'l2' else None,
            atol=tpe_output_layer_regularization_value if tpe_output_layer_regularization == 'atol' else None,
            topk=int(tpe_output_layer_regularization_value) if tpe_output_layer_regularization == 'topk' and tpe_output_layer_regularization_value is not None else None,
            device=compute_device,
        )

        b_tpe = (
            tpencoder.output_layer.bias.detach()
            if tpencoder.output_layer is not None
            else torch.zeros(tpencoder.config.hidden_size)
        )
        b_tpe = b_tpe.to(compute_device)
        b = b_tpe if bias_anchor is None else bias_anchor.to(compute_device)

        # Determine number of features and filler/role mapping
        if role_invariant:
            n_features = n_fillers
            feature_filler_ids = list(range(n_fillers))
            feature_role_ids = None
        else:
            if allowed_filler_role_pairs is not None:
                feature_filler_ids = [int(pair[0]) for pair in allowed_filler_role_pairs]
                feature_role_ids = [int(pair[1]) for pair in allowed_filler_role_pairs]
                n_features = len(allowed_filler_role_pairs)
            else:
                n_features = n_fillers * n_roles
                feature_filler_ids = [idx % n_fillers for idx in range(n_features)]
                feature_role_ids = [idx // n_fillers for idx in range(n_features)]

        # Configure SAE dimensions
        cfg_dict = {
            "input_dim": tpencoder.config.hidden_size,
            "hidden_dim": n_features,
        }
        if sae_config is not None:
            cfg_dict.update(sae_config)
            
        # Build feature map that tracks which filler/role each feature corresponds to
        feature_map: Dict[str, Any] = {}
        
        for feature_idx in range(n_features):
            if role_invariant:
                filler_id = feature_filler_ids[feature_idx]
                feature_map[str(feature_idx)] = {
                    "filler_id": int(filler_id),
                    "role_id": None,
                }
            else:
                filler_id = feature_filler_ids[feature_idx]
                role_id = feature_role_ids[feature_idx]
                feature_map[str(feature_idx)] = {
                    "filler_id": int(filler_id),
                    "role_id": int(role_id),
                }
        
        cfg_dict.setdefault("feature_map", feature_map)
        model = cls(SAEConfig(**cfg_dict))

        # Initialize weights analytically
        target_device = model.decoder.weight.data.device

        if first_layer_construction == "unbinding":
            first_layer_weights = torch.zeros_like(model.encoder.weight.data).to(target_device)
            first_layer_biases = torch.zeros_like(model.encoder.bias.data).to(target_device)
            for feature_idx in range(n_features):
                filler_id = feature_filler_ids[feature_idx]
                if role_invariant:
                    u = role_unembed.sum(dim=1)
                else:
                    role_id = feature_role_ids[feature_idx]
                    u = role_unembed[:, role_id]
                # Build TPR-space detector depending on filler unbinding mode
                d = W_pinv.T @ torch.kron(filler_unembed[filler_id, :], u)
                # Normalize decoder atoms to unit norm so encode-decode projects correctly
                d_norm = d.norm(p=2).clamp_min(1e-12)
                d_unit = d / d_norm
                first_layer_weights[feature_idx, :] = d_unit.to(target_device)
                first_layer_biases[feature_idx] = (-d_unit @ b).to(target_device)
        
        if first_layer_construction == "pinv-tpencoding" or second_layer_construction == "tpencoding":
            second_layer_weights = torch.zeros_like(model.decoder.weight.data).to(target_device)
            role_embed = role_embs.sum(dim=0) if role_invariant else role_embs

            for feature_idx in range(n_features):
                filler_id = feature_filler_ids[feature_idx]
                role_vec = role_embed if role_invariant else role_embed[feature_role_ids[feature_idx], :]

                tpr_contribution = torch.kron(filler_embs[filler_id, :], role_vec)
                if tpencoder.output_layer is not None:
                    linearly_transformed_tpr_contribution = tpencoder.output_layer.weight @ tpr_contribution
                    second_layer_weights[:, feature_idx] = linearly_transformed_tpr_contribution.to(target_device)
                else:
                    raise NotImplementedError("TPEncoding construction is not implemented for TPE without output layer")
            

        # set first layer weights
        if first_layer_construction == "unbinding":
            model.encoder.weight.data.copy_(first_layer_weights)
            model.encoder.bias.data.copy_(first_layer_biases)
        elif first_layer_construction == "pinv-tpencoding":
            model.encoder.weight.data.copy_(torch.pinverse(second_layer_weights, rcond=5e-3))
            # Encoder bias c = -d^T b where d is the decoder column (feature direction)
            # NOT -W_enc @ b, since W_enc = (W_dec)^+ != W_dec^T in general
            model.encoder.bias.data.copy_((-second_layer_weights.T @ b.to(target_device)).to(target_device))
            
        calibration_embeddings = (
            construction_calibration_embeddings
            if construction_calibration_embeddings is not None
            else gating_calibration_embeddings
        )
        if calibration_embeddings is not None:
            calibration_embeddings = calibration_embeddings.to(target_device)

        # set second layer weights
        # Bias is the same regardless of construction method
        # Set decoder bias to the chosen reconstruction anchor.
        model.decoder.bias.data.copy_(b.to(target_device))
        if second_layer_construction == "pinv-unbinding":
            if decoder_pinv_whiten and calibration_embeddings is not None:
                cov_sqrt = _covariance_sqrt(
                    calibration_embeddings,
                    center=b.to(target_device),
                    eps=feature_rescale_eps,
                )
                whitened_encoder = model.encoder.weight.data @ cov_sqrt
                decoder_whitened = regularized_pinv(
                    whitened_encoder,
                    regularization=decoder_pinv_regularization,
                    l2_lambda=decoder_pinv_l2_lambda,
                    atol=decoder_pinv_atol,
                    topk=decoder_pinv_topk,
                )
                model.decoder.weight.data.copy_(cov_sqrt @ decoder_whitened)
            else:
                model.decoder.weight.data.copy_(
                    regularized_pinv(
                        model.encoder.weight.data,
                        regularization=decoder_pinv_regularization,
                        l2_lambda=decoder_pinv_l2_lambda,
                        atol=decoder_pinv_atol,
                        topk=decoder_pinv_topk,
                    )
                )
        elif second_layer_construction == "transpose-unbinding":
            model.decoder.weight.data.copy_(model.encoder.weight.data.T)
        elif second_layer_construction == "tpencoding":
            model.decoder.weight.data.copy_(second_layer_weights)

        if feature_rescale_strategy != "none" and calibration_embeddings is not None:
            _apply_feature_rescaling(
                encoder_weight=model.encoder.weight.data,
                encoder_bias=model.encoder.bias.data,
                decoder_weight=model.decoder.weight.data,
                calibration_embeddings=calibration_embeddings,
                strategy=feature_rescale_strategy,
                eps=feature_rescale_eps,
            )

        # Apply gating calibration to adjust encoder bias
        # Per the TPR-SAE derivation: encoder bias c = -d^T b ensures pre-activation = d^T(h - b)
        # Gating subtracts threshold tau from encoder bias: c' = c - tau = -d^T b - tau
        # This raises the activation bar without changing the decoder (which should still default to b)
        if gating_strategy != "none" and gating_calibration_embeddings is not None:
            calib_embs = gating_calibration_embeddings.to(target_device)
            with torch.no_grad():
                # Compute pre-ReLU activations: W @ x + b
                pre_activations = calib_embs @ model.encoder.weight.data.T + model.encoder.bias.data
                
                if gating_strategy == "quantile":
                    q = 1.0 - gating_target_sparsity
                    thresholds = torch.quantile(pre_activations, q, dim=0)
                    bias_adjustment = thresholds.clamp(min=0.0)
                elif gating_strategy == "mad":
                    medians = torch.median(pre_activations, dim=0).values
                    mad = torch.median(torch.abs(pre_activations - medians), dim=0).values
                    thresholds = medians + gating_mad_scale * mad
                    bias_adjustment = thresholds.clamp(min=0.0)
                
                # Adjust encoder bias: c' = c - tau
                # This shifts the activation threshold without changing the decoder bias
                # Decoder bias remains b so reconstruction defaults to TPE baseline when no features fire
                model.encoder.bias.data -= bias_adjustment

        if decoder_refinement == "ridge" and calibration_embeddings is not None:
            _ridge_refine_decoder(
                encoder_weight=model.encoder.weight.data,
                encoder_bias=model.encoder.bias.data,
                decoder_weight=model.decoder.weight.data,
                decoder_bias=model.decoder.bias.data,
                refinement_embeddings=calibration_embeddings,
                l2_lambda=float(decoder_refinement_l2),
            )

        return model

class TopKSAEConfig(SAEConfig):
    """Configuration for Top-K Sparse Autoencoder."""
    def __init__(self, k: int = 1, threshold_ema_decay: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        # EMA decay for tracking the activation threshold estimate
        self.threshold_ema_decay = threshold_ema_decay
        if self.sparsity_penalty is not None:
            if self.sparsity_penalty > 0.0:
                raise ValueError("sparsity_penalty must be 0.0 for TopKSAEConfig")


class TopKSparseAutoencoder(SparseAutoencoder):
    """Sparse Autoencoder variant that keeps only the top-k activations."""
    config_class = TopKSAEConfig
    
    def __init__(self, config: TopKSAEConfig):
        super().__init__(config)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse hidden representation with top-k masking during training."""
        if not self.training:
            # In eval/inference, use base class encoding with thresholding
            return super().encode(x)
        
        # Training mode: apply top-k masking and estimate threshold
        encoded = self.encoder(x)
        encoded = torch.relu(encoded)
        # Number of active features targeted during training
        k = int(min(self.config.k, encoded.shape[-1]))

        if k > 0:
            # Track the kth largest activation (per example) and update EMA threshold
            values, topk_idx = torch.topk(encoded, k, dim=-1, sorted=True)
            kth_values = values[..., -1]
            batch_threshold = kth_values.mean()
            # Initialize threshold if needed
            if self.config.activation_threshold is None:
                self.config.activation_threshold = float(batch_threshold.detach().item())
            else:
                decay = self.config.threshold_ema_decay
                updated = decay * float(self.config.activation_threshold) + (1.0 - decay) * float(batch_threshold.detach().item())
                self.config.activation_threshold = float(updated)

            # Apply exact top-k during training
            mask = torch.zeros_like(encoded)
            mask.scatter_(-1, topk_idx, 1.0)
            encoded = encoded * mask
        else:
            encoded = torch.zeros_like(encoded)
        
        return encoded


class SupervisedSparseAutoencoder(SparseAutoencoder):
    """Sparse Autoencoder with supervised feature activation loss."""
    def __init__(self, config: SAEConfig):
        super().__init__(config)
        self.supervision_weight = getattr(config, "supervision_weight", 1.0)
    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        feature_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = super().forward(inputs_embeds=inputs_embeds, labels=None)
        # Prefer feature_labels if provided; fall back to labels if present
        target_labels = feature_labels if feature_labels is not None else labels
        if target_labels is not None:
            # to enforce multi class cross-entropy, need to normalize to sum to one
            # there should be 3 features per example
            assert torch.all(target_labels.sum(dim=-1) == 3), "there should be 3 features per example"
            labels = target_labels / target_labels.sum(dim=-1, keepdim=True)
            logits = self.encoder(inputs_embeds)
            supervision_loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
            outputs["supervision_loss"] = supervision_loss
            outputs["loss"] = outputs["loss"] + self.supervision_weight * supervision_loss
        return outputs


def _build_example_labels(
    dataset_split,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
):
    """Construct per-example semantic label sets used by metrics.

    For sentences, labels can be fillers only or (filler, role) pairs.
    For digits, labels map similarly to fillers or (filler, position).

    Returns:
        example_labels: list[set] with labels present per example.
        label_to_indices: dict mapping each label to a list of example indices containing it.
    """
    fillers = dataset_split["filler_ids"]
    roles = dataset_split["role_ids"]

    # Identify roles with exactly one unique filler (often the verb role in sentences datasets)
    singleton_role_ids = set()
    singleton_fillers = set()
    if ignore_singleton_verb:
        role_to_fillers: Dict[int, set] = {}
        for fs, rs in zip(fillers, roles):
            for f, r in zip(fs, rs):
                role_to_fillers.setdefault(int(r), set()).add(int(f))
        for r, fset in role_to_fillers.items():
            if len(fset) == 1:
                singleton_role_ids.add(r)
                singleton_fillers.update(fset)

    example_labels: list = []
    if label_mode == "filler":
        for fs, rs in zip(fillers, roles):
            labels_i = set()
            for f, r in zip(fs, rs):
                f_int, r_int = int(f), int(r)
                if ignore_singleton_verb and f_int in singleton_fillers:
                    continue
                labels_i.add(f_int)
            example_labels.append(labels_i)
    elif label_mode == "filler_role":
        for fs, rs in zip(fillers, roles):
            labels_i = set()
            for f, r in zip(fs, rs):
                f_int, r_int = int(f), int(r)
                if ignore_singleton_verb and r_int in singleton_role_ids:
                    continue
                labels_i.add((f_int, r_int))
            example_labels.append(labels_i)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    # Map each label to the indices of examples that contain it
    label_to_indices: Dict[Any, list] = {}
    for idx, labels in enumerate(example_labels):
        for lbl in labels:
            label_to_indices.setdefault(lbl, []).append(idx)

    return example_labels, label_to_indices


def compute_feature_quality(
    dataset_split,
    activations: Optional[torch.Tensor] = None,
    activation_threshold: float = 0.0,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> float:
    """Compute feature quality as activation-weighted purity over features.

    For each feature, consider the set of examples where the feature activates
    (activation > threshold). Among the labels present in those examples, sum
    activations per label and compute purity = max_weight / total_activation.
    Return the activation-mass-weighted mean of per-feature purity values.

    Args:
        dataset_split: Dataset split containing 'filler_ids' and 'role_ids'.
        activations: Precomputed SAE activations tensor of shape [N, F].
        activation_threshold: Activation threshold to consider a feature active.
        label_mode: 'filler' to use only filler IDs as labels, or 'filler_role'
            to use (filler, role) pairs as distinct labels.
        ignore_singleton_verb: If True, ignore labels corresponding to roles
            that have exactly one unique filler across the dataset split. For
            label_mode='filler', the singleton fillers are ignored globally.

    Returns:
        A float in [0, 1] representing the activation-weighted purity.
    """
    # Build example labels according to requested mode
    example_labels, _ = _build_example_labels(
        dataset_split, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
    )

    num_examples, num_features = activations.shape[0], activations.shape[-1]
    # Boolean activation mask per feature
    active_mask = activations > activation_threshold

    weighted_purity_sum = 0.0
    total_weight = 0.0

    # Iterate over features to compute per-feature purity
    for feature_idx in range(num_features):
        active_indices = torch.nonzero(active_mask[:, feature_idx], as_tuple=False).view(-1)
        num_active = int(active_indices.numel())
        if num_active == 0:
            continue

        label_weights: Dict[Any, float] = {}
        feature_activations = activations[active_indices, feature_idx]
        activation_total = float(feature_activations.sum().item())
        if activation_total <= 0.0:
            continue

        # Sum activation weights per label among activating examples.
        for offset, i in enumerate(active_indices.tolist()):
            weight = float(feature_activations[offset].item())
            for lbl in example_labels[i]:
                label_weights[lbl] = label_weights.get(lbl, 0.0) + weight

        if not label_weights:
            purity = 0.0
        else:
            max_weight = max(label_weights.values())
            purity = float(max_weight) / activation_total

        weighted_purity_sum += activation_total * purity
        total_weight += activation_total

    avg_quality = (weighted_purity_sum / total_weight) if total_weight > 0.0 else 0.0
    return float(avg_quality)


def _compute_auc_score(pos_vals: torch.Tensor, neg_vals: torch.Tensor) -> float:
    """Compute the AUC-style pairwise ranking score for a feature."""
    neg_sorted = torch.sort(neg_vals).values
    count_less = torch.searchsorted(neg_sorted, pos_vals, right=False)
    count_le = torch.searchsorted(neg_sorted, pos_vals, right=True)
    count_equal = count_le - count_less
    per_pos_scores = (count_less.to(torch.float32) + 0.5 * count_equal.to(torch.float32)) / float(neg_vals.numel())
    return float(per_pos_scores.mean().item())


def select_feature_labels_by_well_rankedness(
    dataset_split,
    activations: Optional[torch.Tensor] = None,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> list:
    """Assign the best-matching semantic label to each feature using AUC on a split."""
    if activations is None:
        raise ValueError("activations must be provided for label selection.")
    _, label_to_indices = _build_example_labels(
        dataset_split, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
    )

    num_examples, num_features = activations.shape[0], activations.shape[-1]
    all_indices = torch.arange(num_examples)
    best_labels = [None] * num_features

    for feature_idx in range(num_features):
        a = activations[:, feature_idx]
        if torch.max(a) <= 0:
            continue
        best_score = 0.5
        best_weight = 0
        best_label = None

        for label, pos_list in label_to_indices.items():
            if not pos_list:
                continue
            pos_idx = torch.tensor(pos_list, dtype=torch.long)
            neg_mask = torch.ones(num_examples, dtype=torch.bool)
            neg_mask[pos_idx] = False
            neg_idx = all_indices[neg_mask]

            n_pos = int(pos_idx.numel())
            n_neg = int(neg_idx.numel())
            if n_pos == 0 or n_neg == 0:
                continue

            score = _compute_auc_score(a[pos_idx], a[neg_idx])
            pair_count = int(n_pos * n_neg)

            if score > best_score or (score == best_score and pair_count > best_weight):
                best_score = score
                best_weight = pair_count
                best_label = label

        best_labels[feature_idx] = best_label

    return best_labels


def compute_feature_accuracy(
    dataset_split,
    activations: Optional[torch.Tensor] = None,
    feature_labels: Optional[list] = None,
    activation_threshold: float = 0.0,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> float:
    """Compute per-feature accuracy using preselected feature labels."""
    if activations is None:
        raise ValueError("activations must be provided for accuracy computation.")
    if feature_labels is None:
        raise ValueError("feature_labels must be provided for accuracy computation.")
    _, label_to_indices = _build_example_labels(
        dataset_split, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
    )

    num_examples, num_features = activations.shape[0], activations.shape[-1]
    if len(feature_labels) != num_features:
        raise ValueError("feature_labels length must match activations feature dimension.")
    active_mask = activations > activation_threshold
    all_indices = torch.arange(num_examples)
    weighted_acc_sum = 0.0
    total_weight = 0.0

    for feature_idx in range(num_features):
        label = feature_labels[feature_idx]
        if label is None:
            continue
        pos_list = label_to_indices.get(label, [])
        if not pos_list:
            continue

        pos_idx = torch.tensor(pos_list, dtype=torch.long)
        neg_mask = torch.ones(num_examples, dtype=torch.bool)
        neg_mask[pos_idx] = False
        neg_idx = all_indices[neg_mask]

        n_pos = int(pos_idx.numel())
        n_neg = int(neg_idx.numel())
        if n_pos == 0 or n_neg == 0:
            continue

        active_pos = int(active_mask[pos_idx, feature_idx].sum().item())
        active_neg = int(active_mask[neg_idx, feature_idx].sum().item())
        tp = active_pos
        fn = n_pos - active_pos
        fp = active_neg
        tn = n_neg - active_neg

        total = n_pos + n_neg
        accuracy = (tp + tn) / total
        weighted_acc_sum += total * accuracy
        total_weight += total

    avg_accuracy = (weighted_acc_sum / total_weight) if total_weight > 0.0 else 0.0
    return float(avg_accuracy)


def compute_feature_recall(
    dataset_split,
    activations: Optional[torch.Tensor] = None,
    feature_labels: Optional[list] = None,
    activation_threshold: float = 0.0,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> float:
    """Compute per-feature recall using preselected feature labels."""
    if activations is None:
        raise ValueError("activations must be provided for recall computation.")
    if feature_labels is None:
        raise ValueError("feature_labels must be provided for recall computation.")
    _, label_to_indices = _build_example_labels(
        dataset_split, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
    )

    num_features = activations.shape[-1]
    if len(feature_labels) != num_features:
        raise ValueError("feature_labels length must match activations feature dimension.")
    active_mask = activations > activation_threshold
    weighted_recall_sum = 0.0
    total_weight = 0.0

    for feature_idx in range(num_features):
        label = feature_labels[feature_idx]
        if label is None:
            continue
        pos_list = label_to_indices.get(label, [])
        if not pos_list:
            continue

        pos_idx = torch.tensor(pos_list, dtype=torch.long)
        n_pos = int(pos_idx.numel())
        if n_pos == 0:
            continue
        active_pos = int(active_mask[pos_idx, feature_idx].sum().item())
        recall = active_pos / n_pos
        weighted_recall_sum += n_pos * recall
        total_weight += n_pos

    avg_recall = (weighted_recall_sum / total_weight) if total_weight > 0.0 else 0.0
    return float(avg_recall)


def compute_feature_well_rankedness_per_feature(
    dataset_split,
    activations: Optional[torch.Tensor] = None,
    feature_labels: Optional[list] = None,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> torch.Tensor:
    """Compute per-feature well-rankedness scores for assigned labels."""
    if activations is None:
        raise ValueError("activations must be provided for well-rankedness computation.")
    if feature_labels is None:
        raise ValueError("feature_labels must be provided for well-rankedness computation.")
    _, label_to_indices = _build_example_labels(
        dataset_split, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
    )

    num_examples, num_features = activations.shape[0], activations.shape[-1]
    if len(feature_labels) != num_features:
        raise ValueError("feature_labels length must match activations feature dimension.")
    all_indices = torch.arange(num_examples)
    scores = torch.full((num_features,), float("nan"), dtype=torch.float32)

    for feature_idx in range(num_features):
        label = feature_labels[feature_idx]
        if label is None:
            continue
        pos_list = label_to_indices.get(label, [])
        if not pos_list:
            continue
        pos_idx = torch.tensor(pos_list, dtype=torch.long)
        neg_mask = torch.ones(num_examples, dtype=torch.bool)
        neg_mask[pos_idx] = False
        neg_idx = all_indices[neg_mask]

        n_pos = int(pos_idx.numel())
        n_neg = int(neg_idx.numel())
        if n_pos == 0 or n_neg == 0:
            continue

        a = activations[:, feature_idx]
        score = _compute_auc_score(a[pos_idx], a[neg_idx])
        scores[feature_idx] = float(score)

    return scores


def compute_feature_well_rankedness(
    dataset_split,
    activations: Optional[torch.Tensor] = None,
    feature_labels: Optional[list] = None,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> float:
    """Compute average feature "well-rankedness" as a pairwise ranking metric.

    A feature is "well-ranked" for a given semantic label S if, across all
    examples containing S (positives) and not containing S (negatives), the
    feature activation satisfies: every positive activation is greater than every
    negative activation. We operationalize this with a pairwise ranking score
    (Mann-Whitney U / AUC): for each positive-negative pair, count 1 if
    a_pos > a_neg, 0.5 if equal, 0 otherwise. The feature's well-rankedness is
    the score for its assigned label; labels should be chosen on a separate
    split (e.g., training) to avoid leakage. We then aggregate across features
    using pair-count weighting. This returns 0.5 at random, 1.0 for perfect
    separation, and 0.0 for perfectly reversed ranking.

    Returns:
        A float in [0, 1] representing the pairwise ranking quality.
    """
    scores = compute_feature_well_rankedness_per_feature(
        dataset_split=dataset_split,
        activations=activations,
        feature_labels=feature_labels,
        label_mode=label_mode,
        ignore_singleton_verb=ignore_singleton_verb,
    )
    _, label_to_indices = _build_example_labels(
        dataset_split, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
    )
    if feature_labels is None:
        raise ValueError("feature_labels must be provided for well-rankedness computation.")
    num_examples = activations.shape[0]
    all_indices = torch.arange(num_examples)
    weights = torch.zeros_like(scores)
    for feature_idx in range(scores.shape[0]):
        if torch.isnan(scores[feature_idx]):
            continue
        label = feature_labels[feature_idx]
        if label is None:
            continue
        # Weight by the number of positive-negative pairs for the selected label.
        pos_list = label_to_indices.get(label, [])
        if not pos_list:
            continue
        pos_idx = torch.tensor(pos_list, dtype=torch.long)
        neg_mask = torch.ones(num_examples, dtype=torch.bool)
        neg_mask[pos_idx] = False
        neg_idx = all_indices[neg_mask]
        n_pos = int(pos_idx.numel())
        n_neg = int(neg_idx.numel())
        if n_pos == 0 or n_neg == 0:
            continue
        weights[feature_idx] = float(n_pos * n_neg)

    valid_mask = ~torch.isnan(scores)
    weighted_sum = float((scores[valid_mask] * weights[valid_mask]).sum().item())
    total_weight = float(weights[valid_mask].sum().item())
    avg_score = (weighted_sum / total_weight) if total_weight > 0.0 else 0.5
    return float(avg_score)


def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


@gin.configurable
def evaluate_sae(
    sae: SparseAutoencoder,
    label_dataset: Any,
    eval_dataset: Any,
    eval_embeddings: torch.Tensor,
    eval_reconstructions: torch.Tensor,
    eval_activations: torch.Tensor,
    label_activations: torch.Tensor,
    sae_output_dir: Optional[str] = None,
    label_mode: Literal["filler", "filler_role"] = "filler",
    ignore_singleton_verb: bool = True,
) -> Dict[str, float]:
    """Evaluate a Sparse Autoencoder on a dataset.

    This function is metric-only; forward passes should be run by the caller.

    Args:
        sae: The SparseAutoencoder model to evaluate
        label_dataset: Dataset split used to choose per-feature semantic labels.
        eval_dataset: Dataset split used to compute evaluation metrics.
        eval_embeddings: Precomputed embeddings for eval_dataset.
        eval_reconstructions: Precomputed reconstructions for eval_dataset.
        eval_activations: Precomputed activations for eval_dataset.
        label_activations: Precomputed activations for label_dataset.
        sae_output_dir: Optional directory to save evaluation metrics
        label_mode: Label mode for feature quality and well-rankedness calculation ("filler" or "filler_role")
        ignore_singleton_verb: Whether to ignore singleton verbs in feature quality and well-rankedness calculation

    Returns:
        Dictionary containing evaluation metrics (mse, cosine_similarity, l0_sparsity, avg_feature_purity, avg_feature_well_rankedness)
    """
    # Read activation_threshold from SAE config, defaulting to 0.0 if not set
    activation_threshold = float(getattr(sae.config, "activation_threshold", 0.0) or 0.0)
    eval_embeddings = eval_embeddings.cpu()
    eval_reconstructions = eval_reconstructions.cpu()
    eval_activations = eval_activations.cpu()
    label_activations = label_activations.cpu()

    mse = torch.nn.functional.mse_loss(eval_reconstructions, eval_embeddings).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        eval_reconstructions, eval_embeddings, dim=1
    ).mean().item()
    total_activations = eval_activations.numel()
    l0 = (float((eval_activations > 0).sum().item()) / total_activations) if total_activations > 0 else float("nan")
    r2 = compute_r2(eval_embeddings, eval_reconstructions)
    # Pick labels on the label split to avoid leaking eval semantics.
    feature_labels = select_feature_labels_by_well_rankedness(
        dataset_split=label_dataset,
        activations=label_activations,
        label_mode=label_mode,
        ignore_singleton_verb=ignore_singleton_verb,
    )
    # Compute feature quality and well-rankedness
    metrics = {
        "mse": mse,
        "cosine_similarity": cosine_similarity,
        "r2": r2,
        "l0_sparsity": l0,
        "avg_feature_purity": compute_feature_quality(
            eval_dataset, activations=eval_activations, activation_threshold=activation_threshold, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
        ),
        "avg_feature_accuracy": compute_feature_accuracy(
            eval_dataset, activations=eval_activations, feature_labels=feature_labels, activation_threshold=activation_threshold, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
        ),
        "avg_feature_well_rankedness": compute_feature_well_rankedness(
            eval_dataset, activations=eval_activations, feature_labels=feature_labels, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
        ),
        "avg_feature_recall": compute_feature_recall(
            eval_dataset, activations=eval_activations, feature_labels=feature_labels, activation_threshold=activation_threshold, label_mode=label_mode, ignore_singleton_verb=ignore_singleton_verb
        ),
    }
    
    if sae_output_dir is not None:
        os.makedirs(sae_output_dir, exist_ok=True)
        with open(os.path.join(sae_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    print(metrics)
    
    return metrics


AutoConfig.register("sae", SAEConfig)
AutoModel.register(SAEConfig, SparseAutoencoder)


class SAETrainer(Trainer):
    """Minimal Trainer wrapper for SparseAutoencoder.
    
    Extends HuggingFace Trainer to handle SAE-specific loss computation
    and dead latent tracking.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for SAE training."""
        if "feature_labels" in inputs:
            outputs = model(
                inputs_embeds=inputs["inputs_embeds"],
                feature_labels=inputs["feature_labels"],
            )
        else:
            outputs = model(inputs_embeds=inputs["inputs_embeds"]) 
        loss = outputs["loss"]
        # store latest mask for callbacks and log dead ratio
        model.last_dead_mask = outputs["dead_mask"].detach()
        model.last_dead_ratio = outputs["dead_ratio"].detach()
        return (loss, outputs) if return_outputs else loss


class DeadLatentResampler(TrainerCallback):
    """Callback that reinitializes latents with near-zero activation.
    
    Tracks dead latents during training and reinitializes them at scheduled
    intervals to prevent unused capacity in the sparse autoencoder.
    """

    def __init__(self, threshold: float = 1.0, resample_times: int = 0):
        self.threshold = threshold
        self.resample_times = resample_times
        self.count = 0
        self.running_mask = None
        self.steps = 0
        self.schedule = []

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resample_times > 0:
            # Evenly divide epochs, resample at k/(n+1) of total epochs
            self.schedule = [
                args.num_train_epochs * (i + 1) / (self.resample_times + 1)
                for i in range(self.resample_times)
            ]
        else:
            self.schedule = []

    def on_step_end(self, args, state, control, **kwargs):
        mask = getattr(kwargs["model"], "last_dead_mask", None)
        if mask is None:
            return
        mask = mask.float()
        if self.running_mask is None:
            self.running_mask = mask.clone()
        else:
            self.running_mask += mask
        self.steps += 1

    def _should_resample(self, current_epoch: float) -> bool:
        return (
            self.count < len(self.schedule)
            and current_epoch >= self.schedule[self.count]
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if self.running_mask is None:
            self.steps = 0
            return

        model = kwargs["model"]

        avg_mask = self.running_mask / max(self.steps, 1)
        dead_ratio = avg_mask.mean().item()
        if "metrics" in kwargs:
            kwargs["metrics"]["eval_dead_ratio"] = dead_ratio

        if state.epoch is not None and self._should_resample(state.epoch):
            dead = avg_mask >= self.threshold
            if dead.any():
                model.resample_features(dead)
                self.count += 1
                print(
                    f"[INFO] Resampled {dead.sum().item()} dead latents at epoch {state.epoch:.2f}"
                )

        self.running_mask = None
        self.steps = 0


def sae_trainer_embedding_collator(batch):
    """Collate adapter for Hugging Face Trainer when training the SAE.
    
    Why this is needed:
    - Our `SparseAutoencoder` consumes precomputed embedding vectors rather than token ids.
    - Hugging Face `Trainer` expects a collator that turns a list of dataset
      examples into a single batch dict. The default data collators operate on
      token ids and attention masks; here we must build a dict with the
      `inputs_embeds` tensor explicitly so `Trainer` forwards it as keyword
      arguments to the model.
    
    How it is used:
    - Pass this function as `data_collator` to `Trainer`/`SAETrainer`.
    - Each dataset example must include a `target_embeddings` field containing
      a 1D array-like of floats. This collator stacks them into a float32 tensor
      of shape (batch_size, embedding_dim) under the key `inputs_embeds`.
    - Training configs set `TrainingArguments.label_names = ['inputs_embeds']`
      so Trainer does not try to treat any batch entries as labels.
    """
    embeddings = torch.tensor([ex["target_embeddings"] for ex in batch], dtype=torch.float)
    collated = {"inputs_embeds": embeddings}
    if "feature_labels" in batch[0]:
        labels = torch.tensor([ex["feature_labels"] for ex in batch], dtype=torch.float)
        collated["feature_labels"] = labels
    return collated
