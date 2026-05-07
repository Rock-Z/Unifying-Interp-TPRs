from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Literal, Optional
import torch
from utils import get_st_dimension, search_reg_param
import gin

# For computing pseudoinverses
def tikhonov_pinv(B, lambda_):
    """Compute the Tikhonov-regularized (l2) pseudoinverse of a matrix using SVD.

    Args:
        B (torch.Tensor): Input matrix to compute the pseudoinverse for
        lambda_ (float): Regularization parameter controlling the amount of damping

    Returns:
        torch.Tensor: The Tikhonov-regularized pseudoinverse of B

    Note:
        When lambda_ is 0, this reduces to the standard Moore-Penrose pseudoinverse.
        Larger values of lambda_ increase regularization strength. With known noise
        standard deviation, optimal lambda_ is the noise standard deviation.
    """

    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    S_damped = S / (S**2 + lambda_)
    return Vh.T @ torch.diag(S_damped) @ U.T

def tsvd_pinv(B, k):
    """Computes the truncated SVD pseudoinverse of a matrix.

    Args:
        B (torch.Tensor): The input matrix to compute the pseudoinverse for.
        k (int): The number of singular values to keep.

    Returns:
        torch.Tensor: The truncated SVD pseudoinverse of B, using only the top k
            singular values.
    """

    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    S_inv = torch.zeros_like(S)
    S_inv[:k] = 1 / S[:k]
    return Vh[:k, :].T @ torch.diag(S_inv[:k]) @ U[:, :k].T

def regularized_pinv(
    B: torch.Tensor,
    regularization: Literal["none", "atol", "l2", "topk"] = "none",
    l2_lambda: Optional[float] = None,
    atol: Optional[float] = None,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """Compute a (possibly) regularized pseudoinverse for arbitrary matrices."""
    if regularization in (None, "none"):
        return torch.pinverse(B)
    if regularization == "atol":
        if atol is None:
            raise ValueError("Atol must be specified if regularization is 'atol'")
        return torch.linalg.pinv(B, atol=atol)
    if regularization == "l2":
        if l2_lambda is None:
            raise ValueError("L2 regularization lambda must be specified if regularization is 'l2'")
        return tikhonov_pinv(B, l2_lambda)
    if regularization == "topk":
        if topk is None:
            raise ValueError("Topk must be specified if regularization is 'topk'")
        return tsvd_pinv(B, topk)
    raise ValueError("Regularization must be one of 'none', 'atol', 'l2', or 'topk'")


class LinearProbeConfig(PretrainedConfig):
    model_type='probe_classifier'

    def __init__(
            self, 
            encoder_model_type=None, 
            encoder_hidden_size=None,
            num_labels=None, 
            intermediate_dims=None,
            **kwargs
            ):
        super().__init__()
        self.encoder_model_type = encoder_model_type
        self.encoder_hidden_size = encoder_hidden_size
        self.num_labels = num_labels
        # Avoid mutable default; normalize to list of ints
        if intermediate_dims is None:
            self.intermediate_dims = []
        else:
            self.intermediate_dims = [int(d) for d in intermediate_dims]

    @property
    def num_labels(self):
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels):
        self._num_labels = num_labels
        if num_labels is not None:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        else:
            self.id2label = None
            self.label2id = None

class LinearProbe(PreTrainedModel):
    config_class = LinearProbeConfig

    def __init__(self, config, encoder=None):
        super().__init__(config)
        self.config = config
        self.encoder = encoder
        if config.encoder_hidden_size is None:
            if encoder is None:
                raise ValueError("encoder_hidden_size must be specified in config when encoder is None")
            config.encoder_hidden_size = encoder.config.hidden_size
        # Normalize dimensions to ints
        dims = [int(config.encoder_hidden_size)] + [int(d) for d in (config.intermediate_dims or [])] + [int(config.num_labels)]
        # Explicitly set dtype to avoid None being forwarded to torch.empty by some torch versions
        self.classifier = torch.nn.Sequential(
            *[torch.nn.Linear(dims[i], dims[i+1], dtype=torch.float32) for i in range(len(dims)-1)]
        )
        #self.classifier = torch.nn.Linear(encoder.config.hidden_size, config.num_labels)

    @classmethod
    @gin.configurable("LinearProbe.from_tpencoder", module='probing')
    def from_tpencoder(
            cls, 
            tpencoder, 
            encoder, 
            role_id,
            role_unbinding : Literal['pinv', 'norm'] = 'pinv',
            filler_unbinding : Literal['pinv', 'norm'] = 'norm',
            regularization : Literal['atol', 'l2', 'topk'] = 'atol',
            l2_lambda : Optional[float] = None,
            atol : Optional[float] = None,
            topk : Optional[int] = None,
            role_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'l2',
            role_pinv_l2_lambda: Optional[float] = 1e-2,
            role_pinv_atol: Optional[float] = None,
            role_pinv_topk: Optional[int] = None,
            filler_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'none',
            filler_pinv_l2_lambda: Optional[float] = None,
            filler_pinv_atol: Optional[float] = None,
            filler_pinv_topk: Optional[int] = None,
            mode : Literal['embedding', 'classification']='classification',
            embedding_model_name : Optional[str] = None,
            use_trained_layers : Optional[bool] = True
        ):
        """
        Construct a linear probe by manipulating the weights of a tensor product encoder.
        Args:
            tpencoder: TensorProductEncoder, TensorProductEncoderWithDecodingLoss, or TensorProductEncoderWithBackProjection
            encoder: PreTrainedModel, optional
                Encoder model. Must provide either encoder or embedding_model_name, but not both.
            role_id: int
            role_unbinding: str, default 'pinv'
                Method for unbinding role vectors
            filler_unbinding: str, default 'pinv'
                Method for unbinding filler vectors
            regularization: str, default 'atol'
                Method for regularizing the linear transformation. Options are 'atol', 'l2', or 'topk'.
                'atol' uses the atol parameter to ignore small singular values, 'l2' uses Tikhonov 
                regularization, and 'topk' uses truncated SVD to keep only the top k singular values.
                Not used if tpencoder has trained inverse/back projection layers and use_trained_layers is not False.
            atol: float, optional, default None
                Atol for pinv, used if regularization is 'atol'
            l2_lambda: float, optional, default None
                Lambda for Tikhonov regularization, used if regularization is 'l2'
            topk: int, optional, default None  
                Number of singular values to keep, used if regularization is 'topk'
            role_pinv_regularization: str, default 'l2'
                Regularization used when role_unbinding is 'pinv'.
            role_pinv_l2_lambda: float, optional
                Lambda for role embedding Tikhonov regularization.
            role_pinv_atol: float, optional
                Atol cutoff for role embedding pseudoinverse.
            role_pinv_topk: int, optional
                Number of singular values to keep for role embedding pseudoinverse.
            filler_pinv_regularization: str, default 'none'
                Regularization used when filler_unbinding is 'pinv'.
            filler_pinv_l2_lambda: float, optional
                Lambda for filler embedding Tikhonov regularization.
            filler_pinv_atol: float, optional
                Atol cutoff for filler embedding pseudoinverse.
            filler_pinv_topk: int, optional
                Number of singular values to keep for filler embedding pseudoinverse.
            mode: str, default 'classification'
                Mode of probe, either 'classification' or 'embedding'. If 'classification', 
                the probe will classify fillers. If 'embedding', the probe will output 
                filler embeddings.
            embedding_model_name: str, optional, default None
                Model name/path for SentenceTransformer models. If provided, avoids loading 
                the model by getting dimension from config files instead. Must provide either 
                encoder or embedding_model_name, but not both.
            use_trained_layers: Optional[bool], default True
                Whether to use trained inverse/back projection layers when available. 
                If None or True, uses trained layers when available. If False, always computes pseudoinverse.
        Returns:
            LinearProbe
        """

        # Validate that exactly one of encoder or embedding_model_name is provided
        assert (encoder is not None) != (embedding_model_name is not None), "Must provide either encoder or embedding_model_name, but not both."

        # Determine the base TPE encoder from the input
        if hasattr(tpencoder, 'encoder') and tpencoder.encoder is not None:
            # For TensorProductEncoderWithDecodingLoss or TensorProductEncoderWithBackProjection
            base_tpencoder = tpencoder.encoder
        else:
            # For regular TensorProductEncoder
            base_tpencoder = tpencoder

        # Determine encoder model type, config, and hidden size
        if encoder is None:
            # Using embedding_model_name only
            assert embedding_model_name is not None, "embedding_model_name must be provided when encoder is None"
            encoder_model_type = "sentence-transformers"
            encoder_config = None
            encoder_hidden_size = get_st_dimension(embedding_model_name)
        elif hasattr(encoder, "config") and hasattr(encoder.config, "model_type"):
            encoder_model_type = encoder.config.model_type
            encoder_config = encoder.config
            encoder_hidden_size = encoder.config.hidden_size
        elif encoder.__class__.__name__ == "SentenceTransformer":
            encoder_model_type = "sentence-transformers"
            encoder_config = None
            # Avoid loading model if embedding_model_name is provided
            if embedding_model_name is not None:
                encoder_hidden_size = get_st_dimension(embedding_model_name)
            else:
                # Fallback to loading model (original behavior)
                encoder_hidden_size = encoder.get_sentence_embedding_dimension()
        else:
            raise ValueError("Encoder model type not recognized. Currently supported: SentenceTransformer's or HuggingFace PreTrainedModel")

        if base_tpencoder.output_layer is not None and encoder_hidden_size is not None:
            tpe_hidden_size = int(base_tpencoder.output_layer.out_features)
            if tpe_hidden_size != int(encoder_hidden_size):
                if encoder is not None and getattr(encoder.config, "architecture", None) == "LSTM":
                    encoder_hidden_size = tpe_hidden_size
                else:
                    raise ValueError(
                        "Probe encoder hidden size does not match TPE output size: "
                        f"{encoder_hidden_size=} vs tpe_hidden_size={tpe_hidden_size}. "
                        "Ensure the probe encoder matches the TPE training representation."
                    )
        
        # contruct linear layer weights from tensor product encoder weights
        assert role_id < base_tpencoder.role_embedding.num_embeddings and role_id >= 0, "Role ID out of range!"
        if role_unbinding not in ['pinv', 'norm'] or filler_unbinding not in ['pinv', 'norm']:
            raise ValueError(f"Unbinding mode must be one of 'pinv' or 'norm', found {role_unbinding=} and {filler_unbinding=}")
        if regularization == 'l2' and l2_lambda is None:
            raise ValueError("L2 regularization lambda must be specified if regularization is 'l2'")
        elif regularization == 'atol' and atol is None:
            raise ValueError("Atol for pinv must be specified if regularization is 'atol'")
        elif regularization == 'topk' and topk is None:
            raise ValueError("Topk must be specified if regularization is 'topk'")
        elif regularization not in ['atol', 'l2', 'topk']:
            raise ValueError("Regularization must be one of 'atol', 'l2', or 'topk'")

        # Construct probe weights using helper functions
        device = torch.device('cpu')
        
        # Handle None case - treat as True
        should_use_trained = use_trained_layers is not False
        
        # Check for and use trained layers (inverse_layer or back_projection)
        trained_layer = None
        if should_use_trained:
            if hasattr(tpencoder, 'inverse_layer') and tpencoder.inverse_layer is not None:
                trained_layer = tpencoder.inverse_layer
            elif hasattr(tpencoder, 'back_projection') and tpencoder.back_projection is not None:
                trained_layer = tpencoder.back_projection
        
        # Invert linear transformation
        if trained_layer is not None:
            # Use trained layer weights
            W_inv = trained_layer.weight.clone().detach().to(device)
            bias = trained_layer.bias.clone().detach().to(device)
        else:
            # Fall back to computing pseudoinverse
            W_inv, bias = invert_output_layer(
                tpencoder=base_tpencoder,
                regularization=regularization,
                l2_lambda=l2_lambda,
                atol=atol,
                topk=topk,
                device=device
            )
        
        # Construct unbinding vectors and probe weights
        W_probe = construct_unbinding_vectors(
            tpencoder=base_tpencoder,
            role_id=role_id,
            role_unbinding=role_unbinding,
            filler_unbinding=filler_unbinding,
            role_pinv_regularization=role_pinv_regularization,
            role_pinv_l2_lambda=role_pinv_l2_lambda,
            role_pinv_atol=role_pinv_atol,
            role_pinv_topk=role_pinv_topk,
            filler_pinv_regularization=filler_pinv_regularization,
            filler_pinv_l2_lambda=filler_pinv_l2_lambda,
            filler_pinv_atol=filler_pinv_atol,
            filler_pinv_topk=filler_pinv_topk,
            mode=mode,
            device=device
        )

        filler_dim = base_tpencoder.filler_embedding.embedding_dim
        num_fillers = base_tpencoder.filler_embedding.num_embeddings

        config = LinearProbeConfig(
            encoder_model_type=encoder_model_type,
            encoder_config=encoder_config,
            encoder_hidden_size=encoder_hidden_size,
            num_labels=num_fillers if mode == 'classification' else filler_dim,
            intermediate_dims = [base_tpencoder.output_layer.in_features] if base_tpencoder.output_layer is not None else None
        )

        probe = cls(config, encoder)

        with torch.no_grad():
            # manually set weights
            probe.classifier[0].weight.copy_(W_inv) 
            probe.classifier[0].bias.copy_(bias)
            probe.classifier[1].weight.copy_(W_probe)
            probe.classifier[1].bias.copy_(torch.zeros_like(probe.classifier[1].bias))

        return probe

    def forward(self, labels = None, hidden_states = None, **kwargs):
        if hidden_states is None:
            if self.encoder is None:
                raise ValueError("Cannot generate hidden_states when encoder is None. Please provide hidden_states directly.")
            with torch.no_grad():
                if self.config.encoder_model_type == 'sentence-transformers':
                    # sentence transformers are wrapped and use .encode of raw text strings instead of .forward on tokens
                    hidden_states = self.encoder.encode(kwargs['sentence'])
                    hidden_states = torch.tensor(hidden_states).to(self.device)
                else:
                    outputs = self.encoder(**kwargs)
                    hidden_states = outputs.last_hidden_state 
                    if isinstance(hidden_states, tuple):
                        if len(hidden_states) != 2:
                            raise ValueError(
                                f"Unexpected tuple hidden_states length={len(hidden_states)}; expected (h, c)."
                            )
                        h_state, c_state = hidden_states
                        expected = int(self.config.encoder_hidden_size)
                        if expected == int(h_state.size(-1)):
                            hidden_states = h_state
                        elif expected == int(h_state.size(-1) * 2):
                            hidden_states = torch.cat([h_state, c_state], dim=-1)
                        else:
                            raise ValueError(
                                "Probe encoder_hidden_size does not match LSTM hidden sizes: "
                                f"{expected=} vs h_dim={h_state.size(-1)}."
                            )

        logits = self.classifier(hidden_states)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.contiguous().view(-1))
        else:
            loss = None

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )

def invert_output_layer(
    tpencoder,
    regularization: Literal['atol', 'l2', 'topk'] = 'atol',
    l2_lambda: Optional[float] = None,
    atol: Optional[float] = None,
    topk: Optional[int] = None,
    device: Optional[torch.device] = None
):
    """
    Invert the output layer of a tensor product encoder.
    
    Args:
        tpencoder: TensorProductEncoder
        regularization: str, default 'atol' - Method for regularizing the linear transformation
        l2_lambda: float, optional - Lambda for Tikhonov regularization
        atol: float, optional - Atol for pinv
        topk: int, optional - Number of singular values to keep
        device: torch.device, optional - Device to put tensors on
        
    Returns:
        tuple: (W_inv, bias) where:
            - W_inv: Inverse of output layer weights
            - bias: Bias correction term for the inverted layer
    """
    if device is None:
        device = torch.device('cpu')
        
    # Validate regularization parameters
    if regularization == 'l2' and l2_lambda is None:
        raise ValueError("L2 regularization lambda must be specified if regularization is 'l2'")
    elif regularization == 'atol' and atol is None:
        raise ValueError("Atol for pinv must be specified if regularization is 'atol'")
    elif regularization == 'topk' and topk is None:
        raise ValueError("Topk must be specified if regularization is 'topk'")
    elif regularization not in ['atol', 'l2', 'topk']:
        raise ValueError("Regularization must be one of 'atol', 'l2', or 'topk'")
    
    with torch.no_grad():
        if tpencoder.output_layer is not None:
            W_inv = regularized_pinv(
                tpencoder.output_layer.weight,
                regularization=regularization,
                l2_lambda=l2_lambda,
                atol=atol,
                topk=topk,
            ).clone().to(device)
            original_bias = tpencoder.output_layer.bias.clone().to(device)
            # Calculate the corrected bias for the inverted layer
            bias = -W_inv @ original_bias
        else:
            filler_dim = tpencoder.filler_embedding.embedding_dim
            hidden_size = filler_dim * tpencoder.role_embedding.embedding_dim
            W_inv = torch.eye(hidden_size).to(device)
            original_bias = torch.zeros(hidden_size).to(device)
            bias = torch.zeros(hidden_size).to(device)
    
    return W_inv, bias


def construct_unbinding_vectors(
    tpencoder,
    role_id: int,
    role_unbinding: Literal['pinv', 'norm'] = 'norm',
    filler_unbinding: Literal['pinv', 'norm'] = 'norm',
    role_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'l2',
    role_pinv_l2_lambda: Optional[float] = 1e-2,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    filler_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'none',
    filler_pinv_l2_lambda: Optional[float] = None,
    filler_pinv_atol: Optional[float] = None,
    filler_pinv_topk: Optional[int] = None,
    mode: Literal['embedding', 'classification'] = 'classification',
    device: Optional[torch.device] = None
):
    """
    Construct role and filler unbinding vectors and probe weights.
    
    Args:
        tpencoder: TensorProductEncoder
        role_id: int - Role ID to probe for
        role_unbinding: str, default 'norm' - Method for unbinding role vectors
        filler_unbinding: str, default 'norm' - Method for unbinding filler vectors
        role_pinv_regularization: str, default 'l2' - Regularization for role pinv
        role_pinv_l2_lambda: float, optional - Lambda for role pinv regularization
        role_pinv_atol: float, optional - Atol for role pinv cutoff
        role_pinv_topk: int, optional - Top-k singular values for role pinv
        filler_pinv_regularization: str, default 'none' - Regularization for filler pinv
        filler_pinv_l2_lambda: float, optional - Lambda for filler pinv regularization
        filler_pinv_atol: float, optional - Atol for filler pinv cutoff
        filler_pinv_topk: int, optional - Top-k singular values for filler pinv
        mode: str, default 'classification' - Mode of probe
        device: torch.device, optional - Device to put tensors on
        
    Returns:
        torch.Tensor: Probe weights for classification/embedding recovery
    """
    if device is None:
        device = torch.device('cpu')
        
    # Validate inputs
    assert role_id < tpencoder.role_embedding.num_embeddings and role_id >= 0, "Role ID out of range!"
    if role_unbinding not in ['pinv', 'norm'] or filler_unbinding not in ['pinv', 'norm']:
        raise ValueError(f"Unbinding mode must be one of 'pinv' or 'norm', found {role_unbinding=} and {filler_unbinding=}")
    
    filler_dim = tpencoder.filler_embedding.embedding_dim
    
    with torch.no_grad():
        # construct role unbinding vectors
        if role_unbinding == 'pinv':
            role_unembed = regularized_pinv(
                tpencoder.role_embedding.weight,
                regularization=role_pinv_regularization,
                l2_lambda=role_pinv_l2_lambda,
                atol=role_pinv_atol,
                topk=role_pinv_topk,
            ).to(device)
        else:
            role_unembed = tpencoder.role_embedding.weight / (tpencoder.role_embedding.weight.norm(dim=0) ** 2)
            role_unembed = role_unembed.T.clone().to(device)

        # construct filler unbinding vectors
        if filler_unbinding == 'pinv':
            assert tpencoder.filler_embedding.weight.shape[0] <= tpencoder.filler_embedding.weight.shape[1], "Filler Unbinding only works for full rank fillers"
            filler_unembed = regularized_pinv(
                tpencoder.filler_embedding.weight,
                regularization=filler_pinv_regularization,
                l2_lambda=filler_pinv_l2_lambda,
                atol=filler_pinv_atol,
                topk=filler_pinv_topk,
            ).to(device)
        else:
            filler_unembed = tpencoder.filler_embedding.weight / (tpencoder.filler_embedding.weight.norm(dim=0) ** 2)
            filler_unembed = filler_unembed.T.clone().to(device)

        # weights that recover filler
        W_probe = torch.kron(torch.eye(filler_dim).to(device), role_unembed[:, role_id])

        if mode == 'classification':
            # weights that classify
            W_probe = torch.mm(filler_unembed.T, W_probe)
        else:
            assert mode in ['embedding', 'classification'], "Inverted probe mode can only be one of embedding or classification recovery"

    return W_probe

def apply_analytic_probe(
    aggregated_binding: torch.Tensor,
    tpencoder,
    role_id: int,
    role_unbinding: Literal['pinv', 'norm'] = 'norm',
    filler_unbinding: Literal['pinv', 'norm'] = 'norm',
    regularization: Literal['atol', 'l2', 'topk'] = 'atol',
    l2_lambda: Optional[float] = None,
    atol: Optional[float] = None,
    topk: Optional[int] = None,
    role_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'none',
    role_pinv_l2_lambda: Optional[float] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    filler_pinv_regularization: Literal['none', 'atol', 'l2', 'topk'] = 'none',
    filler_pinv_l2_lambda: Optional[float] = None,
    filler_pinv_atol: Optional[float] = None,
    filler_pinv_topk: Optional[int] = None,
    mode: Literal['embedding', 'classification'] = 'classification',
    use_trained_layers: Optional[bool] = True
) -> torch.Tensor:
    """
    Apply analytic probe to aggregated binding to get classification logits or embeddings.
    
    Args:
        aggregated_binding: torch.Tensor - The aggregated TPR binding (batch_size, hidden_dim)
        tpencoder: TensorProductEncoder, TensorProductEncoderWithDecodingLoss, or TensorProductEncoderWithBackProjection
        role_id: int - Role ID to probe for
        role_unbinding: str, default 'norm' - Method for unbinding role vectors
        filler_unbinding: str, default 'norm' - Method for unbinding filler vectors
        regularization: str, default 'atol' - Method for regularizing the linear transformation
            Not used if tpencoder has trained inverse/back projection layers and use_trained_layers is not False.
        l2_lambda: float, optional - Lambda for Tikhonov regularization
        atol: float, optional - Atol for pinv
        topk: int, optional - Number of singular values to keep
        role_pinv_regularization: str, default 'none' - Regularization for role pinv
        role_pinv_l2_lambda: float, optional - Lambda for role pinv regularization
        role_pinv_atol: float, optional - Atol for role pinv cutoff
        role_pinv_topk: int, optional - Top-k singular values for role pinv
        filler_pinv_regularization: str, default 'none' - Regularization for filler pinv
        filler_pinv_l2_lambda: float, optional - Lambda for filler pinv regularization
        filler_pinv_atol: float, optional - Atol for filler pinv cutoff
        filler_pinv_topk: int, optional - Top-k singular values for filler pinv
        mode: str, default 'classification' - Mode of probe
        use_trained_layers: Optional[bool], default True - Whether to use trained inverse/back projection layers when available.
            If None or True, uses trained layers when available. If False, always computes pseudoinverse.
        
    Returns:
        torch.Tensor: Classification logits (batch_size, num_fillers) or embeddings (batch_size, filler_dim)
    """
    device = aggregated_binding.device
    
    # Handle shape (batch, 1, dim) -> (batch, dim)
    if aggregated_binding.dim() == 3:
        aggregated_binding = aggregated_binding.squeeze(1)
    
    # Determine the base TPE encoder from the input
    if hasattr(tpencoder, 'encoder') and tpencoder.encoder is not None:
        # For TensorProductEncoderWithDecodingLoss or TensorProductEncoderWithBackProjection
        base_tpencoder = tpencoder.encoder
    else:
        # For regular TensorProductEncoder
        base_tpencoder = tpencoder
    
    # Handle None case - treat as True
    should_use_trained = use_trained_layers is not False
    
    # Check for and use trained layers (inverse_layer or back_projection)
    trained_layer = None
    if should_use_trained:
        if hasattr(tpencoder, 'inverse_layer') and tpencoder.inverse_layer is not None:
            trained_layer = tpencoder.inverse_layer
        elif hasattr(tpencoder, 'back_projection') and tpencoder.back_projection is not None:
            trained_layer = tpencoder.back_projection
    
    # Get probe weights using helper functions directly
    if trained_layer is not None:
        # Use trained layer weights
        W_inv = trained_layer.weight.clone().detach().to(device)
        bias = trained_layer.bias.clone().detach().to(device)
    else:
        # Fall back to computing pseudoinverse
        W_inv, bias = invert_output_layer(
            tpencoder=base_tpencoder,
            regularization=regularization,
            l2_lambda=l2_lambda,
            atol=atol,
            topk=topk,
            device=device
        )
    
    W_probe = construct_unbinding_vectors(
        tpencoder=base_tpencoder,
        role_id=role_id,
        role_unbinding=role_unbinding,
        filler_unbinding=filler_unbinding,
        role_pinv_regularization=role_pinv_regularization,
        role_pinv_l2_lambda=role_pinv_l2_lambda,
        role_pinv_atol=role_pinv_atol,
        role_pinv_topk=role_pinv_topk,
        filler_pinv_regularization=filler_pinv_regularization,
        filler_pinv_l2_lambda=filler_pinv_l2_lambda,
        filler_pinv_atol=filler_pinv_atol,
        filler_pinv_topk=filler_pinv_topk,
        mode=mode,
        device=device
    )
    
    # Apply probe transformation using torch.nn.functional.linear (this preserves gradients)
    # First undo output layer transformation
    hidden = torch.nn.functional.linear(aggregated_binding, W_inv, bias)  # (batch, filler_dim * role_dim)
    
    # Apply probe transformation to get logits/embeddings
    logits = torch.nn.functional.linear(hidden, W_probe)  # (batch, num_fillers) or (batch, filler_dim)
    
    return logits

def auto_select_tpe_output_l2_lambda(
    tpencoder,
    target_hidden: torch.Tensor,
    filler_ids: torch.Tensor,
    role_ids: torch.Tensor,
    device: Optional[torch.device] = None,
    log_bounds: tuple[float, float] = (-12.0, 12.0),
    tolerance_ratio: float = 0.01,
):
    """Pick l2 regularization for inverting the TPE output layer by minimizing MSE.

    Args:
        tpencoder: TensorProductEncoder or wrapper exposing `.encoder`.
        target_hidden: [B, D] or [B, 1, D] hidden states to invert.
        filler_ids: [B, N] filler id tensor.
        role_ids: [B, N] role id tensor.
        device: Optional device override.
        log_bounds: Search bounds in log10 space.
        tolerance_ratio: Relative precision for ternary search.

    Returns:
        (best_lambda, best_value, (log_lo, log_hi)) with best_lambda in linear space.
    """
    with torch.no_grad():
        base = tpencoder.encoder if getattr(tpencoder, "encoder", None) is not None else tpencoder
        dev = device or base.filler_embedding.weight.device

        h = target_hidden
        if h.dim() == 3 and h.shape[1] == 1:
            h = h.squeeze(1)
        h = h.to(dev)
        fids = filler_ids.to(dev)
        rids = role_ids.to(dev)

        cache: dict[float, float] = {}

        def objective_fn(log_lambda: float) -> float:
            log_lambda = float(log_lambda)
            if log_lambda in cache:
                return cache[log_lambda]
            W_inv, bias = invert_output_layer(
                tpencoder=base,
                regularization='l2',
                l2_lambda=10.0 ** log_lambda,
                device=dev,
            )
            recovered_tpr = torch.nn.functional.linear(h, W_inv, bias)
            gt = base(filler_ids=fids, role_ids=rids).hidden_states
            if gt.dim() == 3:
                gt = gt.squeeze(1)
            mse = torch.nn.functional.mse_loss(recovered_tpr, gt).item()
            cache[log_lambda] = mse
            return mse

        best = search_reg_param(objective_fn, optimize="min", log_bounds=log_bounds, tolerance_ratio=tolerance_ratio)
        if best is None:
            raise RuntimeError("Auto-selection of l2 regularization did not yield a valid candidate.")
        best_log_lambda, best_value, (log_lo, log_hi) = best
        return 10.0 ** best_log_lambda, float(best_value), (float(log_lo), float(log_hi))


def auto_select_role_pinv_l2_lambda(
    tpencoder,
    filler_ids: torch.Tensor,
    role_ids: torch.Tensor,
    device: Optional[torch.device] = None,
    log_bounds: tuple[float, float] = (-12.0, 12.0),
    tolerance_ratio: float = 0.01,
):
    """Pick l2 regularization for role unbinding using a batch reconstruction objective.

    Args:
        tpencoder: TensorProductEncoder or wrapper exposing `.encoder`.
        filler_ids: [B, N] filler ids for the batch objective.
        role_ids: [B, N] role ids for the batch objective.
        device: Optional device override.
        log_bounds: Search bounds in log10 space.
        tolerance_ratio: Relative precision for ternary search.

    Returns:
        (best_lambda, best_value, (log_lo, log_hi)) with best_lambda in linear space.
    """
    with torch.no_grad():
        base = tpencoder.encoder if getattr(tpencoder, "encoder", None) is not None else tpencoder
        role_embs = base.role_embedding.weight
        filler_embs = base.filler_embedding.weight
        dev = device or role_embs.device
        role_embs = role_embs.to(dev)
        filler_embs = filler_embs.to(dev)
        fids = filler_ids.to(dev)
        rids = role_ids.to(dev)
        tpr = base(filler_ids=fids, role_ids=rids).hidden_states
        if tpr.dim() == 3:
            tpr = tpr.squeeze(1)
        tpr = tpr.view(fids.shape[0], filler_embs.shape[1], role_embs.shape[1])

        cache: dict[float, float] = {}

        def objective_fn(log_lambda: float) -> float:
            log_lambda = float(log_lambda)
            if log_lambda in cache:
                return cache[log_lambda]
            role_unembed = regularized_pinv(
                role_embs,
                regularization="l2",
                l2_lambda=10.0 ** log_lambda,
            )
            role_unembed_t = role_unembed.T
            u = role_unembed_t[rids]
            unbound = torch.einsum("bfr,bnr->bnf", tpr, u)
            targets = filler_embs[fids]
            mse = torch.nn.functional.mse_loss(unbound, targets).item()
            cache[log_lambda] = mse
            return mse

        best = search_reg_param(
            objective_fn,
            optimize="min",
            log_bounds=log_bounds,
            tolerance_ratio=tolerance_ratio,
        )
        if best is None:
            raise RuntimeError("Auto-selection of role unbinding l2 regularization did not yield a valid candidate.")
        best_log_lambda, best_value, (log_lo, log_hi) = best
        return 10.0 ** best_log_lambda, float(best_value), (float(log_lo), float(log_hi))
