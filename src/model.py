from typing import Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from loss import WeightedMSELoss
import transformers
import math
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig
from transformers import EncoderDecoderConfig
from transformers import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput, Seq2SeqLMOutput
from probing import invert_output_layer, construct_unbinding_vectors

## Encoder/Decoder & TPR implementations compatible w/ HF Hub


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Code taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class RecurrentEncoderDecoderModel(PreTrainedModel, GenerationMixin):
    """
    Simplified encoder-decoder model, similar to EncoderDecoderModel provided by Huggingface Transformers
    """

    config_class = EncoderDecoderConfig  # Shares config as one in hub

    def __init__(self, config, encoder: Optional[PreTrainedModel] = None, decoder: Optional[PreTrainedModel] = None):
        if config is None:
            raise ValueError("config is required")

        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)
        if decoder is None:
            decoder = AutoModel.from_config(config.decoder)

        self.encoder, self.decoder = encoder, decoder

        self.encoder.config = config.encoder
        self.decoder.config = config.decoder
        self.config.pad_token_id = self.decoder.config.pad_token_id
        self.config.bos_token_id = self.decoder.config.bos_token_id
        self.config.eos_token_id = self.decoder.config.eos_token_id
        self.config.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        # RNN decoder does not support KV caching; disable to keep generation correct.
        self.config.use_cache = False

        self.post_init()
        if hasattr(self, "generation_config"):
            self.generation_config.use_cache = False

    @classmethod
    def from_encoder_decoder_pretrained(cls, encoder, decoder, *model_args, **model_kwargs):
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        return cls(config, encoder, decoder, *model_args, **model_kwargs)

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        model = super().from_pretrained(*model_args, **kwargs)
        # RNN decoder does not support KV caching; disable to keep generation correct.
        model.config.use_cache = False
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False
        return model

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_lengths: Optional[torch.LongTensor] = None,
        filler_ids: Optional[torch.LongTensor] = None,
        role_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_input_lengths: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ):
        # Compute encoder outputs with input_lengths
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, filler_ids=filler_ids, role_ids=role_ids, input_lengths=input_lengths
            )

        # Prepare decoder inputs
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.decoder.pad_token_id, self.config.decoder.decoder_start_token_id
            )
        decoder_input_lengths = (decoder_input_ids != self.config.decoder.pad_token_id).sum(dim=1)

        # Get decoder outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            input_lengths=decoder_input_lengths,
            hidden=encoder_outputs.last_hidden_state,
            return_hidden_states=output_hidden_states,
        )

        # Compute loss
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss(ignore_index=self.decoder.config.pad_token_id)(
                decoder_outputs.logits.view(-1, self.decoder.config.vocab_size), labels.view(-1)
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
        )


class RecurrentEncoderConfig(PretrainedConfig):
    model_type = "recurrent_encoder"

    def __init__(
        self,
        architecture: Literal["GRU", "RNN", "LSTM"] = "RNN",
        vocab_size: int = 100,
        embedding_size: int = 32,
        hidden_size: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert architecture in ["GRU", "RNN", "LSTM"], (
            f"Unsupported architecture: {architecture}, must be one of ['GRU', 'RNN', 'LSTM']"
        )

        self.architecture = architecture
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.dropout = dropout
        self.pad_token_id = pad_token_id


class RecurrentEncoder(PreTrainedModel):
    config_class = RecurrentEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_size)

        if config.architecture == "GRU":
            self.recurrent_unit = torch.nn.GRU(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.architecture == "RNN":
            self.recurrent_unit = torch.nn.RNN(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.architecture == "LSTM":
            self.recurrent_unit = torch.nn.LSTM(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                batch_first=True,
            )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: Optional[torch.LongTensor],
        return_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithNoAttention:
        # Sort input sequences by length
        if input_lengths is None:
            input_lengths = (input_ids != self.config.pad_token_id).sum(dim=1)
            
        sorted_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
        sorted_input_ids = input_ids[sorted_indices]

        embedded = self.embedding(sorted_input_ids)

        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        packed_output, hidden = self.recurrent_unit(packed_embedded)

        # Unpack the output
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        # Unsort the output and hidden states
        _, unsorted_indices = torch.sort(sorted_indices)
        unsorted_output = output[unsorted_indices]

        if isinstance(hidden, tuple):
            h, c = hidden
            h = h[:, unsorted_indices, :]
            c = c[:, unsorted_indices, :]
            # num layers x batch x hidden to batch x num layers x hidden
            unsorted_hidden = (h.transpose(0, 1), c.transpose(0, 1))
        else:
            unsorted_hidden = hidden[:, unsorted_indices, :].transpose(0, 1)

        # Keep the tuple or tensor in hidden_states if requested
        hidden_states = None
        if return_hidden_states:
            hidden_states = unsorted_output

        return BaseModelOutputWithNoAttention(
            last_hidden_state=unsorted_hidden,
            hidden_states=hidden_states,
        )


class RecurrentDecoderConfig(PretrainedConfig):
    model_type = "recurrent_decoder"

    def __init__(
        self,
        architecture: Literal["GRU", "RNN", "LSTM"] = "RNN",
        vocab_size: int = 100,
        embedding_size: int = 32,
        hidden_size: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        decoder_start_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert architecture in ["GRU", "RNN", "LSTM"], (
            f"Unsupported architecture: {architecture}, must be one of ['GRU', 'RNN', 'LSTM']"
        )

        self.architecture = architecture
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        if decoder_start_token_id == None:
            # models use bos_token_id as default decoding start token
            self.decoder_start_token_id = bos_token_id
        else:
            self.decoder_start_token_id = decoder_start_token_id


class RecurrentDecoder(PreTrainedModel):
    config_class = RecurrentDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_size)

        if config.architecture == "GRU":
            self.recurrent_unit = torch.nn.GRU(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.architecture == "RNN":
            self.recurrent_unit = torch.nn.RNN(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.architecture == "LSTM":
            self.recurrent_unit = torch.nn.LSTM(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout,
                batch_first=True,
            )

        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        hidden: torch.FloatTensor,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> CausalLMOutput:
        if self.config.architecture == "LSTM" and not isinstance(hidden, tuple):
            # Generation supplies a batch x hidden tensor; expand to batch x layers x hidden.
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1)

            if hidden.size(-1) == 2 * self.config.hidden_size:
                # TPE encoders can return concatenated (h, c) states.
                hidden = hidden.split(self.config.hidden_size, dim=-1)
            elif hidden.size(-1) == self.config.hidden_size:
                # If only h is present, provide a zero c state for LSTM compatibility.
                hidden = (hidden, torch.zeros_like(hidden))
            else:
                raise ValueError(
                    "LSTM decoder expected hidden size matching hidden_size or 2*hidden_size; "
                    f"got last-dim={hidden.size(-1)} with hidden_size={self.config.hidden_size}."
                )

        # Sort inputs by length
        sorted_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
        sorted_input_ids = input_ids[sorted_indices]

        # Sort hidden state
        if isinstance(hidden, tuple):
            h, c = hidden
            h_sorted = h[sorted_indices, :, :].transpose(0, 1)
            c_sorted = c[sorted_indices, :, :].transpose(0, 1)
            sorted_hidden = (h_sorted, c_sorted)
        else:
            sorted_hidden = hidden[sorted_indices, :, :].transpose(0, 1)
            
        # Embed sorted inputs
        embedded = self.embedding(sorted_input_ids)

        # Pack sequences
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # Forward pass through RNN
        if isinstance(sorted_hidden, tuple):
            packed_output, _ = self.recurrent_unit(
                packed_embedded,
                (sorted_hidden[0], sorted_hidden[1])
            )
        else:
            packed_output, _ = self.recurrent_unit(
                packed_embedded,
                sorted_hidden
            )

        # Unpack output
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        # Unsort outputs
        _, unsorted_indices = torch.sort(sorted_indices)
        unsorted_output = output[unsorted_indices]

        # Compute logits
        logits = self.lm_head(unsorted_output)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        return CausalLMOutput(
            logits=logits,
            hidden_states=unsorted_output if return_hidden_states else None,
        )


class OuterProduct(nn.Module):
    """
    Outer product layer (used for filler-role bindings)
    for tensor product representations
    """

    def __init__(self, reduction: str = "sum"):
        super(OuterProduct, self).__init__()
        assert reduction in ["sum", "mean"], f"Unsupported reduction: {reduction}, must be one of ['sum', 'mean']"
        self.reduction = reduction

    def forward(self, input1, input2):
        """
        Forward pass for outer product layer.
        Args:
            input1 (torch.Tensor): Tensor of shape (batch_size, seq_len, filler_dim)
            input2 (torch.Tensor): Tensor of shape (batch_size, seq_len, role_dim)
        Returns:
            bindings (torch.Tensor): Tensor of shape (batch_size, seq_len, filler_dim * role_dim), encoding of each input element.
            outputs (torch.Tensor): Tensor of shape (batch_size, 1, filler_dim * role_dim), aggregated over seq_len
        """
        einsum = torch.einsum("blf,blr->blfr", (input1, input2))
        bindings = einsum.view(einsum.shape[0], einsum.shape[1], -1)
        outputs = torch.sum(bindings, dim=1).unsqueeze(1)

        if self.reduction == "mean":
            mask = (outputs != 0).float()
            mask = torch.sum(mask, dim=1).unsqueeze(0)
            outputs = outputs / mask

        return bindings, outputs


class SumFlattenedOuterProduct(nn.Module):
    """
    Equivalent to OuterProduct with reduction='sum' followed by flattening the
    output; more efficient implementation because operations are combined.
    """

    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()

    def forward(self, input1, input2):
        sum_outer_product = torch.bmm(input1.transpose(1, 2), input2)
        flattened_sum_outer_product = sum_outer_product.view(sum_outer_product.size()[0], -1).unsqueeze(1)
        return None, flattened_sum_outer_product


class TensorProductEncoderConfig(PretrainedConfig):
    model_type = "tensor_product_encoder"

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        n_fillers: Optional[int] = None,
        n_roles: Optional[int] = None,
        filler_dim: Optional[int] = None,
        role_dim: Optional[int] = None,
        filler_pad_token_id: Optional[int] = None,
        role_pad_token_id: Optional[int] = None,
        has_linear_layer: bool = True,
        return_bindings: bool = False,
        aggregation: str = "sum",
        role_scheme : Optional[str] = None,
        layer_id: Optional[int] = None,
        target_sequence_length: Optional[int] = None,
        per_token_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.n_fillers = n_fillers
        self.n_roles = n_roles
        self.filler_dim = filler_dim
        self.role_dim = role_dim
        self.filler_pad_token_id = filler_pad_token_id
        self.role_pad_token_id = role_pad_token_id
        self.has_linear_layer = has_linear_layer
        self.return_bindings = return_bindings
        self.aggregation = aggregation
        self.role_scheme = role_scheme
        self.layer_id = layer_id
        self.target_sequence_length = target_sequence_length
        self.per_token_hidden_size = per_token_hidden_size

class TensorProductEncoder(PreTrainedModel):
    config_class = TensorProductEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.filler_embedding = nn.Embedding(
            config.n_fillers, config.filler_dim, padding_idx=config.filler_pad_token_id
        )
        self.role_embedding = nn.Embedding(config.n_roles, config.role_dim, padding_idx=config.role_pad_token_id)

        assert config.aggregation in ["sum", "mean"], (
            f"Unsupported aggregation: {config.aggregation}, must be one of ['sum', 'mean']"
        )
        if config.aggregation == "sum":
            if config.return_bindings:
                self.bind_and_aggregate_layer = OuterProduct(reduction="sum")
            else:
                self.bind_and_aggregate_layer = SumFlattenedOuterProduct()
        else:
            self.bind_and_aggregate_layer = OuterProduct(reduction="mean")

        if config.has_linear_layer:
            self.output_layer = nn.Linear(config.filler_dim * config.role_dim, config.hidden_size)
            # Initialize with unit norm weights
            # with torch.no_grad():
            #     weight_norm = torch.norm(self.output_layer.weight, dim=1, keepdim=True)
            #     self.output_layer.weight.div_(weight_norm)
        else:
            self.output_layer = None

        self.post_init()

    def forward(
        self, filler_ids: torch.LongTensor, role_ids: torch.LongTensor, **kwargs
    ) -> BaseModelOutputWithNoAttention:
        """
        Forward pass for tensor product encoder.
        Args:
            filler_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len) containing the filler IDs.
            role_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len) containing the role IDs.
        Returns:
            BaseModelOutputWithNoAttention: An object containing the last hidden state and hidden states.
             - last_hidden_state (torch.Tensor): The aggregated binding, shape
               (batch_size, 1, filler_dim * role_dim).
             - hidden_states (torch.Tensor): The bindings, shape (batch_size,
              seq_len, filler_dim * role_dim).
        """

        fillers = self.filler_embedding(filler_ids)
        roles = self.role_embedding(role_ids)

        # Get TPR representation; if return_bindings is False, bindings will be None
        # bindings: (batch_size, seq_len, filler_dim * role_dim)
        # aggregated_binding: (batch_size, 1, filler_dim * role_dim)
        bindings, aggregated_binding = self.bind_and_aggregate_layer(fillers, roles)

        if self.output_layer is not None:
            output_binding = self.output_layer(aggregated_binding)
            if self.config.return_bindings:
                bindings = self.output_layer(bindings)
        else:
            output_binding = aggregated_binding

        return BaseModelOutputWithNoAttention(
            last_hidden_state=output_binding,
            hidden_states=aggregated_binding,
        )


class TensorProductEncoderForPretraining(PreTrainedModel):
    """
    Tensor product encoder for pretraining. Forward pass calculates loss
    between aggregated bindings and target embeddings.
    """

    config_class = TensorProductEncoderConfig

    def __init__(
        self,
        config: PretrainedConfig,
        tpencoder: Optional[TensorProductEncoder] = None,
        embedding_model: Optional[PreTrainedModel] = None,
        loss_fn: Optional[torch.nn.Module] = None,
    ):
        super().__init__(config)

        if embedding_model is not None:
            self.config.embedding_model_config = embedding_model.config
        if loss_fn is not None:
            self.config.loss_fn = loss_fn

        if tpencoder is None:
            self.encoder = TensorProductEncoder(config)
        else:
            self.encoder = tpencoder

        if embedding_model is None and 'embedding_model_config' in config:
            embedding_model_config = AutoConfig.for_model(**config.embedding_model_config)
            embedding_model = AutoModel.from_config(embedding_model_config)

        self.embedding_model = embedding_model
        self.loss_fn = loss_fn
        self._cached_target_embeddings: Optional[torch.FloatTensor] = None
        self._cached_raw_tpr_binding: Optional[torch.FloatTensor] = None

    @classmethod
    def from_tpencoder(cls, tpencoder):
        config = tpencoder.config
        return cls(config, tpencoder)

    def _compute_target_embeddings(
        self,
        target_embeddings: Optional[torch.FloatTensor],
        embedding_model_input_ids: Optional[torch.LongTensor],
        embedding_model_input_lengths: Optional[torch.LongTensor],
    ) -> Optional[torch.FloatTensor]:
        """Compute target embeddings if not provided, handling LSTM tuple format."""
        if target_embeddings is None:
            if self.embedding_model and embedding_model_input_ids is not None:
                self.embedding_model.eval()
                with torch.no_grad():
                    target_embeddings = self.embedding_model(
                        input_ids=embedding_model_input_ids,
                        input_lengths=embedding_model_input_lengths,
                    ).last_hidden_state

        if isinstance(target_embeddings, tuple):
            # LSTM encoders return a tuple of (hidden_state, cell_state).
            # Concatenate both so gradients reflect the full state.
            target_embeddings = torch.cat([target_embeddings[0], target_embeddings[1]], dim=-1)

        return target_embeddings

    def forward(
        self,
        filler_ids: torch.LongTensor,
        role_ids: torch.LongTensor,
        embedding_model_input_ids: Optional[torch.LongTensor] = None,
        embedding_model_input_lengths: Optional[torch.LongTensor] = None,
        target_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # get TPR representation
        outputs = self.encoder(filler_ids, role_ids)
        aggregated_binding = outputs.last_hidden_state

        # compute target embeddings once and store for potential reuse by subclasses
        target_embeddings = self._compute_target_embeddings(
            target_embeddings, embedding_model_input_ids, embedding_model_input_lengths
        )
        
        # Store for subclasses to reuse (avoid recomputation)
        self._cached_target_embeddings = target_embeddings
        self._cached_raw_tpr_binding = outputs.hidden_states  # Store raw TPR for back-projection

        loss = None
        if target_embeddings is not None:
            if self.loss_fn is not None:
                loss = self.loss_fn(aggregated_binding.view(-1), target_embeddings.view(-1))
            else:
                loss = torch.nn.MSELoss()(aggregated_binding.view(-1), target_embeddings.view(-1))
                # auxilary loss to penalize non-orthogonal embeddings
                if 'filler_lambda' in self.config:
                    lambda_ = self.config.filler_lambda
                    filler_embeds = self.encoder.filler_embedding.weight
                    loss += lambda_ * torch.nn.CosineSimilarity(dim=1)(filler_embeds, filler_embeds).abs().mean()
                if 'role_lambda' in self.config:
                    lambda_ = self.config.role_lambda
                    role_embeds = self.encoder.role_embedding.weight
                    loss += lambda_ * torch.nn.CosineSimilarity(dim=1)(role_embeds, role_embeds).abs().mean()

        return Seq2SeqLMOutput(encoder_hidden_states=aggregated_binding, loss=loss)


class TensorProductEncoderWithDecodingLoss(TensorProductEncoderForPretraining):
    """
    Tensor product encoder with dynamic probe loss. Probe weights are reconstructed every forward pass
    using 'norm' unbinding for both filler and role, and matrix inverse for the output layer (no grad).
    Probe loss is added to the total loss, and gradients flow through encoder and embeddings.
    """
    def __init__(
            self, 
            config, 
            tpencoder=None, 
            embedding_model=None, 
            loss_fn=None, 
            ):
        super().__init__(config, tpencoder, embedding_model, loss_fn)

        self.train_inverse_layer = config.train_inverse_layer if hasattr(config, 'train_inverse_layer') else True
        assert hasattr(config, 'role_id'), "role_id must be specified in the config"
        self.role_id = config.role_id

        self.probe_loss_weight = config.probe_loss_weight if hasattr(config, 'probe_loss_weight') else 1.0
        self.reconstruction_loss_weight = config.reconstruction_loss_weight if hasattr(config, 'reconstruction_loss_weight') else 1.0
        self.inverse_layer_loss_weight = config.inverse_layer_loss_weight if hasattr(config, 'inverse_layer_loss_weight') else 1.0

        if self.train_inverse_layer and self.encoder.output_layer is not None:
            # initialize linear layer that inverts the TPE linear projection
            self.inverse_layer = nn.Linear(
                self.encoder.output_layer.out_features,
                self.encoder.output_layer.in_features,
            )


    def forward(
        self,
        filler_ids: torch.LongTensor,
        role_ids: torch.LongTensor,
        embedding_model_input_ids: Optional[torch.LongTensor] = None,
        embedding_model_input_lengths: Optional[torch.LongTensor] = None,
        target_embeddings: Optional[torch.FloatTensor] = None,
        probe_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Get base TPR output and reconstruction loss
        tpr_output = super().forward(
            filler_ids=filler_ids,
            role_ids=role_ids,
            embedding_model_input_ids=embedding_model_input_ids,
            embedding_model_input_lengths=embedding_model_input_lengths,
            target_embeddings=target_embeddings,
            **kwargs
        )
        
        aggregated_binding = tpr_output.encoder_hidden_states

        # Get the aggregated binding output
        probe_loss = 0.0
        logits = None
        if probe_labels is not None and self.training:
            device = aggregated_binding.device
        
            # Handle shape (batch, 1, dim) -> (batch, dim)
            if aggregated_binding.dim() == 3:
                aggregated_binding = aggregated_binding.squeeze(1)
            
            # Get probe weights 
            if self.train_inverse_layer and self.encoder.output_layer is not None:
                W_inv, bias = self.inverse_layer.weight, self.inverse_layer.bias
            else:
                W_inv, bias = invert_output_layer(
                    tpencoder=self.encoder,
                    regularization='l2',
                    l2_lambda=math.sqrt(tpr_output.loss.item()) if tpr_output.loss is not None else 0.0,
                    device=device
                )
            
            W_probe = construct_unbinding_vectors(
                tpencoder=self.encoder,
                role_id=self.role_id,
                role_unbinding='norm',
                filler_unbinding='norm',
                mode='classification',
                device=device
            )
            
            # Apply probe transformation using torch.nn.functional.linear (this preserves gradients)
            # First undo output layer transformation
            hidden = torch.nn.functional.linear(aggregated_binding, W_inv, bias)  # (batch, filler_dim * role_dim)
            
            # Apply probe transformation to get logits/embeddings
            logits = torch.nn.functional.linear(hidden, W_probe)  # (batch, num_fillers) or (batch, filler_dim)
            
            # Compute probe loss if labels are provided
            probe_loss = torch.nn.CrossEntropyLoss()(logits, probe_labels.view(-1))
                
        identity_loss = 0.0
        # Compute inverse layer loss if applicable
        if self.train_inverse_layer and self.encoder.output_layer is not None and self.training:
            W1, W2 = self.encoder.output_layer.weight, self.inverse_layer.weight
            I = torch.eye(self.encoder.output_layer.in_features, device=W1.device)
            b1, b2 = self.encoder.output_layer.bias, self.inverse_layer.bias
            
            identity_loss = F.mse_loss(W2 @ W1, I) + \
                F.mse_loss(W2 @ b1 + b2, torch.zeros_like(b2)) 
             
        # Combine losses
        reconstruction_loss = tpr_output.loss
        if reconstruction_loss is None:
            total_loss = None # skip loss if no reconstruction loss is provided
        else:
            total_loss = reconstruction_loss * self.reconstruction_loss_weight + \
                probe_loss * self.probe_loss_weight + \
                identity_loss * self.inverse_layer_loss_weight
        
        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=logits,
            encoder_hidden_states=tpr_output.encoder_hidden_states,
        )


class TensorProductEncoderWithBackProjection(TensorProductEncoderForPretraining):
    """
    Tensor product encoder with back-projection loss. This class adds a back-projection layer
    that maps from the projected hidden space back to the TPR space, enabling additional
    regularization through consistency losses.
    """
    
    def __init__(
        self,
        config: PretrainedConfig,
        tpencoder: Optional[TensorProductEncoder] = None,
        embedding_model: Optional[PreTrainedModel] = None,
        loss_fn: Optional[torch.nn.Module] = None,
    ):
        super().__init__(config, tpencoder, embedding_model, loss_fn)
        
        # Configuration for back-projection loss
        self.back_projection_loss_weight = getattr(config, 'back_projection_loss_weight', 1.0)
        self.back_projection_loss_variant = getattr(config, 'back_projection_loss_variant', 'tpr_to_gt')
        
        if self.encoder.output_layer is None:
            raise ValueError(
                "TensorProductEncoderWithBackProjection requires the encoder to have an output_layer "
                "(linear projection layer). Set has_linear_layer=True in your TPE config, or use "
                "TensorProductEncoderForPretraining instead."
            )
        
        # Create back-projection layer
        self.back_projection = nn.Linear(
            self.encoder.output_layer.out_features,  # hidden_size
            self.encoder.output_layer.in_features,   # filler_dim * role_dim
        )
    
    def forward(
        self,
        filler_ids: torch.LongTensor,
        role_ids: torch.LongTensor,
        embedding_model_input_ids: Optional[torch.LongTensor] = None,
        embedding_model_input_lengths: Optional[torch.LongTensor] = None,
        target_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Validate inputs for training mode
        if self.training and target_embeddings is None and embedding_model_input_ids is None:
            raise ValueError(
                "In training mode, either 'target_embeddings' or 'embedding_model_input_ids' "
                "must be provided to compute target embeddings."
            )
        
        # Get base TPR output and reconstruction loss
        tpr_output = super().forward(
            filler_ids=filler_ids,
            role_ids=role_ids,
            embedding_model_input_ids=embedding_model_input_ids,
            embedding_model_input_lengths=embedding_model_input_lengths,
            target_embeddings=target_embeddings,
            **kwargs
        )
        
        # Use cached target embeddings from parent class
        cached_target_embeddings = getattr(self, '_cached_target_embeddings', None)
        
        # Early return if back-projection is disabled or not applicable
        if not self._should_compute_back_projection_loss(cached_target_embeddings):
            return tpr_output
        
        # Compute back-projection loss using cached embeddings (no recomputation)
        back_projection_loss = self._compute_back_projection_loss(cached_target_embeddings)
        
        # Combine losses
        total_loss = tpr_output.loss + self.back_projection_loss_weight * back_projection_loss
        
        return Seq2SeqLMOutput(
            loss=total_loss,
            encoder_hidden_states=tpr_output.encoder_hidden_states,
        )
    
    def _should_compute_back_projection_loss(self, target_embeddings: Optional[torch.FloatTensor]) -> bool:
        """Check if back-projection loss should be computed."""
        return (
            target_embeddings is not None and 
            self.training
        )
    
    def _flatten_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Helper to flatten (batch, 1, dim) -> (batch, dim) if needed."""
        return tensor.squeeze(1) if tensor.dim() == 3 else tensor
    
    def _prepare_target_embeddings(self, target_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """Prepare target embeddings for loss computation."""
        # Handle LSTM tuple format
        if isinstance(target_embeddings, tuple):
            target_embeddings = torch.cat([target_embeddings[0], target_embeddings[1]], dim=-1)
        
        return self._flatten_tensor(target_embeddings)
    
    def _compute_back_projection_loss(
        self, 
        target_embeddings: torch.FloatTensor
    ) -> torch.Tensor:
        """Compute back-projection loss based on the configured variant."""
        # Use cached raw TPR binding (already computed in parent forward)
        raw_tpr_binding = self._flatten_tensor(self._cached_raw_tpr_binding)
        
        # Prepare target embeddings
        target_embeddings_flat = self._prepare_target_embeddings(target_embeddings)
        
        # Compute loss based on variant
        if self.back_projection_loss_variant == 'tpr_to_gt':
            # Compare raw TPR with back-projected ground truth
            back_projected_gt = self.back_projection(target_embeddings_flat)
            return F.mse_loss(raw_tpr_binding, back_projected_gt)
            
        elif self.back_projection_loss_variant == 'gt_to_projected':
            # Compare ground truth with re-projected back-projection
            back_projected_gt = self.back_projection(target_embeddings_flat)
            re_projected = self.encoder.output_layer(back_projected_gt)
            return F.mse_loss(target_embeddings_flat, re_projected)
            
        else:
            raise ValueError(
                f"Unknown back_projection_loss_variant: '{self.back_projection_loss_variant}'. "
                f"Valid options are: 'tpr_to_gt', 'gt_to_projected'"
            )


# Registering models and configs for compatibility with AutoClasses
AutoConfig.register("tensor_product_encoder", TensorProductEncoderConfig)
AutoModel.register(TensorProductEncoderConfig, TensorProductEncoder)
AutoModel.register(TensorProductEncoderConfig, TensorProductEncoderForPretraining)
AutoConfig.register("recurrent_encoder", RecurrentEncoderConfig)
AutoModel.register(RecurrentEncoderConfig, RecurrentEncoder)
AutoConfig.register("recurrent_decoder", RecurrentDecoderConfig)
AutoModel.register(RecurrentDecoderConfig, RecurrentDecoder)
