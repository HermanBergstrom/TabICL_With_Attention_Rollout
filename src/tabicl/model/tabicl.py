from __future__ import annotations

from typing import Optional, List
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .inference_config import InferenceConfig
import torch


class TabICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    For datasets with more than `max_classes` classes, TabICL switches to hierarchical lassification
    to recursively partition classes into subgroups, forming a multi-level classification tree.

    Parameters
    ----------
    max_classes : int, default=10
        Number of classes that the model supports natively. If the number of classes
        in the dataset exceeds this value, hierarchical classification is used.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer

    col_nhead : int, default=4
        Number of attention heads in the column embedding transformer

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer

    icl_nhead : int, default=4
        Number of attention heads in the in-context learning transformer

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components

    dropout : float, default=0.0
        Dropout probability across all components

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components
    """

    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=row_num_cls,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL
        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

    def _icl_retrieval_forward(
        self, representations, y_train, train_size, k=None, return_logits=True, softmax_temperature=0.9, inference_config=None, return_attention_rollout=False
    ):

        B, T, D = representations.shape
        train_reps = representations[:, :train_size, :]  # (B, train_size, D)
        test_reps = representations[:, train_size:, :]   # (B, test_size, D)
        
        # Compute pairwise distances between test and train representations
        dists = torch.cdist(test_reps, train_reps, p=2)  # (B, test_size, train_size)

        k = min(k, train_size) 

        # For each test sample, find the k closest training samples
        knn_dists, knn_indices = torch.topk(dists, k=k, dim=-1, largest=False)  # (B, test_size, k)

        #Compare against random retrieval
        random_indices = torch.randint(0, train_size, (B, test_reps.size(1), k), device=representations.device)  # (B, test_size, k)

        # Gather the representations of the k nearest neighbors
        knn_reps = torch.gather(
            train_reps.unsqueeze(1).expand(-1, test_reps.size(1), -1, -1),  # (B, test_size, train_size, D)
            2,
            knn_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, test_size, k, D)
        )  # (B, test_size, k, D)

        knn_labels = torch.gather(
            y_train.unsqueeze(1).expand(-1, test_reps.size(1), -1),  # (B, test_size, train_size)
            2,
            knn_indices  # (B, test_size, k)
        )  # (B, test_size, k)

        # For each test sample, make a call to the icl_predictor with its k nearest neighbors as context
        B, test_size, _ = test_reps.shape
        
        # Reshape to (B*test_size, k, D) for batch processing
        context_samples_flat = knn_reps.reshape(B * test_size, k, D)  # (B*test_size, k, D)
        test_samples_flat = test_reps.reshape(B * test_size, 1, D)  # (B*test_size, 1, D)
        
        # Combine context samples and test samples
        icl_input = torch.cat([context_samples_flat, test_samples_flat], dim=1)  # (B*test_size, k+1, D)
        
        # Reshape labels to (B*test_size, k)
        y_train_flat = knn_labels.reshape(B * test_size, k)  # (B*test_size, k)
        
        use_icl_microbatching = False
        icl_microbatch_size = 64  # Adjust based on memory constraints
        if not use_icl_microbatching:
            print("Using single forward pass for ICL retrieval")
            icl_input = torch.cat(
                [context_samples_flat, test_samples_flat], dim=1
            )

            icl_output = self.icl_predictor(
                icl_input,
                y_train=y_train_flat,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=inference_config.ICL_CONFIG,
                return_attention_rollout=return_attention_rollout,
            )  # (B*test_size, k+1, num_classes)

            if return_attention_rollout:
                test_logits_flat = icl_output[0][:, -1, :]
                attention_rollout = icl_output[1]
            else:
                test_logits_flat = icl_output[:, -1, :]
        else:
            print("Using microbatching for ICL retrieval")
            n_classes = torch.unique(y_train).size(0)
            self.icl_predictor.n_classes = n_classes

            outputs = []
            total = context_samples_flat.size(0)

            for start in range(0, total, icl_microbatch_size):
                end = min(start + icl_microbatch_size, total)

                icl_input = torch.cat(
                    [
                        context_samples_flat[start:end],
                        test_samples_flat[start:end],
                    ],
                    dim=1,
                )
  
                icl_out = self.icl_predictor(
                    icl_input,
                    y_train=y_train_flat[start:end],
                    return_logits=return_logits,
                    softmax_temperature=softmax_temperature,
                    mgr_config=inference_config.ICL_CONFIG,
                    return_attention_rollout=return_attention_rollout,
                )

                if return_attention_rollout:
                    outputs.append(icl_out[0][:, -1, :])
                else:
                    outputs.append(icl_out[:, -1, :])

            test_logits_flat = torch.cat(outputs, dim=0)

        # Reshape back to (B, test_size, num_classes)
        icl_outputs = test_logits_flat.reshape(B, test_size, -1)
        
        if return_attention_rollout:
            return icl_outputs, attention_rollout
        else:
            return icl_outputs

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset.

        Returns
        -------
        Tensor
            Raw logits of shape (B, T, max_classes), which will be further handled by the training code.
        """

        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Column-wise embedding -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder(X, d=d, train_size=None if embed_with_test else train_size), d=d
        )

        # Dataset-wise in-context learning
        out = self.icl_predictor(representations, y_train=y_train)

        return out

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
        return_attention_rollout: bool = False,
        return_row_emb: bool = False,
        k: Optional[int] = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        inference_config: InferenceConfig
            Inferenece configuration

        return_attention_rollout : bool, default=False
            Whether to return attention rollout from the row interaction transformer

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            If return_attention_rollout=False: Raw logits or probabilities for test samples of shape (B, test_size, num_classes)
            If return_attention_rollout=True: Tuple of (logits/probabilities, attention rollout matrix)
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        # Column-wise embedding -> Row-wise interaction
        row_result = self.row_interactor(
            self.col_embedder(
                X,
                train_size=None if embed_with_test else train_size,
                feature_shuffles=feature_shuffles,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
            return_attention_rollout=return_attention_rollout,
        )

        if return_attention_rollout:
            representations, row_emb_rollout = row_result
        else:
            representations = row_result

        #save representations for debugging
        #torch.save(representations, '/home/hermanb/scratch/TabICL_Experiments/debugging/representations.pt')
        #Employ knn to determine context for each test sample
        context_retrieval = k is not None
        if context_retrieval:
            out = self._icl_retrieval_forward(
                representations,
                y_train,
                train_size,
                k=k,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
                return_attention_rollout=return_attention_rollout,
            )

            return out
        
        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
            return_attention_rollout=return_attention_rollout,
        )

        if return_attention_rollout:
            out, icl_rollout = out
            if return_row_emb:
                return out, (row_emb_rollout, icl_rollout), representations
            else:
                return out, (row_emb_rollout, icl_rollout), None
        else:
            return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
        return_attention_rollout: bool = False,
        return_row_emb: bool = False,
        k: Optional[int] = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in training mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in training mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in training mode.

        inference_config: InferenceConfig
            Inferenece configuration. Used only in training mode.

        return_attention_rollout : bool, default=False
            Whether to return attention rollout from the row interaction transformer. Used only in inference mode.

        return_row_emb : bool, default=False
            Whether to return row embeddings from the row interaction transformer. Used only in inference mode.

        Returns
        -------
        Tensor | tuple[Tensor, tuple[Tensor, Tensor], Tensor]
            For training mode:
              Raw logits of shape (B, T-train_size, max_classes), which will be further handled by the training code.

            For inference mode:
              If return_attention_rollout=False: Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes).
              If return_attention_rollout=True: Tuple of (logits/probabilities, attention rollout matrix)
        """

        if self.training:
            out = self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
            return out
        else:
            result = self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
                return_attention_rollout=return_attention_rollout,
                return_row_emb=return_row_emb,
                k=k,
            )
            return result
