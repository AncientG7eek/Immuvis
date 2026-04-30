"""Attention-Based Multiple Instance Learning (ABMIL) head.

Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning",
ICML 2018. https://arxiv.org/abs/1802.04712

Drop-in replacement for CropClassifierHead. Plugs into FinetuningModel via
the TaskHead interface:

    head = ABMILHead(
        input_dim=config.encoder_config.pm_embedding_dims[-1],
        num_classes=num_classes,
    )
    model = FinetuningModel(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        head=head,
    )

Everything else — training loop, collate, checkpoint loading — stays the same.

-----------------------------------------------------------------------
WHAT TO TEST (see _smoke_test() at the bottom of this file)
-----------------------------------------------------------------------

1. Output shape
   ABMILHead(E, K).forward(Tensor[N, E]) → Tensor[1, K]
   Must hold for any N (variable bag size).

2. Attention weights
   weights = head.attention_weights(Tensor[N, E])
   weights.shape == (N, 1)
   weights.sum()  ≈ 1.0  (softmax normalisation)
   (weights >= 0).all()

3. Gradient flow
   logits.sum().backward() must not raise and
   head.attention_V[0].weight.grad must not be None.

4. Variable bag size
   Run forward with N=1, N=10, N=961 — no shape errors.

5. Gated vs standard
   ABMILHead(..., gated=True)  — attention_U submodule exists
   ABMILHead(..., gated=False) — attention_U submodule absent,
                                  forward still works

6. Dropout does not break inference
   head.eval(); head(Tensor[N, E]) — no error, weights still sum to 1.

7. forward_with_attention consistency
   logits_a, w = head.forward_with_attention(x)
   logits_b     = head(x)
   torch.allclose(logits_a, logits_b)  # must be True (same computation)

8. Classifier hidden dims
   ABMILHead(E=768, K=3, classifier_hidden_dims=[256, 128])
   output shape still (1, 3).

Run: python -m multiplex_model.modules.abmil
-----------------------------------------------------------------------
"""

import torch
import torch.nn as nn

from .immuvis import TaskHead


# ===========================================================================
# ABMIL HEAD
# ===========================================================================

class ABMILHead(TaskHead):
    """Attention-Based MIL head (Ilse et al., ICML 2018).

    Aggregates N instance embeddings into one bag-level prediction using
    learned, instance-specific attention weights.

    Standard attention:
        a_k = softmax_k( w^T tanh(V h_k) )

    Gated attention (default — more expressive, avoids saturation artefacts):
        a_k = softmax_k( w^T ( tanh(V h_k) ⊙ σ(U h_k) ) )

    After aggregation:
        z   = Σ_k a_k · h_k          (weighted sum, shape E)
        out = MLP(z)                  (shape num_classes)

    Args:
        input_dim              : dimension E of per-crop embeddings from FinetuningModel.encode_crops
        num_classes            : number of output classes
        hidden_dim             : attention network hidden width (paper default: 128)
        classifier_hidden_dims : optional MLP layers between bag embedding and logits.
                                 [] = single linear layer (logistic regression on bag repr)
        gated                  : use gated attention (recommended; default True)
        dropout                : dropout applied to instance embeddings before attention
                                 (regularises attention on small bags; 0.0 = off)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        classifier_hidden_dims: list[int] | None = None,
        gated: bool = True,
        dropout: float = 0.0,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        self.gated = gated

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.classifer_dropout = nn.Dropout(p=classifier_dropout) if classifier_dropout > 0.0 else nn.Identity()

        # --- attention V branch: h_k → tanh(V h_k) -------------------------
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

        # --- attention U branch (gated only): h_k → σ(U h_k) ---------------
        if gated:
            self.attention_U = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),
            )

        # --- attention scoring: hidden_dim → scalar -------------------------
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

        # --- bag-level classifier -------------------------------------------
        classifier_hidden_dims = classifier_hidden_dims or []
        dims = [input_dim] + classifier_hidden_dims + [num_classes]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            # add nonlinearity/dropout only on hidden layers (not on final logits)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if classifier_dropout > 0.0:
                    layers.append(nn.Dropout(p=classifier_dropout))

        self.classifier = nn.Sequential(*layers)

    # -----------------------------------------------------------------------

    def attention_weights(self, instance_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute normalised attention weights for one bag.

        Args:
            instance_embeddings: (N, E)
        Returns:
            weights: (N, 1), non-negative, sum to 1 over the N dimension
        """
        h = self.dropout(instance_embeddings)   # (N, E)

        v = self.attention_V(h)                 # (N, hidden_dim)
        if self.gated:
            u = self.attention_U(h)             # (N, hidden_dim)
            a = self.attention_w(v * u)         # (N, 1)   element-wise gate
        else:
            a = self.attention_w(v)             # (N, 1)

        return torch.softmax(a, dim=0)          # (N, 1)  normalised over instances

    def forward(self, instance_embeddings: torch.Tensor) -> torch.Tensor:
        """Standard forward — returns logits only.

        Args:
            instance_embeddings: (N, E) per-crop embeddings for one image/bag
        Returns:
            logits: (1, num_classes) — raw class scores (no softmax)
        """
        weights = self.attention_weights(instance_embeddings)       # (N, 1)
        bag_embedding = (weights * instance_embeddings).sum(dim=0)  # (E,)
        return self.classifier(bag_embedding).unsqueeze(0)          # (1, num_classes)

    def forward_with_attention(
        self,
        instance_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns attention weights.

        Use this at inference time to visualise which crops drove the prediction.
        Identical computation to forward(); calling both on the same input gives
        identical logits.

        Args:
            instance_embeddings: (N, E)
        Returns:
            logits  : (1, num_classes)
            weights : (N, 1)  — attention weights per crop
        """
        weights = self.attention_weights(instance_embeddings)
        bag_embedding = (weights * instance_embeddings).sum(dim=0)
        logits = self.classifier(bag_embedding).unsqueeze(0)
        return logits, weights


# ===========================================================================
# SMOKE TEST  —  run:  python -m multiplex_model.modules.abmil
# ===========================================================================

def _smoke_test() -> None:
    import sys

    E = 768          # encoder latent dim from config (pm_embedding_dims[-1])
    K = 3            # num_classes
    H = 128          # attention hidden_dim

    passed = 0
    failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1

    print("\n=== ABMILHead smoke tests ===\n")

    # --- 1. Output shape (variable N) ---------------------------------------
    print("1. Output shape")
    head = ABMILHead(input_dim=E, num_classes=K, hidden_dim=H)
    for N in [1, 10, 64, 961]:
        x = torch.randn(N, E)
        out = head(x)
        check(f"  N={N}: output shape (1, {K})", out.shape == (1, K))

    # --- 2. Attention weights -----------------------------------------------
    print("\n2. Attention weights")
    x = torch.randn(20, E)
    w = head.attention_weights(x)
    check("shape (N, 1)",     w.shape == (20, 1))
    check("non-negative",     (w >= 0).all().item())
    check("sum ≈ 1.0",        abs(w.sum().item() - 1.0) < 1e-5)

    # --- 3. Gradient flow ---------------------------------------------------
    print("\n3. Gradient flow")
    head.zero_grad()
    x = torch.randn(15, E, requires_grad=True)
    out = head(x)
    out.sum().backward()
    check("attention_V grad not None", head.attention_V[0].weight.grad is not None)
    check("classifier grad not None",  head.classifier[0].weight.grad is not None)
    check("input grad not None",       x.grad is not None)

    # --- 4. Gated vs standard -----------------------------------------------
    print("\n4. Gated vs standard")
    head_gated    = ABMILHead(E, K, gated=True)
    head_standard = ABMILHead(E, K, gated=False)
    check("gated has attention_U",       hasattr(head_gated, "attention_U"))
    check("standard has no attention_U", not hasattr(head_standard, "attention_U"))
    x = torch.randn(10, E)
    check("gated forward works",    head_gated(x).shape    == (1, K))
    check("standard forward works", head_standard(x).shape == (1, K))

    # --- 5. forward_with_attention consistency ------------------------------
    print("\n5. forward_with_attention")
    head.eval()
    x = torch.randn(12, E)
    with torch.no_grad():
        logits_a, w = head.forward_with_attention(x)
        logits_b    = head(x)
    check("logits identical",      torch.allclose(logits_a, logits_b))
    check("weights shape (N, 1)",  w.shape == (12, 1))
    check("weights sum ≈ 1.0",     abs(w.sum().item() - 1.0) < 1e-5)

    # --- 6. Dropout in eval mode --------------------------------------------
    print("\n6. Dropout in eval mode")
    head_drop = ABMILHead(E, K, dropout=0.5)
    head_drop.eval()
    x = torch.randn(10, E)
    with torch.no_grad():
        w1 = head_drop.attention_weights(x)
        w2 = head_drop.attention_weights(x)
    check("eval is deterministic",  torch.allclose(w1, w2))
    check("weights sum ≈ 1.0",      abs(w1.sum().item() - 1.0) < 1e-5)

    # --- 7. Classifier hidden dims ------------------------------------------
    print("\n7. Classifier hidden dims")
    head_mlp = ABMILHead(E, K, classifier_hidden_dims=[256, 128])
    x = torch.randn(8, E)
    check("shape still (1, K)", head_mlp(x).shape == (1, K))

    # --- 8. Single-instance bag (edge case) ---------------------------------
    print("\n8. Single-instance bag")
    x = torch.randn(1, E)
    w = head.attention_weights(x)
    check("attention weight == 1.0 for N=1", abs(w.item() - 1.0) < 1e-6)

    # --- summary ------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed", "✓" if failed == 0 else "✗")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _smoke_test()
