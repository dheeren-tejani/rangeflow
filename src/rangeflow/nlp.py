"""
RangeFlow NLP Module
====================
Natural Language Processing with uncertainty quantification.
Handles text perturbations like synonyms, typos, paraphrasing.
"""

from .core import RangeTensor, _op
from .layers import RangeModule, RangeLinear, RangeLayerNorm, RangeAttention
from .backend import get_backend
import numpy as np

xp = get_backend()


class RangeEmbedding(RangeModule):
    """
    Word embeddings with uncertainty.
    
    Can handle:
    - Synonym substitutions
    - Word uncertainty
    - Embedding noise
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        padding_idx: Index for padding token
    
    Example:
        >>> embed = RangeEmbedding(vocab_size=10000, embed_dim=768)
        >>> tokens = torch.LongTensor([[1, 2, 3, 4]])
        >>> # Option 1: Standard embeddings
        >>> emb = embed(tokens)
        >>> # Option 2: With uncertainty
        >>> emb_range = embed(tokens, uncertainty=0.1)
    """
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        
        # Initialize embedding matrix
        limit = np.sqrt(1.0 / embed_dim)
        self.weight = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (vocab_size, embed_dim))
        )
        
        # Set padding embedding to zero
        if padding_idx is not None:
            weight_val = self.weight.symbol.value
            if isinstance(weight_val, tuple):
                weight_val[0][padding_idx] = 0
                weight_val[1][padding_idx] = 0
            else:
                weight_val[padding_idx] = 0
    
    def forward(self, tokens, uncertainty=0.0):
        """
        Forward pass with optional uncertainty.
        
        Args:
            tokens: Token indices (batch_size, seq_len)
            uncertainty: Embedding noise level (0 = deterministic)
        
        Returns:
            RangeTensor of embeddings (batch_size, seq_len, embed_dim)
        """
        # Get embeddings
        weight_min, weight_max = self.weight.decay()
        
        # Index embeddings
        if hasattr(tokens, 'cpu'):  # PyTorch tensor
            indices = tokens.cpu().numpy()
        else:
            indices = tokens
        
        embeddings = weight_min[indices]  # Center value
        
        if uncertainty > 0:
            # Add uncertainty
            return RangeTensor.from_epsilon_ball(embeddings, uncertainty)
        else:
            return RangeTensor.from_array(embeddings)


class RangePositionalEncoding(RangeModule):
    """
    Positional encoding with optional uncertainty.
    
    Useful for modeling position ambiguity in sequences.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = xp.arange(max_len).reshape(-1, 1)
        div_term = xp.exp(xp.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = xp.zeros((max_len, d_model))
        pe[:, 0::2] = xp.sin(position * div_term)
        pe[:, 1::2] = xp.cos(position * div_term)
        
        self.pe = RangeTensor.from_array(pe)
    
    def forward(self, x, uncertainty=0.0):
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            uncertainty: Position uncertainty
        
        Returns:
            Embeddings with positional encoding
        """
        seq_len = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
        
        # Get positional encodings
        pos_enc_min, pos_enc_max = self.pe.decay()
        pos_enc = pos_enc_min[:seq_len]
        
        if uncertainty > 0:
            pos_enc_range = RangeTensor.from_epsilon_ball(pos_enc, uncertainty)
            return x + pos_enc_range
        else:
            return x + RangeTensor.from_array(pos_enc)


class RangeTransformerBlock(RangeModule):
    """
    Complete Transformer block with range propagation.
    
    Includes:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    
    Example:
        >>> block = RangeTransformerBlock(d_model=768, num_heads=12, d_ff=3072)
        >>> x_range = RangeTensor.from_epsilon_ball(x, 0.1)
        >>> output_range = block(x_range)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention
        self.attention = RangeAttention(d_model, num_heads)
        self.norm1 = RangeLayerNorm(d_model)
        
        # Feed-forward network
        self.ff1 = RangeLinear(d_model, d_ff)
        self.ff2 = RangeLinear(d_ff, d_model)
        self.norm2 = RangeLayerNorm(d_model)
        
        from .layers import RangeDropout
        self.dropout1 = RangeDropout(dropout)
        self.dropout2 = RangeDropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input (batch, seq_len, d_model) as RangeTensor
        
        Returns:
            Output (batch, seq_len, d_model) as RangeTensor
        """
        # Self-attention with residual
        attn_out = self.attention(x)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff1(x).relu()
        ff_out = self.dropout2(self.ff2(ff_out))
        x = self.norm2(x + ff_out)
        
        return x


# ==========================================
# TEXT PERTURBATION UTILITIES
# ==========================================

def word_substitution(text, synonyms_dict, prob=0.1):
    """
    Create text range by modeling synonym substitutions.
    
    Args:
        text: Input text or token IDs
        synonyms_dict: Dictionary mapping words to synonyms
        prob: Probability of substitution per word
    
    Returns:
        RangeTensor representing all possible substitutions
    
    Example:
        >>> synonyms = {'good': ['great', 'excellent'], 'bad': ['poor', 'terrible']}
        >>> text = "This movie is good"
        >>> text_range = word_substitution(text, synonyms, prob=0.2)
    """
    # For token IDs, create ranges
    if isinstance(text, (list, np.ndarray)):
        tokens = xp.array(text)
        
        # For now, add small uncertainty to simulate synonym variation
        # Full implementation would track actual synonym mappings
        return RangeTensor.from_epsilon_ball(tokens, prob * 0.5)
    
    # For text strings, this would require tokenization
    raise NotImplementedError("String input requires tokenizer")


def typo_perturbation(tokens, typo_rate=0.05):
    """
    Model typing errors as token uncertainty.
    
    Args:
        tokens: Token IDs (batch, seq_len)
        typo_rate: Rate of typos (0-1)
    
    Returns:
        RangeTensor with typo uncertainty
    
    Example:
        >>> tokens = torch.LongTensor([[1, 2, 3, 4, 5]])
        >>> tokens_range = typo_perturbation(tokens, typo_rate=0.1)
    """
    if hasattr(tokens, 'cpu'):
        tokens = tokens.cpu().numpy()
    
    # Model as small perturbation in embedding space
    # (after embedding layer)
    return RangeTensor.from_epsilon_ball(tokens, typo_rate * 0.3)


def paraphrase_uncertainty(embeddings, paraphrase_strength=0.1):
    """
    Model paraphrasing as embedding perturbations.
    
    Args:
        embeddings: Token embeddings (batch, seq_len, dim)
        paraphrase_strength: How different paraphrases can be
    
    Returns:
        RangeTensor with paraphrase uncertainty
    """
    return RangeTensor.from_epsilon_ball(embeddings, paraphrase_strength)


def adversarial_text_perturbation(tokens, epsilon=1.0):
    """
    Model adversarial text attacks (character/word level).
    
    Args:
        tokens: Token IDs
        epsilon: Attack budget
    
    Returns:
        RangeTensor representing adversarial perturbations
    """
    return RangeTensor.from_epsilon_ball(tokens, epsilon)


# ==========================================
# VERIFICATION UTILITIES
# ==========================================

def verify_sentiment_robustness(model, text_embeddings, label, 
                                substitutions=None, epsilon=0.1):
    """
    Verify if sentiment prediction is stable under text perturbations.
    
    Args:
        model: Sentiment classifier
        text_embeddings: Input embeddings
        label: Ground truth label
        substitutions: Possible word substitutions
        epsilon: Perturbation budget
    
    Returns:
        (is_robust, margin): Certification result
    
    Example:
        >>> embeddings = embedding_layer(tokens)
        >>> is_robust, margin = verify_sentiment_robustness(
        ...     model, embeddings, label=1, epsilon=0.2
        ... )
        >>> if is_robust:
        ...     print(f"Certified robust with margin {margin:.3f}")
    """
    # Create range around embeddings
    emb_range = RangeTensor.from_epsilon_ball(text_embeddings, epsilon)
    
    # Forward pass
    logits_range = model(emb_range)
    min_logits, max_logits = logits_range.decay()
    
    # Check if prediction is stable
    target = label
    correct_min = min_logits[0, target]
    
    # Get max of other classes
    mask = xp.ones(min_logits.shape[1], dtype=bool)
    mask[target] = False
    others_max = xp.max(max_logits[0, mask])
    
    margin = float(correct_min - others_max)
    is_robust = margin > 0
    
    return is_robust, margin


def verify_nli_robustness(model, premise_emb, hypothesis_emb, label, epsilon=0.1):
    """
    Verify Natural Language Inference robustness.
    
    Args:
        model: NLI model
        premise_emb: Premise embeddings
        hypothesis_emb: Hypothesis embeddings
        label: Ground truth (entailment/contradiction/neutral)
        epsilon: Perturbation budget
    
    Returns:
        (is_robust, margin): Certification result
    """
    # Create ranges
    premise_range = RangeTensor.from_epsilon_ball(premise_emb, epsilon)
    hypothesis_range = RangeTensor.from_epsilon_ball(hypothesis_emb, epsilon)
    
    # Forward (model should accept tuple of inputs)
    logits_range = model(premise_range, hypothesis_range)
    min_logits, max_logits = logits_range.decay()
    
    # Check certification
    correct_min = min_logits[0, label]
    mask = xp.ones(min_logits.shape[1], dtype=bool)
    mask[label] = False
    others_max = xp.max(max_logits[0, mask])
    
    margin = float(correct_min - others_max)
    return margin > 0, margin


def certified_bleu_bounds(model, source_range, reference):
    """
    Compute bounds on BLEU score under input uncertainty.
    
    Args:
        model: Translation model
        source_range: Source text with uncertainty
        reference: Reference translation
    
    Returns:
        (min_bleu, max_bleu): BLEU score bounds
    """
    # Forward pass
    translation_range = model(source_range)
    
    # Get min/max translations (would need beam search with ranges)
    # For now, return placeholder
    return 0.0, 1.0  # TODO: Implement proper BLEU bounds


# ==========================================
# UTILITIES
# ==========================================

def create_attention_mask_range(mask, uncertainty=0.0):
    """
    Create attention mask with uncertainty.
    
    Useful for modeling uncertain token importance.
    """
    return RangeTensor.from_epsilon_ball(mask, uncertainty)


def merge_token_ranges(ranges_list):
    """
    Merge multiple token-level ranges into sequence range.
    
    Args:
        ranges_list: List of RangeTensors for each token
    
    Returns:
        Combined sequence range
    """
    # Stack along sequence dimension
    mins = [r.decay()[0] for r in ranges_list]
    maxs = [r.decay()[1] for r in ranges_list]
    
    combined_min = xp.stack(mins, axis=1)
    combined_max = xp.stack(maxs, axis=1)
    
    return RangeTensor.from_range(combined_min, combined_max)