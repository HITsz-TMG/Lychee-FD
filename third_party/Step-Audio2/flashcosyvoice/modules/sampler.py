import torch
from torch import nn


class Sampler(nn.Module):
    """
    Optimized sampler implementation using vectorized operations instead of loops, significantly improving performance

    Performance optimizations:
    1. Using batch processing instead of sequence loops, reducing Python loop overhead
    2. Using PyTorch's vectorized operations (like torch.sort, torch.gather) for parallel computation
    3. Using mask operations to apply top-k filtering at once, avoiding per-sequence processing
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_k: int = None):
        """
        Perform sampling operation using vectorized method for top-k filtering

        Args:
            logits: Logits tensor with shape [batch_size, vocab_size]
            temperatures: Temperature parameters with shape [batch_size]
            top_k: Top-k value for filtering (uniform across all sequences)

        Returns:
            Sampled token IDs
        """
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)  # Greedy decoding result, used when temperature=0
        logits.div_(temperatures.unsqueeze(dim=1))  # Apply temperature scaling

        # Apply uniform top-k filtering if top_k is provided
        if top_k is not None and top_k > 0:
            vocab_size = logits.size(-1)

            # Create a mask to store which positions should be kept
            mask = torch.zeros_like(logits, dtype=torch.bool)

            # Batch sorting for all sequences at once
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

            # Get threshold for each sequence (the k-th largest value)
            k_value = min(top_k, vocab_size)  # Ensure k doesn't exceed vocab size
            thresholds = sorted_logits[:, k_value-1:k_value]  # Shape [batch_size, 1]
            thresholds = thresholds.expand(-1, vocab_size)    # Expand to match logits shape

            # Create mask: only keep logits greater than or equal to threshold
            mask = logits >= thresholds

            # Apply mask: set logits not in top-k to negative infinity
            logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=logits.device))

        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


class RasSampler(nn.Module):
    """
    Optimized Repetition Aware Sampling implementation

    Performance optimizations:
    1. Using vectorized nucleus sampling instead of loop implementation, improving sampling efficiency
    2. Using tensor operations to calculate repetition rate, reducing Python loop overhead
    3. Optimizing EOS handling logic, reducing unnecessary resampling
    4. Using PyTorch's vectorized operations for parallel computation
    5. Batch processing for all sequences, dramatically improving throughput
    6. Robust handling for sequences of any length, including empty sequences
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, decoded_tokens_list: list,
                win_size: int = 10, tau_r: float = 0.1,
                top_p: float = 0.8, top_k: int = 25,
                eos_token: int = 6561, min_tokens: list[int] = None):
        """
        Execute repetition-aware sampling using optimized vectorized operations with batch processing

        Args:
            logits: Input logits with shape [batch_size, vocab_size]
            decoded_tokens_list: List of decoded tokens, each element is a token list for a batch
            win_size: Window size for repetition detection (uniform across all batch items)
            tau_r: Repetition threshold (uniform across all batch items)
            top_p: Nucleus sampling probability threshold (uniform across all batch items)
            top_k: Nucleus sampling top-k threshold (uniform across all batch items)
            eos_token: End of sequence token ID (uniform across all batch items)
            min_tokens: List of minimum tokens to generate before allowing EOS, one per batch item
        Returns:
            Selected token IDs
        """
        batch_size = logits.size(0)
        device = logits.device
        result = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Set default values if not provided
        if min_tokens is None:
            min_tokens = [2] * batch_size

        # Ensure min_tokens list has the correct length
        assert len(min_tokens) == batch_size, f"min_tokens length {len(min_tokens)} != batch_size {batch_size}"

        # Force continue decode first token
        for i in range(batch_size):
            if i < len(decoded_tokens_list) and len(decoded_tokens_list[i]) == 0:
                logits[i, eos_token] = -float('inf')

        # 1. First, perform nucleus sampling for all sequences
        probs = torch.softmax(logits, dim=-1)

        # Use vectorized nucleus sampling for all sequences
        # This can be done in batch since top_p and top_k are uniform
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create masks for top-p and top-k filtering
        top_p_mask = cumulative_probs <= top_p

        # Create top-k mask (first top_k positions are True)
        top_k_mask = torch.zeros_like(top_p_mask)
        top_k_mask[:, :top_k] = True

        # Combine masks
        mask = top_p_mask & top_k_mask

        # Ensure at least one token is selected per sequence
        first_token_mask = torch.zeros_like(mask)
        first_token_mask[:, 0] = True
        mask = mask | first_token_mask

        # Sample from the filtered distribution
        sample_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
        sample_probs = sample_probs / sample_probs.sum(dim=-1, keepdim=True)

        # Sample indices from the filtered distribution
        sampled_indices = torch.multinomial(sample_probs, 1).squeeze(-1)
        top_ids = torch.gather(sorted_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)

        # 2. Check for repetitions and apply random sampling if needed
        # Extract recent tokens for each sequence, handling empty or short sequences
        recent_tokens_list = []
        for i in range(batch_size):
            # Handle index out of range or empty tokens
            if i < len(decoded_tokens_list):
                tokens = decoded_tokens_list[i]
                if len(tokens) > 0:
                    start_idx = max(0, len(tokens) - win_size)
                    recent_tokens_list.append(tokens[start_idx:])
                else:
                    recent_tokens_list.append([])  # Empty list for empty tokens
            else:
                recent_tokens_list.append([])  # Empty list for missing batch items

        # Check if we have any tokens to process for repetition detection
        if any(len(tokens) > 0 for tokens in recent_tokens_list):
            # Convert to padded tensor for batch processing
            max_recent_len = max(len(tokens) for tokens in recent_tokens_list)
            if max_recent_len > 0:  # Only proceed if we have tokens
                recent_tokens_tensor = torch.zeros((batch_size, max_recent_len), dtype=torch.long, device=device) - 1
                for i, tokens in enumerate(recent_tokens_list):
                    if len(tokens) > 0:
                        recent_tokens_tensor[i, -len(tokens):] = torch.tensor(tokens, device=device)

                # Create a mask for valid positions and to avoid division by zero
                valid_positions_mask = torch.zeros_like(recent_tokens_tensor, dtype=torch.bool)
                for i, tokens in enumerate(recent_tokens_list):
                    if len(tokens) > 0:
                        valid_positions_mask[i, -len(tokens):] = True

                # Check repetition rates
                repetition_counts = torch.zeros(batch_size, device=device)
                for i in range(batch_size):
                    if len(recent_tokens_list[i]) > 0:
                        repetition_counts[i] = (recent_tokens_tensor[i] == top_ids[i]).sum()

                # Calculate repetition rates, avoiding division by zero
                recent_lengths = torch.tensor([max(1, len(tokens)) for tokens in recent_tokens_list], device=device)
                repetition_rates = repetition_counts / recent_lengths

                # Identify sequences needing random sampling
                need_random = repetition_rates >= tau_r

                # Apply random sampling where needed
                if need_random.any():
                    random_indices = torch.multinomial(probs[need_random], 1).squeeze(-1)
                    top_ids[need_random] = random_indices

        # 3. Handle EOS tokens
        # Create mask for sequences that should ignore EOS tokens
        ignore_eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if i < len(decoded_tokens_list):
                ignore_eos_mask[i] = len(decoded_tokens_list[i]) < min_tokens[i]
            else:
                ignore_eos_mask[i] = True  # Default to ignoring EOS for missing sequences

        is_eos_mask = top_ids == eos_token
        need_resample = ignore_eos_mask & is_eos_mask

        # Resample for sequences that need it
        if need_resample.any():
            max_trials = 100
            for attempt in range(max_trials):
                # Break if no more resampling needed
                if not need_resample.any():
                    break

                # Sample new tokens for sequences that need resampling
                new_samples = torch.multinomial(probs[need_resample], 1).squeeze(-1)

                # Update top_ids with new samples
                top_ids[need_resample] = new_samples

                # Update which sequences still need resampling
                is_eos_mask = top_ids == eos_token
                need_resample = ignore_eos_mask & is_eos_mask

            # If still have EOS tokens that should be ignored, force them to be non-EOS
            if need_resample.any():
                # Force to a non-EOS token (e.g., the second most likely token)
                for i in range(batch_size):
                    if need_resample[i]:
                        # Get second most likely token (or first if only one token)
                        second_best_idx = 1 if sorted_indices.size(1) > 1 else 0
                        top_ids[i] = sorted_indices[i, second_best_idx]

        result = top_ids

        return result
