"""
Library for batched inference using llama.cpp Python bindings.

This provides a clean API for generating multiple sequences in parallel.
"""


import ctypes
import multiprocessing
import time
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

import llama_cpp


class BatchedLLM:
    """
    A class for batched inference using llama.cpp Python bindings.

    This class provides methods for generating multiple sequences in parallel
    using the low-level API of llama.cpp.
    """

    def __init__(
        self,
        model_or_path: Union[str, ctypes.c_void_p],
        n_ctx: Optional[int] = None,
        n_threads: int = multiprocessing.cpu_count(),
        numa: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the BatchedLLM.

        :param model_or_path: Either a path to the model file or a loaded model pointer
        :type model_or_path: Union[str, ctypes.c_void_p]
        :param n_ctx: Context size (if None, will be determined automatically)
        :type n_ctx: Optional[int]
        :param n_threads: Number of threads to use
        :type n_threads: int
        :param numa: Whether to use NUMA optimization
        :type numa: bool
        :param verbose: Whether to print verbose output
        :type verbose: bool
        :raises RuntimeError: If the model cannot be loaded
        """
        self.n_threads = n_threads
        self.numa = numa
        self.verbose = verbose
        self.model_path = None

        # Initialize the llama backend if loading from path
        if isinstance(model_or_path, str):
            self.model_path = model_or_path
            llama_cpp.llama_backend_init()
            llama_cpp.llama_numa_init(numa)

            # Initialize the model
            model_params = llama_cpp.llama_model_default_params()

            self.model = llama_cpp.llama_model_load_from_file(model_or_path.encode("utf-8"), model_params)

            if self.model is None:
                raise RuntimeError(f"Unable to load model from {model_or_path}")
        else:
            # Use the provided model pointer
            self.model = model_or_path

        # Get the vocabulary
        self.vocab = llama_cpp.llama_model_get_vocab(self.model)

        # Initialize the context
        self.ctx_params = llama_cpp.llama_context_default_params()
        if n_ctx is not None:
            self.ctx_params.n_ctx = n_ctx
        self.ctx_params.n_threads = n_threads
        self.ctx_params.n_ctx_per_seq = n_ctx
        self.ctx_params.n_ctx = n_ctx

        self.ctx = llama_cpp.llama_init_from_model(self.model, self.ctx_params)

        if self.ctx is None:
            llama_cpp.llama_model_free(self.model)
            llama_cpp.llama_backend_free()
            raise RuntimeError("Failed to create the llama_context")

        # Get the actual context size
        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)
        print(f"{self.n_ctx=}")

        # Initialize other attributes
        self.batch = None
        self.smpl = None

    def __del__(self):
        """Clean up resources when the object is deleted.

        This method is automatically called when the object is garbage collected.
        """
        self.cleanup()

    def cleanup(self):
        """Clean up resources.

        Frees all allocated resources including batch, sampler, context, and model.
        """
        if hasattr(self, 'batch') and self.batch is not None:
            llama_cpp.llama_batch_free(self.batch)
            self.batch = None

        if hasattr(self, 'smpl') and self.smpl is not None:
            llama_cpp.llama_sampler_free(self.smpl)
            self.smpl = None

        if hasattr(self, 'ctx') and self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

        # Only free the model if we loaded it ourselves (i.e., from a path)
        if hasattr(self, 'model') and self.model is not None and self.model_path is not None:
            llama_cpp.llama_model_free(self.model)
            self.model = None

            # Only free the backend if we initialized it
            llama_cpp.llama_backend_free()

    def token_to_piece(self, token_id: int) -> str:
        """
        Convert a token ID to its string representation.

        :param token_id: The token ID to convert
        :type token_id: int
        :return: The string representation of the token
        :rtype: str
        """
        buf_size = 32
        buf = (ctypes.c_char * buf_size)()
        n = llama_cpp.llama_token_to_piece(
            self.vocab,
            llama_cpp.llama_token(token_id),
            buf,
            buf_size,
            0,  # offset
            True  # allow special tokens
        )
        assert n <= buf_size
        return buf[:n].decode("utf-8", errors="replace")

    def batch_add(self, batch, token_id: int, pos: int, seq_ids: List[int], compute_logits: bool = False):
        """
        Add a token to the batch.

        :param batch: The batch to add the token to
        :param token_id: The token ID to add
        :type token_id: int
        :param pos: The position of the token
        :type pos: int
        :param seq_ids: The sequence IDs to associate with the token
        :type seq_ids: List[int]
        :param compute_logits: Whether to compute logits for this token
        :type compute_logits: bool
        """
        i = batch.n_tokens
        batch.token[i] = token_id
        batch.pos[i] = pos
        batch.n_seq_id[i] = len(seq_ids)
        for j, seq_id in enumerate(seq_ids):
            batch.seq_id[i][j] = seq_id
        batch.logits[i] = compute_logits
        batch.n_tokens += 1

    def batch_clear(self, batch):
        """
        Clear the batch.

        :param batch: The batch to clear
        """
        batch.n_tokens = 0

    def logits_to_logprobs(self, logits, axis=-1):
        """
        Convert logits to log probabilities.

        :param logits: The logits to convert
        :type logits: numpy.ndarray
        :param axis: The axis along which to compute softmax
        :type axis: int
        :return: Log probabilities
        :rtype: numpy.ndarray
        """
        # Implementation based on llama_cpp.Llama.logits_to_logprobs
        logits_maxs = np.amax(logits, axis=axis, keepdims=True)
        if logits_maxs.ndim > 0:
            logits_maxs[~np.isfinite(logits_maxs)] = 0
        elif not np.isfinite(logits_maxs):
            logits_maxs = 0
        subtract_maxs = np.subtract(logits, logits_maxs, dtype=np.float32)
        exp = np.exp(subtract_maxs)
        # Suppress warnings about log of zero
        with np.errstate(divide="ignore"):
            summed = np.sum(exp, axis=axis, keepdims=True)
            out = np.log(summed)
        return subtract_maxs - out

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize a text string.

        :param text: The text to tokenize
        :type text: str
        :return: A list of token IDs
        :rtype: List[int]
        :raises RuntimeError: If tokenization fails
        """
        text_bytes = text.encode("utf-8")
        tokens_buf_size = len(text_bytes) * 2  # Allocate more space than needed
        tokens_buf = (llama_cpp.llama_token * tokens_buf_size)()

        n_tokens = llama_cpp.llama_tokenize(
            self.vocab,
            text_bytes,
            len(text_bytes),
            tokens_buf,
            tokens_buf_size,
            True,  # Add BOS token
            True,  # Special tokens are allowed
        )

        if n_tokens <= 0:
            raise RuntimeError("Failed to tokenize the text")

        # Convert to a Python list
        return [tokens_buf[i] for i in range(n_tokens)]

    def setup_sampler(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        temp: float = 0.8,
        seed: int = 42,
        min_keep: int = 1
    ):
        """
        Set up a sampler chain with various sampling methods.

        :param top_k: Top-k sampling parameter
        :type top_k: int
        :param top_p: Top-p sampling parameter
        :type top_p: float
        :param temp: Temperature parameter
        :type temp: float
        :param seed: Random seed
        :type seed: int
        :param min_keep: Minimum number of tokens to keep
        :type min_keep: int
        :return: The sampler chain
        :rtype: ctypes.c_void_p
        """
        sparams = llama_cpp.llama_sampler_chain_default_params()
        sparams.no_perf = not self.verbose

        self.smpl = llama_cpp.llama_sampler_chain_init(sparams)

        # Add samplers to the chain
        llama_cpp.llama_sampler_chain_add(self.smpl, llama_cpp.llama_sampler_init_top_k(top_k))
        llama_cpp.llama_sampler_chain_add(self.smpl, llama_cpp.llama_sampler_init_top_p(top_p, min_keep))
        llama_cpp.llama_sampler_chain_add(self.smpl, llama_cpp.llama_sampler_init_temp(temp))
        llama_cpp.llama_sampler_chain_add(self.smpl, llama_cpp.llama_sampler_init_dist(seed))

        return self.smpl

    def generate(
        self,
        prompt: str,
        n_predict: int = 32,
        n_parallel: int = 4,
        top_k: int = 40,
        top_p: float = 0.95,
        temp: float = 0.8,
        seed: int = 42,
        min_keep: int = 1
    ) -> Tuple[List[str], Dict[str, Any], Optional[List[List[float]]]]:
        """
        Generate multiple sequences in parallel.

        :param prompt: The prompt to start generation with
        :type prompt: str
        :param n_predict: Number of tokens to predict
        :type n_predict: int
        :param n_parallel: Number of parallel sequences to generate
        :type n_parallel: int
        :param top_k: Top-k sampling parameter
        :type top_k: int
        :param top_p: Top-p sampling parameter
        :type top_p: float
        :param temp: Temperature parameter
        :type temp: float
        :param seed: Random seed
        :type seed: int
        :param min_keep: Minimum number of tokens to keep
        :type min_keep: int
        :return: A tuple containing (generated_sequences, statistics, token_logprobs)
        :rtype: Tuple[List[str], Dict[str, Any], Optional[List[List[float]]]]
        :raises RuntimeError: If generation fails
        """
        # Tokenize the prompt
        tokens_list = self.tokenize(prompt)

        # Calculate required KV cache size
        n_kv_req = len(tokens_list) + (n_predict - len(tokens_list)) * n_parallel

        # Update context parameters if needed
        if n_kv_req > self.n_ctx:
            raise RuntimeError(
                f"Required KV cache size ({n_kv_req}) exceeds context size ({self.n_ctx}). "
                "Either reduce n_parallel or increase n_ctx."
            )

        # Set up the sampler chain
        self.setup_sampler(top_k, top_p, temp, seed, min_keep)

        # Create a batch
        batch_size = max(len(tokens_list), n_parallel)
        self.batch = llama_cpp.llama_batch_init(batch_size, 0, n_parallel)

        # Evaluate the initial prompt
        for i, token_id in enumerate(tokens_list):
            self.batch_add(self.batch, token_id, i, [0], False)

        # Check if the model has an encoder
        if llama_cpp.llama_model_has_encoder(self.model):
            if llama_cpp.llama_encode(self.ctx, self.batch) != 0:
                raise RuntimeError("Failed to encode")

            decoder_start_token_id = llama_cpp.llama_model_decoder_start_token(self.model)
            if decoder_start_token_id == llama_cpp.LLAMA_TOKEN_NULL:
                decoder_start_token_id = llama_cpp.llama_vocab_bos(self.vocab)

            self.batch_clear(self.batch)
            self.batch_add(self.batch, decoder_start_token_id, 0, [0], False)

        # Compute logits for the last token of the prompt
        self.batch.logits[self.batch.n_tokens - 1] = True

        if llama_cpp.llama_decode(self.ctx, self.batch) != 0:
            raise RuntimeError("llama_decode() failed")

        # Assign the system KV cache to all parallel sequences
        # This way, the parallel sequences will "reuse" the prompt tokens without having to copy them
        for i in range(1, n_parallel):
            llama_cpp.llama_kv_cache_seq_cp(self.ctx, 0, i, -1, -1)

        # Main loop

        # Store the generated sequences
        streams = [""] * n_parallel

        # Store logprobs for each token in each sequence
        token_logprobs = [[] for _ in range(n_parallel)]

        # Remember the batch index of the last token for each parallel sequence
        i_batch = [self.batch.n_tokens - 1] * n_parallel

        n_cur = self.batch.n_tokens
        n_decode = 0

        t_main_start = time.time()

        while n_cur <= n_predict:
            # Prepare the next batch
            self.batch_clear(self.batch)

            # Sample the next token for each parallel sequence
            for i in range(n_parallel):
                if i_batch[i] < 0:
                    # The stream has already finished
                    continue
                print(f"Sampling token {n_cur} for sequence {i}")

                # Sample the next token
                new_token_id = llama_cpp.llama_sampler_sample(self.smpl, self.ctx, i_batch[i])
                # print(f"{new_token_id=} ")
                # Get the logits for the sampled token
                logits = llama_cpp.llama_get_logits_ith(self.ctx, i_batch[i])
                # print(f"{logits=} {type(logits)=}")
                n_vocab = llama_cpp.llama_vocab_n_tokens(self.vocab)
                # print(f"{n_vocab=}")

                logits_array = np.array(
                    [logits[j] for j in range(n_vocab)],
                    dtype=np.float32
                )
                # print(f"{logits_array=} {type(logits_array)=}")

                # Convert logits to logprobs
                logprobs_array = self.logits_to_logprobs(logits_array)
                # print(f"{logprobs_array=} {type(logprobs_array)=}")

                # Store the logprob of the sampled token
                token_logprobs[i].append(logprobs_array[new_token_id])

                # Check if it's an end of generation
                if llama_cpp.llama_vocab_is_eog(self.vocab, new_token_id) or n_cur == n_predict:
                    i_batch[i] = -1
                    continue

                # Add the token to the stream
                streams[i] += self.token_to_piece(new_token_id)

                # Update the batch index
                i_batch[i] = self.batch.n_tokens

                # Add the token to the batch
                self.batch_add(self.batch, new_token_id, n_cur, [i], True)

                n_decode += 1

            # All streams are finished
            if self.batch.n_tokens == 0:
                break

            n_cur += 1

            # Evaluate the current batch
            if llama_cpp.llama_decode(self.ctx, self.batch) != 0:
                raise RuntimeError("Failed to decode")

        t_main_end = time.time()

        # Prepare the results
        results = []
        for i in range(n_parallel):
            results.append(prompt + streams[i])

        # Prepare statistics
        stats = {
            "n_decode": n_decode,
            "time_s": t_main_end - t_main_start,
            "tokens_per_second": n_decode / (t_main_end - t_main_start) if t_main_end > t_main_start else 0,
        }

        # Clean up batch and sampler
        llama_cpp.llama_batch_free(self.batch)
        self.batch = None

        llama_cpp.llama_sampler_free(self.smpl)
        self.smpl = None

        return results, stats, token_logprobs
