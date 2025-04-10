"""
Unit tests for the BatchedLLM class.

This module contains tests for the BatchedLLM class, which provides batched inference
using llama.cpp Python bindings. The tests use mocking to avoid requiring actual model files.
"""

import pytest
import ctypes
import time
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from typing import List, Dict, Any, Tuple, Optional

# Import the class to test
from services.batched_llm import BatchedLLM


@pytest.fixture
def mock_llama_cpp():
    """Create a comprehensive mock for the llama_cpp module."""
    with patch('services.batched_llm.llama_cpp') as mock:
        # Mock model loading
        mock.llama_backend_init = MagicMock()
        mock.llama_numa_init = MagicMock()
        mock.llama_model_default_params = MagicMock(return_value=MagicMock())
        mock.llama_model_load_from_file = MagicMock(return_value=MagicMock())
        
        # Mock context creation
        mock.llama_context_default_params = MagicMock(return_value=MagicMock())
        mock.llama_init_from_model = MagicMock(return_value=MagicMock())
        mock.llama_n_ctx = MagicMock(return_value=2048)
        
        # Mock vocabulary functions
        mock.llama_model_get_vocab = MagicMock(return_value=MagicMock())
        mock.llama_token_to_piece = MagicMock(return_value=5)  # Return a length for the token piece
        mock.llama_vocab_n_tokens = MagicMock(return_value=32000)
        mock.llama_vocab_is_eog = MagicMock(return_value=False)
        mock.llama_vocab_bos = MagicMock(return_value=1)
        
        # Mock tokenization
        mock.llama_token = MagicMock(side_effect=lambda x: x)  # Identity function
        mock.llama_tokenize = MagicMock(return_value=5)  # Return a token count
        
        # Mock batch operations
        mock.llama_batch_init = MagicMock()
        mock.llama_batch_free = MagicMock()
        
        # Mock sampling
        mock.llama_sampler_chain_default_params = MagicMock(return_value=MagicMock())
        mock.llama_sampler_chain_init = MagicMock(return_value=MagicMock())
        mock.llama_sampler_chain_add = MagicMock()
        mock.llama_sampler_init_top_k = MagicMock(return_value=MagicMock())
        mock.llama_sampler_init_top_p = MagicMock(return_value=MagicMock())
        mock.llama_sampler_init_temp = MagicMock(return_value=MagicMock())
        mock.llama_sampler_init_dist = MagicMock(return_value=MagicMock())
        mock.llama_sampler_sample = MagicMock(return_value=42)  # Return a token ID
        mock.llama_sampler_free = MagicMock()
        
        # Mock decoding
        mock.llama_decode = MagicMock(return_value=0)  # Success
        mock.llama_get_logits_ith = MagicMock(return_value=[0.1] * 32000)  # Mock logits
        
        # Mock KV cache operations
        mock.llama_kv_cache_seq_cp = MagicMock()
        
        # Mock encoder-decoder handling
        mock.llama_model_has_encoder = MagicMock(return_value=False)
        mock.llama_encode = MagicMock(return_value=0)  # Success
        mock.llama_model_decoder_start_token = MagicMock(return_value=2)
        mock.LLAMA_TOKEN_NULL = -1
        
        # Mock cleanup
        mock.llama_free = MagicMock()
        mock.llama_model_free = MagicMock()
        mock.llama_backend_free = MagicMock()
        
        yield mock


@pytest.fixture
def mock_batch():
    """Create a mock batch object with the necessary attributes."""
    batch = MagicMock()
    batch.n_tokens = 0
    batch.token = [0] * 1024
    batch.pos = [0] * 1024
    batch.n_seq_id = [0] * 1024
    batch.seq_id = [[0] * 4 for _ in range(1024)]
    batch.logits = [False] * 1024
    return batch


@pytest.fixture
def batched_llm(mock_llama_cpp):
    """Create a BatchedLLM instance with mocked dependencies."""
    # Mock the tokenize method to return a list of tokens
    with patch.object(BatchedLLM, 'tokenize', return_value=[1, 2, 3, 4, 5]):
        # Mock the token_to_piece method to return a string
        with patch.object(BatchedLLM, 'token_to_piece', return_value="token"):
            llm = BatchedLLM("mock_model_path.gguf")
            # Set the batch attribute directly
            llm.batch = mock_batch()
            yield llm


class TestBatchedLLM:
    """Test suite for the BatchedLLM class."""

    def test_initialization(self, mock_llama_cpp):
        """Test that the BatchedLLM initializes correctly."""
        # Test initialization from a path
        llm = BatchedLLM("mock_model_path.gguf")
        
        # Verify llama.cpp initialization calls
        mock_llama_cpp.llama_backend_init.assert_called_once()
        mock_llama_cpp.llama_numa_init.assert_called_once()
        mock_llama_cpp.llama_model_load_from_file.assert_called_once()
        mock_llama_cpp.llama_context_default_params.assert_called_once()
        mock_llama_cpp.llama_init_from_model.assert_called_once()
        
        # Test initialization from a model pointer
        model_ptr = ctypes.c_void_p()
        llm = BatchedLLM(model_ptr)
        
        # Verify no additional backend initialization
        assert mock_llama_cpp.llama_backend_init.call_count == 1
        
        # Verify context creation
        assert mock_llama_cpp.llama_init_from_model.call_count == 2

    def test_cleanup(self, mock_llama_cpp):
        """Test that resources are properly cleaned up."""
        llm = BatchedLLM("mock_model_path.gguf")
        
        # Set batch and sampler attributes
        llm.batch = MagicMock()
        llm.smpl = MagicMock()
        
        # Call cleanup
        llm.cleanup()
        
        # Verify cleanup calls
        mock_llama_cpp.llama_batch_free.assert_called_once()
        mock_llama_cpp.llama_sampler_free.assert_called_once()
        mock_llama_cpp.llama_free.assert_called_once()
        mock_llama_cpp.llama_model_free.assert_called_once()
        mock_llama_cpp.llama_backend_free.assert_called_once()
        
        # Verify attributes are reset
        assert llm.batch is None
        assert llm.smpl is None
        assert llm.ctx is None
        assert llm.model is None

    def test_token_to_piece(self, mock_llama_cpp, batched_llm):
        """Test token to piece conversion."""
        result = batched_llm.token_to_piece(42)
        
        # Verify the function was called with correct parameters
        mock_llama_cpp.llama_token_to_piece.assert_called_once()
        args = mock_llama_cpp.llama_token_to_piece.call_args[0]
        assert args[0] == batched_llm.vocab
        assert args[1] == 42
        
        # Since we're mocking the C function, we can't easily verify the buffer contents
        # But we can check that the result is a string
        assert isinstance(result, str)

    def test_batch_operations(self, batched_llm, mock_batch):
        """Test batch add and clear operations."""
        # Test batch_add
        batched_llm.batch_add(mock_batch, 42, 0, [0, 1], True)
        
        # Verify batch state after add
        assert mock_batch.token[0] == 42
        assert mock_batch.pos[0] == 0
        assert mock_batch.n_seq_id[0] == 2
        assert mock_batch.seq_id[0][0] == 0
        assert mock_batch.seq_id[0][1] == 1
        assert mock_batch.logits[0] is True
        assert mock_batch.n_tokens == 1
        
        # Test batch_clear
        batched_llm.batch_clear(mock_batch)
        
        # Verify batch state after clear
        assert mock_batch.n_tokens == 0

    def test_logits_to_logprobs(self, batched_llm):
        """Test conversion from logits to log probabilities."""
        # Create test logits
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Convert to log probabilities
        logprobs = batched_llm.logits_to_logprobs(logits)
        
        # Verify the result is a numpy array
        assert isinstance(logprobs, np.ndarray)
        
        # Verify the shape matches the input
        assert logprobs.shape == logits.shape
        
        # Verify the values are log probabilities (sum of exp should be 1)
        # We can check that the max value is <= 0 (log probabilities are always <= 0)
        assert np.max(logprobs) <= 0
        
        # Test with non-finite values
        logits_with_inf = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        logprobs_with_inf = batched_llm.logits_to_logprobs(logits_with_inf)
        
        # Verify the result is still a valid array without NaNs
        assert not np.isnan(logprobs_with_inf).any()

    def test_tokenize(self, mock_llama_cpp):
        """Test text tokenization."""
        # Create a new instance to avoid the patched tokenize method
        llm = BatchedLLM("mock_model_path.gguf")
        
        # Mock the llama_tokenize function to populate the token buffer
        def mock_tokenize_impl(*args, **kwargs):
            # Extract the token buffer from the arguments
            tokens_buf = args[3]
            # Fill the buffer with some token IDs
            for i in range(5):
                tokens_buf[i] = i + 1
            return 5  # Return the number of tokens
        
        mock_llama_cpp.llama_tokenize.side_effect = mock_tokenize_impl
        
        # Call the tokenize method
        tokens = llm.tokenize("Test text")
        
        # Verify the result
        assert tokens == [1, 2, 3, 4, 5]
        
        # Verify the function was called with correct parameters
        mock_llama_cpp.llama_tokenize.assert_called_once()
        args = mock_llama_cpp.llama_tokenize.call_args[0]
        assert args[0] == llm.vocab
        assert args[1] == b"Test text"
        assert args[4] > 0  # Buffer size should be positive
        assert args[5] is True  # Add BOS token
        assert args[6] is True  # Allow special tokens

    def test_setup_sampler(self, mock_llama_cpp, batched_llm):
        """Test sampler chain setup."""
        # Call the setup_sampler method
        batched_llm.setup_sampler(top_k=40, top_p=0.95, temp=0.8, seed=42, min_keep=1)
        
        # Verify the sampler chain was initialized
        mock_llama_cpp.llama_sampler_chain_default_params.assert_called_once()
        mock_llama_cpp.llama_sampler_chain_init.assert_called_once()
        
        # Verify the samplers were added to the chain
        assert mock_llama_cpp.llama_sampler_chain_add.call_count == 4
        
        # Verify the samplers were initialized with correct parameters
        mock_llama_cpp.llama_sampler_init_top_k.assert_called_once_with(40)
        mock_llama_cpp.llama_sampler_init_top_p.assert_called_once_with(0.95, 1)
        mock_llama_cpp.llama_sampler_init_temp.assert_called_once_with(0.8)
        mock_llama_cpp.llama_sampler_init_dist.assert_called_once_with(42)

    def test_generate(self, mock_llama_cpp, batched_llm):
        """Test the generate method."""
        # Mock the helper methods
        with patch.object(batched_llm, '_prepare_prompt_and_batch', return_value=([1, 2, 3, 4, 5], 10)):
            with patch.object(batched_llm, '_handle_encoder_decoder'):
                with patch.object(batched_llm, '_sample_token', return_value=(42, -0.5)):
                    with patch.object(batched_llm, '_finalize_results', return_value=(["Generated text"], {"tokens_per_second": 10})):
                        # Call the generate method
                        results, stats, logprobs = batched_llm.generate(
                            prompt="Test prompt",
                            n_predict=10,
                            n_parallel=2,
                            top_k=40,
                            top_p=0.95,
                            temp=0.8,
                            seed=42,
                            min_keep=1
                        )
                        
                        # Verify the helper methods were called
                        batched_llm._prepare_prompt_and_batch.assert_called_once_with("Test prompt", 10, 2)
                        batched_llm._handle_encoder_decoder.assert_called_once()
                        
                        # Verify the setup_sampler method was called
                        assert mock_llama_cpp.llama_sampler_chain_init.call_count > 0
                        
                        # Verify the KV cache was copied to parallel sequences
                        mock_llama_cpp.llama_kv_cache_seq_cp.assert_called_once_with(batched_llm.ctx, 0, 1, -1, -1)
                        
                        # Verify the results
                        assert results == ["Generated text"]
                        assert "tokens_per_second" in stats
                        assert isinstance(logprobs, list)

    def test_prepare_prompt_and_batch(self, mock_llama_cpp, batched_llm):
        """Test the _prepare_prompt_and_batch helper method."""
        # Call the method directly
        tokens_list, batch_size = batched_llm._prepare_prompt_and_batch("Test prompt", 10, 2)
        
        # Verify the tokenize method was called
        assert isinstance(tokens_list, list)
        
        # Verify the batch was initialized
        mock_llama_cpp.llama_batch_init.assert_called()
        
        # Verify the batch size is correct
        assert batch_size > 0

    def test_handle_encoder_decoder(self, mock_llama_cpp, batched_llm):
        """Test the _handle_encoder_decoder helper method."""
        # Test when the model is not an encoder-decoder
        mock_llama_cpp.llama_model_has_encoder.return_value = False
        batched_llm._handle_encoder_decoder()
        
        # Verify encode was not called
        mock_llama_cpp.llama_encode.assert_not_called()
        
        # Test when the model is an encoder-decoder
        mock_llama_cpp.llama_model_has_encoder.return_value = True
        batched_llm._handle_encoder_decoder()
        
        # Verify encode was called
        mock_llama_cpp.llama_encode.assert_called_once()
        
        # Verify the batch was cleared
        assert batched_llm.batch.n_tokens == 0

    def test_sample_token(self, mock_llama_cpp, batched_llm):
        """Test the _sample_token helper method."""
        # Mock the logits_to_logprobs method
        with patch.object(batched_llm, 'logits_to_logprobs', return_value=np.array([-0.5] * 32000)):
            # Call the method
            token_id, logprob = batched_llm._sample_token(0, [5], 6)
            
            # Verify the sampler was called
            mock_llama_cpp.llama_sampler_sample.assert_called_once_with(batched_llm.smpl, batched_llm.ctx, 5)
            
            # Verify the logits were retrieved
            mock_llama_cpp.llama_get_logits_ith.assert_called_once_with(batched_llm.ctx, 5)
            
            # Verify the results
            assert token_id == 42  # From the mock
            assert logprob == -0.5

    def test_finalize_results(self, batched_llm):
        """Test the _finalize_results helper method."""
        # Call the method
        streams = ["generated1", "generated2"]
        t_start = time.time() - 1  # 1 second ago
        t_end = time.time()
        
        results, stats = batched_llm._finalize_results(
            prompt="Test prompt",
            streams=streams,
            n_parallel=2,
            n_decode=10,
            t_main_start=t_start,
            t_main_end=t_end
        )
        
        # Verify the results
        assert len(results) == 2
        assert results[0] == "Test prompt" + streams[0]
        assert results[1] == "Test prompt" + streams[1]
        
        # Verify the stats
        assert stats["n_decode"] == 10
        assert stats["time_s"] > 0
        assert stats["tokens_per_second"] > 0
