"""Test Fish Speech model integration with vLLM."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from vllm.model_executor.models.fish_speech import (
    FishSpeechConfig,
    FishSpeechForCausalLM,
    FishSpeechAttention,
    FishSpeechMLP,
    FishSpeechDecoderLayer,
    FishSpeechModel,
)
from vllm.config import VllmConfig, ModelConfig, CacheConfig
from vllm.sequence import SamplingParams


class TestFishSpeechConfig:
    """Test FishSpeechConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FishSpeechConfig()
        
        assert config.model_type == "fish_speech"
        assert config.vocab_size == 32000
        assert config.n_layer == 32
        assert config.n_head == 32
        assert config.dim == 4096
        assert config.codebook_size == 160
        assert config.num_codebooks == 4
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FishSpeechConfig(
            vocab_size=50000,
            n_layer=24,
            n_head=16,
            dim=2048,
        )
        
        assert config.vocab_size == 50000
        assert config.n_layer == 24
        assert config.n_head == 16
        assert config.dim == 2048
        
    def test_post_init(self):
        """Test post initialization logic."""
        config = FishSpeechConfig(
            n_head=32,
            dim=4096,
        )
        
        # Test default values are set correctly
        assert config.n_local_heads == 32  # Should default to n_head
        assert config.intermediate_size == 16384  # Should be 4 * dim


class TestFishSpeechComponents:
    """Test individual Fish Speech model components."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return FishSpeechConfig(
            vocab_size=1000,
            n_layer=4,
            n_head=8,
            dim=512,
            intermediate_size=2048,
            n_local_heads=8,
            head_dim=64,
        )
    
    @pytest.fixture
    def vllm_config(self, config):
        """Create a mock vLLM configuration."""
        model_config = MagicMock()
        model_config.hf_config = config
        
        cache_config = MagicMock()
        quant_config = None
        lora_config = None
        
        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        vllm_config.cache_config = cache_config
        vllm_config.quant_config = quant_config
        vllm_config.lora_config = lora_config
        
        return vllm_config
    
    def test_fish_speech_attention_init(self, config):
        """Test FishSpeechAttention initialization."""
        with patch('vllm.model_executor.models.fish_speech.get_tensor_model_parallel_world_size', return_value=1):
            with patch('vllm.model_executor.models.fish_speech.get_rope') as mock_get_rope:
                with patch('vllm.model_executor.models.fish_speech.Attention') as mock_attention:
                    mock_get_rope.return_value = MagicMock()
                    mock_attention.return_value = MagicMock()
                    
                    attention = FishSpeechAttention(config, layer_idx=0)
                    
                    assert attention.layer_idx == 0
                    assert attention.total_num_heads == config.n_head
                    assert attention.head_dim == config.head_dim
    
    def test_fish_speech_mlp_init(self, config):
        """Test FishSpeechMLP initialization."""
        with patch('vllm.model_executor.models.fish_speech.MergedColumnParallelLinear') as mock_merged:
            with patch('vllm.model_executor.models.fish_speech.RowParallelLinear') as mock_row:
                mock_merged.return_value = MagicMock()
                mock_row.return_value = MagicMock()
                
                mlp = FishSpeechMLP(config)
                
                # Check that the components were initialized
                mock_merged.assert_called_once()
                mock_row.assert_called_once()
    
    def test_fish_speech_decoder_layer_init(self, config):
        """Test FishSpeechDecoderLayer initialization."""
        with patch('vllm.model_executor.models.fish_speech.FishSpeechAttention') as mock_attn:
            with patch('vllm.model_executor.models.fish_speech.FishSpeechMLP') as mock_mlp:
                with patch('vllm.model_executor.models.fish_speech.RMSNorm') as mock_norm:
                    mock_attn.return_value = MagicMock()
                    mock_mlp.return_value = MagicMock()
                    mock_norm.return_value = MagicMock()
                    
                    layer = FishSpeechDecoderLayer(config, layer_idx=1)
                    
                    assert layer.layer_idx == 1
                    mock_attn.assert_called_once_with(config, 1)
                    mock_mlp.assert_called_once_with(config)


class TestFishSpeechModel:
    """Test complete Fish Speech model."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return FishSpeechConfig(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            dim=256,
            intermediate_size=1024,
            n_local_heads=4,
            head_dim=64,
            codebook_size=100,
            num_codebooks=2,
        )
    
    @pytest.fixture
    def vllm_config(self, config):
        """Create a mock vLLM configuration."""
        model_config = MagicMock()
        model_config.hf_config = config
        
        cache_config = MagicMock()
        quant_config = None
        lora_config = None
        
        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        vllm_config.cache_config = cache_config
        vllm_config.quant_config = quant_config
        vllm_config.lora_config = lora_config
        
        return vllm_config
    
    def test_model_init(self, vllm_config):
        """Test FishSpeechModel initialization."""
        with patch('vllm.model_executor.models.fish_speech.get_pp_group') as mock_pp:
            with patch('vllm.model_executor.models.fish_speech.VocabParallelEmbedding') as mock_embed:
                with patch('vllm.model_executor.models.fish_speech.make_layers') as mock_layers:
                    with patch('vllm.model_executor.models.fish_speech.RMSNorm') as mock_norm:
                        # Mock pipeline parallel group
                        mock_pp_group = MagicMock()
                        mock_pp_group.is_last_rank = True
                        mock_pp.return_value = mock_pp_group
                        
                        # Mock other components
                        mock_embed.return_value = MagicMock()
                        mock_layers.return_value = (0, 2, [MagicMock(), MagicMock()])
                        mock_norm.return_value = MagicMock()
                        
                        model = FishSpeechModel(vllm_config=vllm_config)
                        
                        # Verify initialization
                        assert model.config == vllm_config.model_config.hf_config
                        assert model.vocab_size == vllm_config.model_config.hf_config.vocab_size
    
    def test_causal_lm_init(self, vllm_config):
        """Test FishSpeechForCausalLM initialization."""
        with patch('vllm.model_executor.models.fish_speech.get_pp_group') as mock_pp:
            with patch('vllm.model_executor.models.fish_speech.FishSpeechModel') as mock_model:
                with patch('vllm.model_executor.models.fish_speech.ParallelLMHead') as mock_lm_head:
                    with patch('vllm.model_executor.models.fish_speech.RMSNorm') as mock_norm:
                        with patch('vllm.model_executor.models.fish_speech.LogitsProcessor') as mock_processor:
                            # Mock pipeline parallel group
                            mock_pp_group = MagicMock()
                            mock_pp_group.is_last_rank = True
                            mock_pp.return_value = mock_pp_group
                            
                            # Mock other components
                            mock_model.return_value = MagicMock()
                            mock_lm_head.return_value = MagicMock()
                            mock_norm.return_value = MagicMock()
                            mock_processor.return_value = MagicMock()
                            
                            model = FishSpeechForCausalLM(vllm_config=vllm_config)
                            
                            # Verify initialization
                            assert model.config == vllm_config.model_config.hf_config
                            mock_model.assert_called_once()


class TestFishSpeechIntegration:
    """Test Fish Speech integration with vLLM."""
    
    def test_model_registration(self):
        """Test that Fish Speech model is properly registered."""
        from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
        
        # Check if Fish Speech model is registered
        assert "FishSpeechForCausalLM" in _TEXT_GENERATION_MODELS
        
        module_name, class_name = _TEXT_GENERATION_MODELS["FishSpeechForCausalLM"]
        assert module_name == "fish_speech"
        assert class_name == "FishSpeechForCausalLM"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = FishSpeechConfig(
            n_head=8,
            dim=512,
            n_local_heads=8,
        )
        
        # Should not raise any exceptions
        assert config.dim % config.n_head == 0
        assert config.dim % config.n_local_heads == 0
        
        # Invalid config should raise assertion error in post_init
        with pytest.raises(AssertionError):
            FishSpeechConfig(
                n_head=7,  # Not divisible by dim
                dim=512,
            )


if __name__ == "__main__":
    pytest.main([__file__])