"""Unit tests for Liger Triton GRPO loss function."""

import pytest
import torch

# The liger_kernel may not be installed in test environments
from nemo_rl.algorithms.loss.liger_grpo_loss import (
    LigerGRPOLossConfig,
    LigerGRPOLossFn,
    _LIGER_AVAILABLE,
)


class TestLigerGRPOLossConfig:
    def test_config_structure(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.04,
            "temperature": 1.0,
            "loss_type": "grpo",
        }
        assert cfg["clip_eps_low"] == 0.2
        assert cfg["beta"] == 0.04

    def test_config_with_is_correction(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "grpo",
            "enable_vllm_is_correction": True,
            "vllm_is_correction_type": "tis",
            "vllm_is_truncated_threshold": [0.5, 2.0],
        }
        assert cfg["enable_vllm_is_correction"] is True


class TestLigerGRPOLossFnInit:
    @pytest.mark.skipif(not _LIGER_AVAILABLE, reason="liger_kernel not installed")
    def test_init_succeeds(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.04,
            "temperature": 1.0,
            "loss_type": "grpo",
        }
        loss_fn = LigerGRPOLossFn(cfg)
        from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType

        assert loss_fn.input_type == LossInputType.LOGIT
        assert loss_fn.loss_type == LossType.TOKEN_LEVEL

    @pytest.mark.skipif(_LIGER_AVAILABLE, reason="liger_kernel is installed")
    def test_init_raises_without_liger(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "grpo",
        }
        with pytest.raises(ImportError, match="liger_kernel"):
            LigerGRPOLossFn(cfg)

    @pytest.mark.skipif(not _LIGER_AVAILABLE, reason="liger_kernel not installed")
    def test_invalid_is_correction_type(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "grpo",
            "enable_vllm_is_correction": True,
            "vllm_is_correction_type": "invalid",
            "vllm_is_truncated_threshold": [0.5, 2.0],
        }
        with pytest.raises(ValueError, match="Invalid vllm_is_correction_type"):
            LigerGRPOLossFn(cfg)


class TestLigerGRPOLossVllmISRatio:
    @pytest.mark.skipif(not _LIGER_AVAILABLE, reason="liger_kernel not installed")
    def test_no_is_correction_returns_none(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "grpo",
        }
        loss_fn = LigerGRPOLossFn(cfg)
        ratio, kl = loss_fn._compute_vllm_is_ratio(
            torch.zeros(2, 10), torch.zeros(2, 10), torch.ones(2, 10)
        )
        assert ratio is None
        assert kl is None

    @pytest.mark.skipif(not _LIGER_AVAILABLE, reason="liger_kernel not installed")
    def test_tis_correction(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "grpo",
            "enable_vllm_is_correction": True,
            "vllm_is_correction_type": "tis",
            "vllm_is_truncated_threshold": [0.5, 2.0],
        }
        loss_fn = LigerGRPOLossFn(cfg)
        old_lp = torch.randn(2, 10)
        rollout_lp = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        ratio, kl = loss_fn._compute_vllm_is_ratio(old_lp, rollout_lp, mask)
        assert ratio is not None
        assert kl is not None
        assert ratio.shape == (2, 10)
        # TIS should clamp values
        assert (ratio >= 0.5).all()
        assert (ratio <= 2.0).all()

    @pytest.mark.skipif(not _LIGER_AVAILABLE, reason="liger_kernel not installed")
    def test_icepop_correction(self):
        cfg: LigerGRPOLossConfig = {
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.2,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "grpo",
            "enable_vllm_is_correction": True,
            "vllm_is_correction_type": "icepop",
            "vllm_is_truncated_threshold": [0.5, 2.0],
        }
        loss_fn = LigerGRPOLossFn(cfg)
        old_lp = torch.randn(2, 10)
        rollout_lp = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        ratio, kl = loss_fn._compute_vllm_is_ratio(old_lp, rollout_lp, mask)
        assert ratio is not None
        # ICEPOP zeros out values outside range
        assert (ratio >= 0.0).all()


class TestLigerGRPOLossLazyImport:
    def test_lazy_import_from_init(self):
        from nemo_rl.algorithms.loss import LigerGRPOLossFn as LazyLiger

        assert LazyLiger is LigerGRPOLossFn

    def test_lazy_import_config(self):
        from nemo_rl.algorithms.loss import LigerGRPOLossConfig as LazyConfig

        assert LazyConfig is LigerGRPOLossConfig
