"""
Unit tests for the Hybrid EfficientNetV2 + Swin Transformer architecture.
Verifies shapes, forward pass, gradient flow, and fusion modes.
"""

import unittest
import torch

from src.models.backbones import EfficientNetV2Backbone, SwinTransformerBackbone
from src.models.fusion import GatedAdaptiveFusion, GatedCrossAttentionFusion, ConcatenationFusion
from src.models.hybrid_classifier import HybridMedicinalClassifier


class TestHybridModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_classes = 10
        self.dummy_input = torch.randn(self.batch_size, 3, 224, 224)
        
    def test_backbones(self):
        """Test CNN and Swin backbones return correct spatial and pooled dimensions."""
        cnn = EfficientNetV2Backbone(pretrained=False)
        f_cnn_spatial, f_cnn_pooled = cnn(self.dummy_input)
        self.assertEqual(f_cnn_spatial.shape, (self.batch_size, 1280, 7, 7))
        self.assertEqual(f_cnn_pooled.shape, (self.batch_size, 1280))
        
        swin = SwinTransformerBackbone(pretrained=False)
        f_swin_spatial, f_swin_pooled = swin(self.dummy_input)
        self.assertEqual(f_swin_spatial.shape, (self.batch_size, 768, 7, 7))
        self.assertEqual(f_swin_pooled.shape, (self.batch_size, 768))
        
    def test_fusion_modes(self):
        """Test all 3 fusion layers."""
        f_cnn_spatial = torch.randn(self.batch_size, 1280, 7, 7)
        f_cnn_pooled = torch.randn(self.batch_size, 1280)
        f_swin_spatial = torch.randn(self.batch_size, 768, 7, 7)
        f_swin_pooled = torch.randn(self.batch_size, 768)
        
        # 1. Gated Cross Attention
        gcaf = GatedCrossAttentionFusion(cnn_dim=1280, swin_dim=768, fusion_dim=512)
        fused_gcaf, meta_gcaf = gcaf(f_cnn_spatial, f_cnn_pooled, f_swin_spatial, f_swin_pooled)
        self.assertEqual(fused_gcaf.shape, (self.batch_size, 512))
        self.assertIn("cnn_weight", meta_gcaf)
        self.assertIn("swin_weight", meta_gcaf)
        
        # 2. Gated Adaptive Fusion
        gaf = GatedAdaptiveFusion(cnn_dim=1280, swin_dim=768, fusion_dim=512)
        fused_gaf, meta_gaf = gaf(f_cnn_spatial, f_cnn_pooled, f_swin_spatial, f_swin_pooled)
        self.assertEqual(fused_gaf.shape, (self.batch_size, 512))
        
        # 3. Concatenation
        concat = ConcatenationFusion(cnn_dim=1280, swin_dim=768, fusion_dim=512)
        fused_cat, _ = concat(f_cnn_spatial, f_cnn_pooled, f_swin_spatial, f_swin_pooled)
        self.assertEqual(fused_cat.shape, (self.batch_size, 512))
        
    def test_hybrid_classifier_forward_backward(self):
        """Test full model forward pass and gradient backpropagation."""
        model = HybridMedicinalClassifier(
            num_classes=self.num_classes,
            fusion_type="gated_cross_attention",
            pretrained=False
        )
        logits, metadata = model(self.dummy_input, return_features=True)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn("fusion_metadata", metadata)
        
        # Backward pass test
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist in both branches
        cnn_has_grad = any(p.grad is not None for p in model.cnn_branch.parameters())
        swin_has_grad = any(p.grad is not None for p in model.swin_branch.parameters())
        self.assertTrue(cnn_has_grad)
        self.assertTrue(swin_has_grad)


if __name__ == "__main__":
    unittest.main()
