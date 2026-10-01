"""
Unit tests for the Explainable AI (XAI) Suite and Visualizer.
"""

import unittest
import torch
import numpy as np

from src.models.hybrid_classifier import HybridMedicinalClassifier
from src.explainability.gradcam import HybridExplainer
from src.explainability.visualizer import apply_colormap_overlay, create_side_by_side_comparison


class TestXAISuite(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.model = HybridMedicinalClassifier(
            num_classes=self.num_classes,
            fusion_type="gated_cross_attention",
            pretrained=False
        )
        self.explainer = HybridExplainer(self.model)
        self.dummy_tensor = torch.randn(1, 3, 224, 224)
        self.dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
    def test_explanation_generation(self):
        """Verify CAMs are generated with proper spatial shape and valid range [0, 1]."""
        results = self.explainer.generate_explanations(self.dummy_tensor, target_class=2)
        
        # Check keys
        self.assertIn("cnn_cam", results)
        self.assertIn("swin_cam", results)
        self.assertIn("fused_cam", results)
        self.assertIn("faithfulness_drop_percent", results)
        
        # Check shapes
        self.assertEqual(results["cnn_cam"].shape, (224, 224))
        self.assertEqual(results["swin_cam"].shape, (224, 224))
        self.assertEqual(results["fused_cam"].shape, (224, 224))
        
        # Check value bounds
        self.assertTrue(0.0 <= np.min(results["cnn_cam"]) and np.max(results["cnn_cam"]) <= 1.0)
        self.assertTrue(0.0 <= np.min(results["swin_cam"]) and np.max(results["swin_cam"]) <= 1.0)
        self.assertTrue(0.0 <= np.min(results["fused_cam"]) and np.max(results["fused_cam"]) <= 1.0)
        
    def test_visualizer_rendering(self):
        """Verify color overlay blending and 4-panel image generation."""
        dummy_cam = np.random.rand(224, 224).astype(np.float32)
        overlay = apply_colormap_overlay(self.dummy_rgb, dummy_cam, colormap="turbo", alpha=0.5)
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        
        explanations = {
            "cnn_cam": dummy_cam,
            "swin_cam": dummy_cam,
            "fused_cam": dummy_cam,
            "cnn_weight": 0.52,
            "swin_weight": 0.48,
            "confidence": 0.94,
            "faithfulness_drop_percent": 18.5
        }
        comparison_pil = create_side_by_side_comparison(
            self.dummy_rgb, explanations, species_name="Azadirachta_indica"
        )
        self.assertIsNotNone(comparison_pil)
        self.assertGreater(comparison_pil.size[0], 500)


if __name__ == "__main__":
    unittest.main()
