import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask2former_detector import Mask2FormerDetector
from new_object_detector import ObjectDetector, SegBackend


class TestMask2FormerIntegration(unittest.TestCase):
    """Test Mask2Former integration with hover functionality requirements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a dummy 64x64 test image
        self.test_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.test_image[10:30, 10:30] = [255, 0, 0]  # Red square
        self.test_image[40:60, 40:60] = [0, 255, 0]  # Green square
        
        # Save test image
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(self.temp_file.name, self.test_image)
        
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('mask2former_detector.replicate.run')
    @patch('mask2former_detector.requests.get')
    def test_mask2former_detector_interface(self, mock_requests_get, mock_replicate_run):
        """Test that Mask2Former detector returns the correct interface."""
        # Mock Replicate API response
        mock_replicate_run.return_value = "https://example.com/segmentation.png"
        
        # Create a mock segmentation image
        seg_image = np.zeros((64, 64, 3), dtype=np.uint8)
        seg_image[10:30, 10:30] = [100, 100, 100]  # Object 1
        seg_image[40:60, 40:60] = [150, 150, 150]  # Object 2
        
        # Mock the download of segmentation image
        seg_pil = Image.fromarray(seg_image)
        seg_bytes = MagicMock()
        seg_bytes.content = self._pil_to_bytes(seg_pil)
        mock_requests_get.return_value = seg_bytes
        
        # Test detector with mocked API token
        with patch.dict(os.environ, {'REPLICATE_API_TOKEN': 'test_token'}):
            detector = Mask2FormerDetector()
            image, objects, seg_path = detector.detect_objects(self.temp_file.name)
        
        # Verify interface
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(objects, list)
        self.assertIsInstance(seg_path, str)
        
        # Verify object structure for hover functionality
        for obj in objects:
            self.assertIn('id', obj)
            self.assertIn('class', obj)
            self.assertIn('confidence', obj)
            self.assertIn('bbox', obj)
            self.assertIn('contours', obj)
            self.assertIn('source', obj)
            
            # Verify contours format for hover
            self.assertIsInstance(obj['contours'], list)
            self.assertGreater(len(obj['contours']), 0)
            self.assertIsInstance(obj['contours'][0], list)
            
            # Verify bbox format
            self.assertIsInstance(obj['bbox'], list)
            self.assertEqual(len(obj['bbox']), 4)
            
            # Verify source
            self.assertEqual(obj['source'], 'mask2former')
    
    @patch('mask2former_detector.replicate.run')
    def test_mask2former_fallback_to_yolo(self, mock_replicate_run):
        """Test fallback to YOLO when Mask2Former fails."""
        # Make Replicate fail
        mock_replicate_run.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {'REPLICATE_API_TOKEN': 'test_token'}):
            # Create ObjectDetector with Mask2Former backend
            detector = ObjectDetector(backend=SegBackend.MASK2FORMER)
            
            # Mock YOLO model to avoid actual loading
            with patch.object(detector, '_load_model'):
                with patch.object(detector, 'model') as mock_model:
                    # Setup mock YOLO results
                    mock_results = MagicMock()
                    mock_results[0].masks = None  # No masks for simplicity
                    mock_model.return_value = [mock_results]
                    
                    # Should fall back to YOLO
                    image, objects, seg_path = detector.detect_objects(self.temp_file.name)
                    
                    # Verify it returned valid results
                    self.assertIsInstance(image, np.ndarray)
                    self.assertIsInstance(objects, list)
                    self.assertIsInstance(seg_path, str)
    
    def test_contour_format_for_hover(self):
        """Test that contours are in the correct format for hover functionality."""
        # Create a simple mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contour as in detector
        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        contour_points = approx.reshape(-1, 2).tolist()
        
        # Verify format
        self.assertIsInstance(contour_points, list)
        self.assertGreater(len(contour_points), 2)  # At least 3 points for a polygon
        for point in contour_points:
            self.assertIsInstance(point, list)
            self.assertEqual(len(point), 2)  # [x, y]
            self.assertIsInstance(point[0], (int, np.integer))
            self.assertIsInstance(point[1], (int, np.integer))
    
    def test_backend_enum(self):
        """Test SegBackend enum values."""
        self.assertEqual(SegBackend.YOLOV8.value, "yolov8")
        self.assertEqual(SegBackend.MASK2FORMER.value, "mask2former")
    
    @patch('new_object_detector.ObjectDetector._load_mask2former')
    def test_object_detector_initialization(self, mock_load_mask2former):
        """Test ObjectDetector initialization with different backends."""
        # Test YOLO backend
        with patch('new_object_detector.ObjectDetector._load_model') as mock_load_yolo:
            detector_yolo = ObjectDetector(backend=SegBackend.YOLOV8)
            mock_load_yolo.assert_called_once()
            mock_load_mask2former.assert_not_called()
        
        # Reset mocks
        mock_load_mask2former.reset_mock()
        
        # Test Mask2Former backend
        with patch('new_object_detector.ObjectDetector._load_model') as mock_load_yolo:
            detector_mask2former = ObjectDetector(backend=SegBackend.MASK2FORMER)
            mock_load_yolo.assert_not_called()
            mock_load_mask2former.assert_called_once()
    
    def _pil_to_bytes(self, pil_image):
        """Convert PIL image to bytes for mocking."""
        from io import BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return buffer.getvalue()


class TestMask2FormerColorParsing(unittest.TestCase):
    """Test color parsing and mask generation from Mask2Former output."""
    
    @patch('mask2former_detector.requests.get')
    def test_parse_segmentation_output(self, mock_requests_get):
        """Test parsing of segmentation output with multiple objects."""
        # Create a mock segmentation image with different colored regions
        seg_image = np.zeros((100, 100, 3), dtype=np.uint8)
        seg_image[10:40, 10:40] = [50, 50, 50]   # Object 1
        seg_image[60:90, 60:90] = [100, 100, 100]  # Object 2
        seg_image[20:50, 60:80] = [150, 150, 150]  # Object 3
        
        # Mock the download
        seg_pil = Image.fromarray(seg_image)
        seg_bytes = MagicMock()
        seg_bytes.content = self._pil_to_bytes(seg_pil)
        mock_requests_get.return_value = seg_bytes
        
        with patch.dict(os.environ, {'REPLICATE_API_TOKEN': 'test_token'}):
            detector = Mask2FormerDetector()
            masks, color_mapping = detector._parse_segmentation_output("https://example.com/seg.png")
        
        # Verify we got masks for non-black regions
        self.assertGreater(len(masks), 0)
        self.assertGreater(len(color_mapping), 0)
        
        # Verify mask properties
        for class_name, mask in masks.items():
            self.assertIsInstance(mask, np.ndarray)
            self.assertEqual(mask.shape, (100, 100))
            self.assertIn(mask.dtype, [np.uint8])
            # Verify binary mask (only 0 and 255)
            unique_values = np.unique(mask)
            self.assertTrue(all(v in [0, 255] for v in unique_values))
    
    def _pil_to_bytes(self, pil_image):
        """Convert PIL image to bytes for mocking."""
        from io import BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return buffer.getvalue()


if __name__ == '__main__':
    unittest.main()