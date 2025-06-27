import unittest
from unittest.mock import Mock, patch
import numpy as np
from sam_detector import SAMDetector
from new_object_detector import ObjectDetector

class TestSAMIntegration(unittest.TestCase):
    
    @patch.dict('os.environ', {'REPLICATE_API_TOKEN': ''})
    def test_sam_disabled_without_token(self):
        """Test SAM is disabled when no API token"""
        detector = SAMDetector()
        self.assertFalse(detector.enabled)

    @patch.dict('os.environ', {'REPLICATE_API_TOKEN': 'fake_token', 'ENABLE_SAM_DETECTION': 'off'})
    def test_sam_disabled_by_flag(self):
        """Test SAM is disabled by the feature flag"""
        detector = SAMDetector()
        self.assertFalse(detector.enabled)

    def test_merge_detections_no_overlap(self):
        """Test merging when objects don't overlap"""
        yolo_objects = [{'bbox': [0, 0, 10, 10], 'class': 'chair', 'source': 'yolo'}]
        sam_objects = [{'bbox': [20, 20, 30, 30], 'class': 'table', 'source': 'sam'}]
        
        detector = ObjectDetector()
        merged = detector._merge_detections(yolo_objects, sam_objects)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['source'], 'yolo')
        self.assertEqual(merged[1]['source'], 'sam')

    def test_merge_detections_with_overlap(self):
        """Test merging when objects overlap significantly (IoU > 0.5)"""
        yolo_objects = [{'bbox': [0, 0, 10, 10], 'class': 'chair', 'source': 'yolo'}]
        sam_objects = [{'bbox': [1, 1, 11, 11], 'class': 'armchair', 'source': 'sam'}]

        detector = ObjectDetector()
        merged = detector._merge_detections(yolo_objects, sam_objects)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['source'], 'yolo') # Should keep the YOLO object

    def test_merge_detections_updates_class(self):
        """Test that merging updates class for 'unknown' YOLO objects"""
        yolo_objects = [{'bbox': [0, 0, 10, 10], 'class': 'object', 'source': 'yolo'}]
        sam_objects = [{'bbox': [1, 1, 11, 11], 'class': 'lamp', 'source': 'sam'}]

        detector = ObjectDetector()
        merged = detector._merge_detections(yolo_objects, sam_objects)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['class'], 'lamp') # Class should be updated

    @patch('replicate.run')
    def test_api_failure_fallback(self, mock_replicate_run):
        """Test graceful fallback when SAM API call fails"""
        mock_replicate_run.side_effect = Exception("API Error")
        
        # We need to patch the ObjectDetector's sam_detector instance
        with patch.dict('os.environ', {'REPLICATE_API_TOKEN': 'fake-token'}):
            main_detector = ObjectDetector()
            # Ensure the detector is enabled for the test
            main_detector.sam_detector.enabled = True
            
            # This call should not raise an exception, but log an error
            # and return an empty list from the sam_detector.
            sam_results = main_detector.sam_detector.detect_objects('dummy_path.jpg')
            self.assertEqual(sam_results, [])

if __name__ == '__main__':
    # We need to be able to import the other modules, so we add the parent dir to path
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    
    # Mock the model loading in ObjectDetector for testing purposes
    with patch('new_object_detector.ObjectDetector._load_model', return_value=None):
        unittest.main() 