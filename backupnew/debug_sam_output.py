import os
import json
import replicate
from sam_detector import SAMDetector

# Test with the example image from the API
test_url = "https://replicate.delivery/pbxt/IeDgvgehYgR4YpUT8SqRjP7qLisjjKbJ0MsAUaHII5FhHpVN/a.jpg"

output = replicate.run(
    "cjwbw/semantic-segment-anything:b2691db53f2d96add0051a4a98e7a3861bd21bf5972031119d344d956d2f8256",
    input={"image": test_url}
)

# Save the raw output for inspection
with open("sam_raw_output.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"Output type: {type(output)}")
print(f"Output: {output}")

# Test the detector
detector = SAMDetector()
objects = detector._process_sam_output(output)
print(f"\nProcessed {len(objects)} objects:")
for obj in objects:
    print(f"  - {obj['class']} (confidence: {obj['confidence']:.2f})") 