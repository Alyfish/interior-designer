#!/usr/bin/env python3
"""
Debug script to check the actual SAM API response format
"""
import replicate
import json
import os
import sys
from config import REPLICATE_API_TOKEN

# Set up API token
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def test_sam_api():
    """Test the SAM API with a sample furniture image"""
    print("🔍 Testing SAM API Response Format...")
    
    # Test with a furniture image from Replicate's examples
    test_url = "https://replicate.delivery/pbxt/IeDgvgehYgR4YpUT8SqRjP7qLisjjKbJ0MsAUaHII5FhHpVN/a.jpg"
    
    try:
        print("Creating prediction...")
        prediction = replicate.predictions.create(
            "cjwbw/semantic-segment-anything:b2691db53f2d96add0051a4a98e7a3861bd21bf5972031119d344d956d2f8256",
            input={"image": test_url}
        )
        
        print(f"Prediction ID: {prediction.id}")
        print("Waiting for completion...")
        
        # Wait for completion
        prediction.wait()
        
        print(f"Status: {prediction.status}")
        print(f"Output type: {type(prediction.output)}")
        
        if prediction.output:
            # Save the output for inspection
            with open("sam_debug_output.json", "w") as f:
                json.dump(prediction.output, f, indent=2, default=str)
            
            print("✅ Output saved to sam_debug_output.json")
            
            if isinstance(prediction.output, dict):
                print(f"📋 Output keys: {list(prediction.output.keys())}")
                
                # Check for json_out
                if 'json_out' in prediction.output:
                    json_out = prediction.output['json_out']
                    print(f"📊 json_out type: {type(json_out)}")
                    
                    if isinstance(json_out, str):
                        print(f"📝 json_out is a string (first 200 chars): {json_out[:200]}...")
                        try:
                            parsed = json.loads(json_out)
                            if isinstance(parsed, list):
                                print(f"🎯 Found {len(parsed)} annotations after parsing")
                                if parsed:
                                    print(f"📝 First annotation keys: {list(parsed[0].keys())}")
                                    print(f"📝 First annotation sample: {json.dumps(parsed[0], indent=2)[:500]}...")
                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse json_out as JSON: {e}")
                    
                    elif isinstance(json_out, list):
                        print(f"🎯 Found {len(json_out)} annotations directly")
                        if json_out:
                            print(f"📝 First annotation keys: {list(json_out[0].keys())}")
                            print(f"📝 First annotation sample: {json.dumps(json_out[0], indent=2)[:500]}...")
                    
                    else:
                        print(f"🤔 Unexpected json_out type: {type(json_out)}")
                
                # Check for img_out
                if 'img_out' in prediction.output:
                    img_out = prediction.output['img_out']
                    print(f"🖼️ img_out type: {type(img_out)}")
                    if isinstance(img_out, str):
                        print(f"🖼️ img_out URL: {img_out}")
        else:
            print("❌ No output received!")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()

def test_with_replicate_run():
    """Test using replicate.run() method for comparison"""
    print("\n🔄 Testing with replicate.run() method...")
    
    test_url = "https://replicate.delivery/pbxt/IeDgvgehYgR4YpUT8SqRjP7qLisjjKbJ0MsAUaHII5FhHpVN/a.jpg"
    
    try:
        output = replicate.run(
            "cjwbw/semantic-segment-anything:b2691db53f2d96add0051a4a98e7a3861bd21bf5972031119d344d956d2f8256",
            input={"image": test_url}
        )
        
        print(f"✅ replicate.run() output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"📋 replicate.run() keys: {list(output.keys())}")
            
            # Save this output too
            with open("sam_run_output.json", "w") as f:
                json.dump(output, f, indent=2, default=str)
            print("✅ replicate.run() output saved to sam_run_output.json")
        
    except Exception as e:
        print(f"❌ replicate.run() test failed: {e}")

if __name__ == "__main__":
    print("🧪 SAM API Debug Script")
    print("="*50)
    
    # Test both methods
    test_sam_api()
    test_with_replicate_run()
    
    print("\n✅ Debug script completed!")
    print("Check sam_debug_output.json and sam_run_output.json for details.") 