import runpod
import os
import json
from fastapi.responses import HTMLResponse

# Set environment variables
os.environ["ELEVENLABS_API_KEY"] = "sk_86910b1ddb6b98f817323852eb68a6b6d6665f8f4ce51d2d"
os.environ["ELEVENLABS_VOICE_ID"] = "kdmDKE6EkgrWrrykO9Qt"


def handler(event):
    """
    RunPod serverless handler for Twilio webhooks
    """
    try:
        # Get the HTTP method and path from the event
        method = event.get('requestContext', {}).get(
            'http', {}).get('method', 'GET')
        path = event.get('requestContext', {}).get('http', {}).get('path', '/')

        # Handle root endpoint
        if path == '/':
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"message": "Custom Twilio Voice Bot is running!"})
            }

        # Handle Twilio webhook endpoint
        elif path == '/incoming-call':
            response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Hello! This is working from RunPod serverless!</Say>
    <Hangup/>
</Response>"""
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/xml"},
                "body": response
            }

        # Default response
        else:
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Not found"})
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
