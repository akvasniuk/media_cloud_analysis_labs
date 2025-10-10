import boto3
import time
import requests
import json
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
FILE_NAME = os.getenv("FILE_NAME")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
AWS_S3_URL_EXPIRATION = 300

job_name = str(uuid.uuid4())

s3 = boto3.client("s3", region_name=AWS_REGION)
transcribe = boto3.client("transcribe", region_name=AWS_REGION)

def upload_file():
    s3.upload_file(FILE_NAME, BUCKET_NAME, FILE_NAME)
    print(f"Uploaded {FILE_NAME} to s3://{BUCKET_NAME}/{FILE_NAME}")

def transcribe_with_aws():
    media_uri = f"s3://{BUCKET_NAME}/{FILE_NAME}"

    # Note: For new AWS accounts created after 15 July 2025,
    # Amazon Transcribe is initially disabled. To use it, you must
    # subscribe to a paid support plan. I use Deepgram API to transcribe audio
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': media_uri},
        MediaFormat="mp3",
        LanguageCode="uk-UA"
    )

    print("Waiting for transcription to complete...")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        state = status['TranscriptionJob']['TranscriptionJobStatus']
        if state in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if state == "COMPLETED":
        url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        print(f"AWS Transcribe completed. JSON result URL: {url}")

        result = requests.get(url).json()
        text = result["results"]["transcripts"][0]["transcript"]
        print("Transcript (AWS):", text)
    else:
        print("AWS Transcribe failed")

def transcribe_with_deepgram():
    presigned_url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': FILE_NAME},
        ExpiresIn=AWS_S3_URL_EXPIRATION
    )
    print(f"Presigned URL for Deepgram: {presigned_url}")

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}"
    }

    response = requests.post(
        DEEPGRAM_API_URL,
        headers=headers,
        json={"url": presigned_url},
        params={
            "punctuate": "true",
            "language": "uk"
        }
    )

    result = response.json()
    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    print("Transcript (Deepgram):", transcript)

    output_file = FILE_NAME.replace(".mp3", "_transcript.txt").replace(".wav", "_transcript.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"Transcript saved to {output_file}")


if __name__ == "__main__":
    upload_file()
    transcribe_with_deepgram()