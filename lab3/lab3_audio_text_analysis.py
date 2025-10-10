import argparse
import requests
import os
from dotenv import load_dotenv
from langdetect import detect
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from spacy.matcher import PhraseMatcher

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

def transcribe_with_deepgram(audio_file):
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    with open(audio_file, "rb") as f:
        audio_data = f.read()

    response = requests.post(
        DEEPGRAM_API_URL,
        headers=headers,
        data=audio_data,
        params={"punctuate": "true", "language": "en"}
    )

    if response.status_code != 200:
        raise Exception(f"Deepgram API error: {response.text}")

    result = response.json()
    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript.strip()

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    return lang

def analyze_sentiment(text):
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)

    analyzer = SentimentIntensityAnalyzer()
    sentences = nltk.tokenize.sent_tokenize(text)

    print("\n--- Sentence-level Sentiment ---")
    for sentence in sentences:
        scores = analyzer.polarity_scores(sentence)
        print(f"\nSentence: {sentence}")
        for k in sorted(scores):
            print(f"  {k}: {scores[k]}")
    
    overall_scores = analyzer.polarity_scores(text)
    compound = overall_scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment

def search_phrase_and_entities(text, phrase):
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(phrase)]
    matcher.add("Phrase", patterns)

    matches = matcher(doc)
    if matches:
        match_id, start, end = matches[0]
        span = doc[start:end]
        phrase_result = (f'Phrase FOUND: "{span.text}" '
                         f'(tokens {start}-{end}, chars {span.start_char}-{span.end_char})')
    else:
        phrase_result = "Phrase NOT found"

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return phrase_result, entities

def save_results_to_file(filename, transcript, lang, sentiment, phrase_result, entities):
    """Save analysis results to a text file in the requested structure"""
    output_file = filename.replace(".wav", "_analysis.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Transcription:\n")
        f.write(transcript + "\n\n")
        f.write(f"Language:\n{lang}\n\n")
        f.write(f"Sentiment:\n{sentiment}\n\n")
        f.write(f"{phrase_result}\n\n")
        f.write("Named entities:\n")
        if entities:
            for text, label in entities:
                f.write(f"  - {text} ({label})")
        else:
            f.write("  None\n")
    print(f"\n Analysis saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Audio transcription and analysis tool.")
    parser.add_argument("--audio-source", required=True, help="Path to WAV audio file")
    parser.add_argument("--phrase", required=True, help="Phrase to search in transcript")
    args = parser.parse_args()

    transcript = transcribe_with_deepgram(args.audio_source)

    print("\n--- Transcription ---")
    print(transcript)

    lang = detect_language(transcript)
    sentiment = analyze_sentiment(transcript)
    phrase_result, entities = search_phrase_and_entities(transcript, args.phrase)

    print("\n--- Analysis Results ---")
    print(f"Language: {lang}")
    print(f"Sentiment: {sentiment}")
    print(phrase_result)

    if entities:
        print("Named entities:")
        for text, label in entities:
            print(f"  - {text} ({label})")
    else:
        print("No named entities found.")
    
    save_results_to_file(args.audio_source, transcript, lang, sentiment, phrase_result, entities)

if __name__ == "__main__":
    main()