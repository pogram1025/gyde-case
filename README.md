# 1. Clone repo
git clone <your-repo-url>
cd gyde-voice-ai-poc

# 2. Install dependencies
pip install openai python-dotenv gtts sounddevice soundfile pydub

# 3. Run POC
python local_voice_ai_poc.py

# 4. Follow prompts
#    - Select mode (1=automated, 2=interactive)
#    - Enable audio (y/n)
#    - Select voice provider (1=gTTS, 2=OpenAI, 3=ElevenLabs)

# 5. Review outputs
#    - Transcripts: transcript_*.json, transcript_*.txt
#    - Audio: audio_output/full_conversation_*.mp3

# 6. Simulate 'Do Not Call'
#    - Set environment variable 'SIMULATE_DNC=true'
#    - (Optional) Set environment variable 'DNC_PROBABILITY=<between 0 - 1>' (default .3)
#    -- The probability the response will be a 'do not call'
