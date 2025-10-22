"""
Gyde Voice AI Cross-Sell Agent - LOCAL POC
No telephony provider needed - runs completely locally with simulated audio

Requirements:
pip install openai python-dotenv sounddevice numpy pydub soundfile gtts
"""

import os
import json
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import time

# For local audio simulation (optional - for demo purposes)
try:
    from gtts import gTTS
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Note: Audio libraries not available. Running in text-only mode.")
    print("Install with: pip install gtts sounddevice soundfile")

# Check for OpenAI TTS (good quality voices)
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_TTS_AVAILABLE = True
except ImportError:
    OPENAI_TTS_AVAILABLE = False

# ==================== CONFIGURATION ====================

# Set your OpenAI API key (optional for demo)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_REAL_LLM = bool(OPENAI_API_KEY)

SIMULATE_DNC = os.getenv("SIMULATE_DNC", "false").lower() == "true"
DNC_PROBABILITY = float(os.getenv("DNC_PROBABILITY", "0.3"))  # 30% default


# ==================== DATA MODELS ====================

class ConversationStage(Enum):
    INTRO = "intro"
    VERIFY = "verify"
    PRESENT = "present"
    QUALIFY = "qualify"
    SCHEDULE = "schedule"
    CONFIRM = "confirm"
    CLOSE = "close"
    OPT_OUT = "opt_out"  # DNC requests


@dataclass
class ClientData:
    client_id: str
    first_name: str
    last_name: str
    phone: str
    email: str
    aca_plan: str
    has_dental: bool
    has_vision: bool
    has_spouse: bool
    broker_name: str
    broker_id: str


@dataclass
class ConversationTurn:
    timestamp: str
    speaker: str  # "agent" or "client"
    text: str
    audio_file: Optional[str] = None


# ==================== MOCK LLM (For demo without API key) ====================

class MockLLM:
    """Simulates LLM responses for demo purposes"""

    def __init__(self):
        self.response_templates = {
            "positive": [
                "That sounds great!",
                "Yes, I'm interested in learning more.",
                "Sure, I have a few minutes.",
                "Okay, tell me more about that.",
            ],
            "neutral": [
                "I'm not sure, what's the cost?",
                "How does that work?",
                "Can you explain that again?",
            ],
            "objection": [
                "That seems expensive.",
                "I don't really go to the dentist much.",
                "I need to think about it.",
            ],
            "schedule": [
                "Thursday at 2pm works for me.",
                "Friday morning is better.",
            ],
            "dnc": [
                "Please stop calling me.",
                "I'm not interested, don't call again.",
                "Remove me from your list.",
                "Take me off your calling list.",
                "I don't want any more calls.",
                "Stop contacting me."
            ]
        }

    def generate_response(self, stage: ConversationStage, conversation_history: List, client_data) -> str:
        """Generate contextually appropriate response"""
        if SIMULATE_DNC and stage != ConversationStage.INTRO:
            if random.random() < DNC_PROBABILITY:
                dnc_response = random.choice(self.response_templates["dnc"])
                print(f"   [🎲 Simulated DNC response triggered]")
                return dnc_response

        if stage == ConversationStage.INTRO:
            return "I'm doing well, thanks. What's this about?"
        elif stage == ConversationStage.VERIFY:
            return "Sure, I have a couple minutes."
        elif stage == ConversationStage.PRESENT:
            return "I did have a dental cleaning a few months ago. What's the coverage like?"
        elif stage == ConversationStage.QUALIFY:
            return "Yes, that would be helpful to learn more about the options."
        elif stage == ConversationStage.SCHEDULE:
            return "Thursday at 2pm works for me."
        elif stage == ConversationStage.CONFIRM:
            return "No, that's all. Thank you!"
        else:
            return "Okay, sounds good."


# ==================== REAL LLM (OpenAI) ====================

class RealLLM:
    """Uses OpenAI API for realistic responses"""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
        except Exception as e:
            print(f"OpenAI client initialization failed: {e}")
            self.available = False

    def generate_response(self, stage: ConversationStage, conversation_history: List, client_data) -> str:
        """Generate realistic client response using GPT"""
        if not self.available:
            return MockLLM().generate_response(stage, conversation_history)

        # Create context for the LLM
        context = f"""You are simulating a real client named {client_data.first_name} receiving a call 
about dental and vision insurance. You currently have {client_data.aca_plan} but no dental or vision coverage.
You should respond naturally as a real person would - sometimes interested, sometimes skeptical, 
sometimes asking questions. Be realistic and varied in your responses.

Current conversation stage: {stage.value}
"""

        messages = [
            {"role": "system", "content": context},
        ]

        # Add conversation history
        for turn in conversation_history[-4:]:  # Last 4 turns for context
            role = "assistant" if turn['speaker'] == "agent" else "user"
            messages.append({"role": role, "content": turn['text']})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API error: {e}")
            return MockLLM().generate_response(stage, conversation_history)


# ==================== CONVERSATION ENGINE ====================

class LocalConversationEngine:
    """Manages the conversation flow locally"""

    def __init__(self, client_data: ClientData, use_real_llm: bool = False):
        self.client = client_data
        self.stage = ConversationStage.INTRO
        self.conversation_history: List[Dict] = []
        self.appointment_time = None

        # Initialize LLM
        if use_real_llm and OPENAI_API_KEY:
            self.llm = RealLLM(OPENAI_API_KEY)
        else:
            self.llm = MockLLM()

    def check_for_opt_out(self, text: str) -> bool:
        """Check if client wants to opt-out/be added to DNC list"""
        opt_out_phrases = [
            'stop calling',
            "don't call",
            'do not call',
            'remove me',
            'take me off',
            'not interested',
            'leave me alone',
            'stop contacting',
            'unsubscribe',
            'opt out',
            'opt-out',
            'no more calls',
            'never call again'
        ]

        text_lower = text.lower()
        return any(phrase in text_lower for phrase in opt_out_phrases)

    def get_agent_script(self) -> str:
        """Get agent's script for current stage"""
        scripts = {
            ConversationStage.INTRO:
                f"Hi {self.client.first_name}, this is Emma calling from {self.client.broker_name}'s office. How are you doing today?",

            ConversationStage.VERIFY:
                f"I wanted to reach out because you're currently enrolled in our {self.client.aca_plan} health plan. I noticed you might be eligible for some additional coverage options that could benefit you. Do you have a couple minutes to chat?",

            ConversationStage.PRESENT:
                f"Great! I see you don't currently have dental or vision coverage through us. Many of our ACA clients add these for comprehensive protection. Dental coverage starts at just $25/month and vision at $15/month. {'Since you have a spouse on your plan, you could cover both of you.' if self.client.has_spouse else ''} Have you had any dental or vision expenses recently?",

            ConversationStage.QUALIFY:
                f"I completely understand. These plans can really help with routine care - cleanings, exams, glasses, contacts. Plus they're separate from your ACA deductible, so you get immediate benefits. Would it make sense to have {self.client.broker_name} walk you through the specific plan options and pricing for your situation?",

            ConversationStage.SCHEDULE:
                f"Perfect! Let me get you scheduled. {self.client.broker_name} has availability this week. Would Thursday at 2pm or Friday at 10am work better for you?",

            ConversationStage.CONFIRM:
                f"Excellent! I've got you scheduled for Thursday at 2pm. You'll receive a calendar invite at {self.client.email}, and {self.client.broker_name} will call you then. He'll have all your information ready and can enroll you right on that call if you decide to move forward. Is there anything else I can help with today?",

            ConversationStage.OPT_OUT:
                f"I completely understand, {self.client.first_name}. I've added you to our do-not-call list right now, and you won't receive any more calls from us. I apologize for any inconvenience. Have a good day.",

            ConversationStage.CLOSE:
                f"Great talking with you, {self.client.first_name}! Have a wonderful day and we'll talk to you Thursday!"
        }
        return scripts[self.stage]

    def advance_stage(self) -> bool:
        """Move to next conversation stage"""
        stage_order = list(ConversationStage)
        current_index = stage_order.index(self.stage)

        if current_index < len(stage_order) - 1:
            self.stage = stage_order[current_index + 1]
            return True
        return False

    def add_turn(self, speaker: str, text: str, audio_file: Optional[str] = None):
        """Record a conversation turn"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "text": text,
            "audio_file": audio_file
        }
        self.conversation_history.append(turn)

    def get_client_response(self) -> str:
        """Generate client's response"""
        return self.llm.generate_response(self.stage, self.conversation_history, self.client)


# ==================== AUDIO SIMULATION ====================

class AudioSimulator:
    """Simulates text-to-speech for demo purposes"""

    def __init__(self, output_dir: str = "audio_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.audio_enabled = AUDIO_AVAILABLE
        self.audio_segments = []  # Store all audio segments for combining

    def text_to_speech(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech and save to file"""
        if not self.audio_enabled:
            return None

        try:
            filepath = os.path.join(self.output_dir, filename)
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)

            # Store segment info for later combining
            self.audio_segments.append({
                'filepath': filepath,
                'text': text,
                'filename': filename
            })

            return filepath
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def combine_audio_segments(self, output_filename: str = "full_conversation.mp3") -> Optional[str]:
        """Combine all audio segments into one file"""
        if not self.audio_enabled or not self.audio_segments:
            return None

        try:
            from pydub import AudioSegment
            from pydub.utils import which

            # Check if ffmpeg is available
            if which("ffmpeg") is None and which("avconv") is None:
                print("⚠️  ffmpeg/avconv not found in PATH")
                print("   Install ffmpeg:")
                print("   - Windows: Download from https://ffmpeg.org/download.html and add to PATH")
                print("   - Mac: brew install ffmpeg")
                print("   - Linux: apt-get install ffmpeg")
                return self._combine_audio_simple(output_filename)

            print(f"🎵 Combining {len(self.audio_segments)} audio segments...")

            # Verify all files exist
            missing_files = []
            for seg in self.audio_segments:
                if not os.path.exists(seg['filepath']):
                    missing_files.append(seg['filepath'])

            if missing_files:
                print(f"⚠️  Missing audio files: {missing_files}")
                return self._combine_audio_simple(output_filename)

            # Load first segment
            combined = AudioSegment.from_mp3(self.audio_segments[0]['filepath'])

            # Add silence between turns (500ms)
            silence = AudioSegment.silent(duration=500)

            # Combine all segments with pauses
            for i, segment_info in enumerate(self.audio_segments[1:], 1):
                try:
                    combined += silence
                    audio_segment = AudioSegment.from_mp3(segment_info['filepath'])
                    combined += audio_segment
                except Exception as e:
                    print(f"⚠️  Error loading segment {i}: {e}")
                    continue

            # Export combined audio
            output_path = os.path.join(self.output_dir, output_filename)
            combined.export(output_path, format="mp3")

            return output_path

        except ImportError:
            print("⚠️  pydub not installed. Install with: pip install pydub")
            return self._combine_audio_simple(output_filename)
        except Exception as e:
            print(f"⚠️  Error combining audio: {e}")
            print(f"   Falling back to segment list...")
            return self._combine_audio_simple(output_filename)

    def _combine_audio_simple(self, output_filename: str) -> Optional[str]:
        """Simple audio combining without pydub (creates file list and instructions)"""
        try:
            print("📝 Creating audio segments list (pydub/ffmpeg not available)...")

            output_path = os.path.join(self.output_dir, output_filename)

            # Save list of files
            list_path = output_path.replace('.mp3', '_segments.txt')
            with open(list_path, 'w') as f:
                f.write("Audio Segments (in order):\n")
                f.write("=" * 70 + "\n\n")
                for i, seg in enumerate(self.audio_segments, 1):
                    f.write(f"{i}. {seg['filename']}\n")
                    f.write(f"   Path: {seg['filepath']}\n")
                    f.write(f"   Text: {seg['text'][:50]}...\n\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("To combine manually with ffmpeg:\n")
                f.write("=" * 70 + "\n\n")

                # Create ffmpeg command
                f.write("1. Install ffmpeg:\n")
                f.write("   Windows: https://ffmpeg.org/download.html\n")
                f.write("   Mac: brew install ffmpeg\n")
                f.write("   Linux: apt-get install ffmpeg\n\n")

                f.write("2. Create file list (concat_list.txt):\n")
                concat_list = output_path.replace('.mp3', '_concat_list.txt')
                with open(concat_list, 'w') as concat_f:
                    for seg in self.audio_segments:
                        # Use forward slashes for cross-platform compatibility
                        file_path = seg['filepath'].replace('\\', '/')
                        concat_f.write(f"file '{file_path}'\n")

                f.write(f"   Created: {concat_list}\n\n")

                f.write("3. Run ffmpeg command:\n")
                f.write(f"   ffmpeg -f concat -safe 0 -i {concat_list} -c copy {output_path}\n\n")

            print(f"✓ Audio segments list saved: {list_path}")
            print(f"✓ Concat list saved: {concat_list}")
            print(f"\n💡 To combine audio files manually:")
            print(f"   1. Install ffmpeg (see instructions in {list_path})")
            print(f"   2. Run: ffmpeg -f concat -safe 0 -i {concat_list} -c copy {output_path}")

            return list_path

        except Exception as e:
            print(f"⚠️  Error creating audio list: {e}")
            return None

    def play_audio(self, filepath: str):
        """Play audio file"""
        if not self.audio_enabled or not filepath or not os.path.exists(filepath):
            return

        try:
            data, samplerate = sf.read(filepath)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Audio playback error: {e}")


# ==================== LOCAL VOICE AI AGENT ====================

class LocalVoiceAIAgent:
    """Main agent that runs completely locally"""

    def __init__(self, client_data: ClientData, use_real_llm: bool = False, enable_audio: bool = False):
        self.client = client_data
        self.engine = LocalConversationEngine(client_data, use_real_llm)
        self.audio_sim = AudioSimulator() if enable_audio else None
        self.call_start_time = None
        self.call_end_time = None

    async def run_conversation(self, interactive: bool = False):
        """Run the complete conversation"""
        print("\n" + "=" * 70)
        print(f"VOICE AI CALL SIMULATION")
        print(f"Client: {self.client.first_name} {self.client.last_name}")
        print(f"Phone: {self.client.phone}")
        print("=" * 70 + "\n")

        self.call_start_time = datetime.now()

        # Simulate call connection
        print("📞 Dialing...")
        await asyncio.sleep(1)
        print("✓ Call connected\n")

        while True:
            # Agent speaks
            agent_text = self.engine.get_agent_script()
            print(f"🤖 AGENT: {agent_text}")

            # Generate audio if enabled
            audio_file = None
            if self.audio_sim:
                audio_file = self.audio_sim.text_to_speech(
                    agent_text,
                    f"agent_{self.engine.stage.value}.mp3"
                )
                if audio_file:
                    print(f"   [Audio saved: {audio_file}]")

            self.engine.add_turn("agent", agent_text, audio_file)

            # Simulate speaking time
            await asyncio.sleep(1.5)

            # Check if conversation is complete
            if self.engine.stage in [ConversationStage.CLOSE, ConversationStage.OPT_OUT]:
                break

            # Client responds
            if interactive:
                client_text = input(f"👤 {self.client.first_name.upper()}: ")
            else:
                # Auto-generate response
                await asyncio.sleep(0.5)
                client_text = self.engine.get_client_response()
                print(f"👤 {self.client.first_name.upper()}: {client_text}")

            self.engine.add_turn("client", client_text)

            # Check for opt-out/DNC request FIRST (highest priority)
            if self.engine.check_for_opt_out(client_text):
                print(f"\n⚠️  DNC REQUEST DETECTED")
                self.engine.stage = ConversationStage.OPT_OUT
                self.engine.dnc_requested = True
                # TODO: Call function to add client to DNC database
                continue  # Skip to opt-out response

            # Check for appointment scheduling
            if self.engine.stage == ConversationStage.SCHEDULE and "thursday" in client_text.lower():
                self.engine.appointment_time = "Thursday at 2pm"

            # Advance conversation
            await asyncio.sleep(1)
            self.engine.advance_stage()

        self.call_end_time = datetime.now()

        # Print call summary
        self._print_call_summary()

        # Save transcript
        self._save_transcript()

    def _print_call_summary(self):
        """Print summary of the call"""
        duration = (self.call_end_time - self.call_start_time).seconds

        print("\n" + "=" * 70)
        print("CALL SUMMARY")
        print("=" * 70)
        print(f"Duration: {duration} seconds")
        print(f"Turns: {len(self.engine.conversation_history)}")
        print(f"Appointment: {'✓ Scheduled' if self.engine.appointment_time else '✗ Not scheduled'}")
        if self.engine.appointment_time:
            print(f"  Time: {self.engine.appointment_time}")
        print(f"Outcome: Successful cross-sell conversation")
        print("=" * 70 + "\n")

    def _save_transcript(self):
        """Save conversation transcript to file"""
        filename = f"transcript_{self.client.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        transcript = {
            "client_id": self.client.client_id,
            "client_name": f"{self.client.first_name} {self.client.last_name}",
            "call_start": self.call_start_time.isoformat(),
            "call_end": self.call_end_time.isoformat(),
            "duration_seconds": (self.call_end_time - self.call_start_time).seconds,
            "appointment_scheduled": bool(self.engine.appointment_time),
            "appointment_time": self.engine.appointment_time,
            "conversation": self.engine.conversation_history
        }

        with open(filename, 'w') as f:
            json.dump(transcript, f, indent=2)

        print(f"💾 Transcript saved: {filename}")

        # Also save as readable text
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w') as f:
            f.write(f"GYDE VOICE AI CALL TRANSCRIPT\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(f"Client: {self.client.first_name} {self.client.last_name}\n")
            f.write(f"Call Date: {self.call_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {(self.call_end_time - self.call_start_time).seconds} seconds\n")
            f.write(f"\n{'=' * 70}\n\n")

            for turn in self.engine.conversation_history:
                speaker = "AGENT" if turn['speaker'] == "agent" else self.client.first_name.upper()
                f.write(f"{speaker}: {turn['text']}\n\n")

            f.write(f"{'=' * 70}\n")
            f.write(f"OUTCOME: {'Appointment Scheduled' if self.engine.appointment_time else 'No Appointment'}\n")
            if self.engine.appointment_time:
                f.write(f"Appointment: {self.engine.appointment_time}\n")

        print(f"📄 Text transcript saved: {text_filename}")

        # Combine audio segments if audio was enabled
        if self.audio_sim and self.audio_sim.audio_segments:
            print(f"\n🎵 Combining {len(self.audio_sim.audio_segments)} audio segments...")
            combined_audio = self.audio_sim.combine_audio_segments(
                f"full_conversation_{self.client.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )
            if combined_audio and combined_audio.endswith('.mp3'):
                print(f"✓ Full conversation audio: {combined_audio}")
        else:
            if self.audio_sim:
                print(f"\n⚠️  No audio segments to combine (audio may not have been generated)")


# ==================== EXAMPLE USAGE ====================

async def main():
    """Run the local POC"""

    # Sample client data
    client = ClientData(
        client_id="CL-12345",
        first_name="Sarah",
        last_name="Martinez",
        phone="+1-555-123-4567",
        email="sarah.m@example.com",
        aca_plan="Silver Select 2024",
        has_dental=False,
        has_vision=False,
        has_spouse=True,
        broker_name="Michael Chen",
        broker_id="BR-789"
    )

    print("\n" + "=" * 70)
    print("GYDE VOICE AI AGENT - LOCAL POC")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  OpenAI API: {'✓ Enabled' if USE_REAL_LLM else '✗ Using mock responses'}")
    print(f"  Audio: {'✓ Enabled' if AUDIO_AVAILABLE else '✗ Text-only mode'}")
    print("\nModes:")
    print("  1. Automated (agent converses with simulated client)")
    print("  2. Interactive (you play the client)")

    mode = input("\nSelect mode (1 or 2): ").strip()
    interactive = mode == "2"

    enable_audio = False
    if AUDIO_AVAILABLE:
        audio_choice = input("Enable audio generation? (y/n): ").strip().lower()
        enable_audio = audio_choice == 'y'

    # Create and run agent
    agent = LocalVoiceAIAgent(
        client_data=client,
        use_real_llm=USE_REAL_LLM,
        enable_audio=enable_audio
    )

    await agent.run_conversation(interactive=interactive)

    print("\n✓ POC Complete!")
    print("\nGenerated Files:")
    print("  - JSON transcript (for analysis)")
    print("  - Text transcript (human-readable)")
    if enable_audio:
        print("  - Audio files (in audio_output/ directory)")


if __name__ == "__main__":
    asyncio.run(main())