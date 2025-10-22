# Technical Design Documentation
## Gyde Voice AI Cross-Sell Agent

---

## 1. Scaling to 100k+ Clients

**Architecture:** Event-driven, horizontally scalable microservices on Kubernetes

**Key Components:**
- **Load Balancer** → **API Gateway** → **Message Queue (SQS/Kafka)** → **Worker Pool (Auto-scaling)**
- Workers scale 10→1000+ pods based on queue depth
- Target: 100 concurrent calls per pod = 100k capacity
- **Database:** PostgreSQL sharded by broker_id (10k clients/shard), Redis for sessions
- **Multi-region:** US-East, US-West, EU for <100ms latency
- **Rate limiting:** 1000 calls/sec, LLM request batching (5-10ms windows)

**Performance:** 99.9% uptime, <500ms end-to-end latency, <200ms API response

---

## 2. LLM Prompting & Evaluations

**Prompt Architecture:**
```python
{
  "identity": "You are Emma, health insurance assistant...",
  "objective": "Schedule dental/vision appointments",
  "context": "{client_history, eligibility, broker_availability}",
  "constraints": "No price guarantees, respect DNC, HIPAA-compliant",
  "examples": [few_shot_examples]
}
```

**Evaluation Framework:**

**Offline (Pre-deployment):**
- Test set: 500 synthetic conversations
- Metrics: Scheduling accuracy (95%+), HIPAA violations (0), hallucination rate (<2%)

**Online (Production):**
- A/B testing: 5% traffic to experimental prompts
- Shadow mode: Run new prompts alongside production
- Human-in-loop: 2% random call review
- Real-time: Conversion rate, sentiment, appointment show-rate

**Version Control:** Git-based with automated regression testing, 30-second rollback

---

## 3. Data Privacy & HIPAA Compliance

**Encryption:**
- At rest: AES-256 (AWS KMS)
- In transit: TLS 1.3
- Key rotation: Automatic

**Access Controls (RBAC):**
```python
ai_agent: read=["name","plan","coverage_gaps"], forbidden=["ssn","medical_records"]
broker: read/write=["all_client_data"]
auditor: read=["logs","audit_trails"]
```

**Compliance Requirements:**
- Data minimization: AI receives only necessary fields
- Audit logs: All data access tracked with timestamp, actor, fields accessed
- Call recording: Verbal consent before proceeding, 90-day auto-deletion
- BAAs signed: Twilio, OpenAI/Azure, ElevenLabs, AWS
- Incident response: <1hr security notification, 24hr client notification

---

## 4. Twilio Voice Integration

**Real-time Media Stream via WebSocket:**

```python
# Step 1: Initiate call
call = twilio_client.calls.create(
    to=client_phone,
    from_=TWILIO_NUMBER,
    url=f"{BASE_URL}/twiml/connect"
)

# Step 2: Get consent
response = VoiceResponse()
response.say("This call may be recorded. Say 'I consent' to continue.")
gather = response.gather(input='speech', action='/consent-response')

# Step 3: Connect WebSocket for bidirectional audio
connect = Connect()
stream = connect.stream(url=f'wss://{BASE_URL}/media-stream', track='both_tracks')

# Step 4: Process audio in real-time
async for message in websocket:
    audio_bytes = base64.b64decode(message['media']['payload'])
    transcript = await stt_client.transcribe(audio_bytes)  # Deepgram
    response_text = await llm_client.get_response(transcript)  # OpenAI
    audio_response = await tts_client.synthesize(response_text)  # ElevenLabs
    await websocket.send({'event': 'media', 'payload': base64.b64encode(audio_response)})
```

**Features:** Interrupt handling, call transfer, <300ms latency

---

## 5. LLM/STT/TTS Provider Integration

**Multi-Provider Strategy with Fallbacks:**

| Service | Primary | Fallback | Emergency |
|---------|---------|----------|-----------|
| **STT** | Deepgram (40ms, 95%) | AssemblyAI (80ms) | Google (120ms) |
| **LLM** | OpenAI GPT-4o (200ms) | Claude Sonnet (250ms) | Llama3 local (500ms) |
| **TTS** | ElevenLabs (150ms) | OpenAI TTS (100ms) | Google (80ms) |

**Reliability Pattern:**
```python
class AIServiceClient:
    def __init__(self):
        self.circuit_breakers = {
            provider: CircuitBreaker(failure_threshold=5, recovery_timeout=30)
            for provider in ['deepgram', 'openai', 'elevenlabs']
        }
    
    async def stt_with_fallback(self, audio):
        for provider in ['deepgram', 'assembly_ai', 'google']:
            try:
                async with self.circuit_breakers[provider]:
                    return await self.stt_clients[provider].transcribe(audio)
            except: continue
        raise AllProvidersFailedError()
```

**Optimization:** Request batching, response caching (1hr TTL), parallel processing

---

## 6. Conversation Flow Handling

**State Machine (7 Stages):**
```
INTRO → VERIFY → PRESENT → QUALIFY → SCHEDULE → CONFIRM → CLOSE
```

**Special Flow Handling:**

**Opt-Out/DNC:**
```python
if detect_opt_out(user_input):
    await dnc_service.add_immediately(client_phone)
    await crm_service.update(client_id, {"dnc_status": True})
    return "I've added you to our do-not-call list. Have a good day."
```

**Transfer to Broker:**
```python
if detect_transfer_request(user_input):
    broker = await get_available_broker(broker_id)
    if broker:
        return transfer_call(broker.phone)
    else:
        return "Let me schedule a callback instead."
```

**Confusion Detection & Fallbacks:**
```python
FALLBACK_CHAIN = ["rephrase", "simplify", "examples", "transfer", "callback"]

if confusion_detected(user_input, history):
    for fallback in FALLBACK_CHAIN:
        response = execute_fallback(fallback)
        if user_satisfied(await get_response()): break
```

**Tool Calling (Data Read/Write):**
```python
TOOLS = [
    "get_client_eligibility(client_id, product_type)",
    "get_product_pricing(product, coverage_level, has_spouse)",
    "check_broker_availability(broker_id, date_range)",
    "schedule_appointment(broker_id, client_id, datetime, products)",
    "update_client_notes(client_id, note, category)"
]

# Example: Schedule appointment writes to 3 systems
appointment = await calendly.create_event(broker_id, datetime)
await crm.create_task(broker_id, client_id, datetime)
await db.appointments.insert(appointment_data)
await email.send_confirmation(client_id, appointment)
```

---

## System Architecture Diagram

```
                    ┌─────────────┐
                    │   Client    │
                    │    Phone    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │  Twilio Voice   │
                    │  (WebSocket)    │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │ Load Balancer   │
                    └──────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐         ┌────────▼───────┐
       │ REST API    │         │   WebSocket    │
       │ Gateway     │         │    Gateway     │
       │             │         │                │
       │ - Initiate  │         │ - Audio stream │
       │   calls     │         │ - Real-time    │
       │ - Status    │         │   events       │
       │ - Callbacks │         │ - Monitoring   │
       └──────┬──────┘         └───────┬────────┘
              │                        │
              └───────────┬────────────┘
                          │
                   ┌──────▼───────┐
                   │ Message Queue│
                   │  (SQS/Kafka) │
                   └──────┬───────┘
                          │
         ┌────────────────┴────────────────┐
         │   Kubernetes Worker Pods        │
         │   (Auto-scale: 10-1000+)        │
         │  ┌────────────────────────┐     │
         │  │ Conversation Manager   │     │
         │  │ - State machine        │     │
         │  │ - Tool executor        │     │
         │  │ - Audio processor      │     │
         │  └────────────────────────┘     │
         └────────────────┬────────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
┌─────▼──────┐   ┌────────▼────────┐   ┌─────▼─────┐
│    STT     │   │      LLM        │   │    TTS    │
│ Deepgram   │   │  OpenAI GPT-4o  │   │ ElevenLabs│
│ AssemblyAI │   │  Claude Sonnet  │   │ OpenAI    │
│ Google     │   │  Llama3 (local) │   │ Google    │
└─────┬──────┘   └────────┬────────┘   └─────┬─────┘
      │                   │                   │
      └───────────────────┼───────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
       ┌──────▼──────┐       ┌───────▼────────┐
       │   Cache     │       │   Database     │
       │   (Redis)   │       │  (PostgreSQL)  │
       │ - Sessions  │       │  - Clients     │
       │ - Product   │       │  - Call logs   │
       │   info      │       │  - Appointments│
       └─────────────┘       └───────┬────────┘
                                     │
                            ┌────────┴────────┐
                            │                 │
                     ┌──────▼──────┐   ┌──────▼──────┐
                     │     CRM     │   │  Scheduling │
                     │ (Salesforce)│   │  (Calendly) │
                     └─────────────┘   └─────────────┘

External Dependencies: Twilio, OpenAI, Deepgram, ElevenLabs, 
Salesforce, Calendly, AWS (ECS/RDS/SQS), Pinecone
```

---

## Performance & Cost

**Targets:**
- 99.9% uptime, <500ms latency, 10k concurrent calls/region

**Cost at 100k clients/month:**
- Twilio: $13k | LLM: $8k | STT: $5.6k | TTS: $3.9k | AWS: $12k
- **Total:** $42.5k/month | **Revenue:** $7.5M (15% conversion × $500 commission)
- **ROI:** 17,500%