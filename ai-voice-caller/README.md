# AI Voice Caller 🤖📞

A powerful AI-powered phone calling system built with Twilio, OpenAI, and ElevenLabs. Make and receive intelligent phone calls with custom text-to-speech and conversational AI.

## 🚀 Features

### Core Functionality
- **Inbound Call Handling**: Answer calls with AI assistant
- **Outbound Calling**: Make calls programmatically with AI conversations  
- **Custom TTS**: High-quality voice synthesis (ElevenLabs, Google, Azure)
- **AI Conversations**: Powered by OpenAI GPT-4 with context memory
- **Call Management**: Complete call routing, analytics, and logging

### Advanced Features
- **Web Dashboard**: Real-time call monitoring and management
- **Multiple TTS Providers**: ElevenLabs, Google Cloud TTS, Azure Speech
- **Call Analytics**: Detailed conversation analysis and metrics
- **Business Hours Routing**: Smart call routing based on time/day
- **VIP Caller Support**: Priority handling for important contacts
- **Intent Recognition**: Automatic detection of call purpose and urgency

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Telephony**: Twilio Voice API
- **AI**: OpenAI GPT-4
- **TTS**: ElevenLabs (primary), Google TTS, Azure Speech
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **Frontend**: Bootstrap 5, Chart.js
- **Real-time**: WebSockets for live updates

## 📦 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-voice-caller
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 4. Required Environment Variables
```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# AI Configuration
OPENAI_API_KEY=your_openai_api_key

# TTS Configuration (optional but recommended)
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_preferred_voice_id

# Server Configuration
NGROK_URL=https://your-ngrok-url.ngrok.io
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite:///./calls.db
```

## 🚀 Quick Start

### 1. Start the Application
```bash
python main.py
```

### 2. Set up Ngrok (for development)
```bash
# In a separate terminal
ngrok http 8000
# Copy the HTTPS URL to your .env file as NGROK_URL
```

### 3. Configure Twilio Webhook
1. Go to your [Twilio Console](https://console.twilio.com/)
2. Navigate to Phone Numbers → Manage → Active Numbers
3. Click on your Twilio phone number
4. Set the webhook URL to: `https://your-ngrok-url.ngrok.io/webhook/voice`
5. Set HTTP method to `POST`

### 4. Access Dashboard
Open your browser to `http://localhost:8000` to access the web dashboard.

## 📱 Usage

### Making Outbound Calls
```python
# Via Web Dashboard
# Go to dashboard and use "Make Outbound Call" form

# Via API
curl -X POST "http://localhost:8000/make-call" \
  -F "phone_number=+1234567890" \
  -F "message=Hello, this is an AI assistant calling to..."
```

### Receiving Inbound Calls
Simply call your Twilio phone number! The AI will:
1. Answer with a personalized greeting
2. Listen to the caller's input
3. Generate intelligent responses
4. Handle the conversation naturally
5. Log everything to the database

## 🎛️ Configuration

### AI Conversation Settings
Edit `ai_conversation.py` to customize:
- System prompts and personality
- Response length and style
- Conversation memory limits
- Greeting messages

### TTS Voice Selection
Configure voice options in `tts_service.py`:
- ElevenLabs voice IDs
- Google Cloud TTS voices
- Azure Speech voices
- Fallback to Twilio TTS

### Call Routing Rules
Modify `call_manager.py` for:
- Business hours configuration
- VIP caller lists
- Emergency keyword detection
- Transfer rules and escalation

## 📊 Dashboard Features

### Real-time Monitoring
- Live call status updates
- System health monitoring
- Service status indicators
- Real-time analytics

### Call Management
- View call history and details
- Listen to conversation transcripts
- Analyze call patterns
- Export call data

### Analytics & Insights
- Call volume by hour/day
- Average call duration
- Success/failure rates
- Caller demographics

## 🔧 API Endpoints

### Core Endpoints
- `POST /webhook/voice` - Twilio voice webhook
- `POST /webhook/status` - Call status updates
- `POST /make-call` - Initiate outbound call
- `GET /calls/{call_id}` - Get call details
- `GET /health` - System health check

### Dashboard API
- `GET /` - Main dashboard
- `GET /calls` - Call history page
- `GET /analytics` - Analytics dashboard
- `GET /settings` - Configuration page

### LLM Test Endpoint
- `POST /api/llm-test` - Quick verification of the active LLM provider
  - Headers: `X-Admin-Key: $ADMIN_API_KEY` (required)
  - Body:
    - `prompt` (string, required)
    - `system_prompt` (string, optional)
    - `provider` (string, optional: `vapi|local|grok|openai`)

Examples
```bash
# Local provider (no external calls)
curl -sS -X POST http://localhost:8000/api/llm-test \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -d '{"prompt":"Say pong.","provider":"local"}'

# Vapi provider (requires Vapi config in .env)
curl -sS -X POST http://localhost:8000/api/llm-test \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -d '{"prompt":"Reply with VAPI-OK only.","provider":"vapi"}'
```

## 🛡️ Security Features

- Input validation and sanitization
- Rate limiting on API endpoints
- Secure webhook verification
- Environment-based configuration
- Database query protection

## 📈 Scaling & Production

### Database Upgrade
Replace SQLite with PostgreSQL:
```python
DATABASE_URL=postgresql://user:password@localhost/ai_voice_caller
```

### Load Balancing
Deploy multiple instances behind a load balancer:
```bash
# Start multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Monitoring
- Add application monitoring (New Relic, DataDog)
- Set up logging aggregation
- Configure health checks
- Monitor TTS usage and costs

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Manual Testing
1. Call your Twilio number
2. Test various conversation scenarios
3. Check dashboard for call logs
4. Verify TTS quality and responsiveness

## 🐛 Troubleshooting

### Common Issues

**Calls not connecting:**
- Verify Twilio webhook URL is correct
- Check ngrok is running and URL is up-to-date
- Ensure firewall allows incoming connections

**TTS not working:**
- Check ElevenLabs API key and credits
- Verify voice ID is valid
- Test fallback to Twilio TTS

**AI responses slow:**
- Monitor OpenAI API rate limits
- Check network connectivity
- Consider caching common responses

**Database errors:**
- Ensure SQLite file permissions are correct
- Check disk space availability
- Monitor database connection pool

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Twilio](https://www.twilio.com/) for voice infrastructure
- [OpenAI](https://openai.com/) for conversational AI
- [ElevenLabs](https://elevenlabs.io/) for high-quality TTS
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

For support, email support@yourcompany.com or join our [Discord community](https://discord.gg/yourserver).

---

**Made with ❤️ by [Your Name/Company]**
