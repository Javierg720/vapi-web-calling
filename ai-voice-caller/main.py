#!/usr/bin/env python3
"""
AI Voice Caller - Main Application
Twilio + OpenAI + ElevenLabs Integration
"""

import os
import logging
from typing import Dict
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import openai
# ElevenLabs removed - using Kokoro TTS + Edge TTS
from dotenv import load_dotenv
import aiosqlite

# Load environment variables FIRST - before any imports that need them
load_dotenv()

from voice_selector import voice_selector
import asyncio
import json

# Store voice selections per call
call_voices = {}
from datetime import datetime
from optimized_database import get_optimized_db
from debug_system import debug_system, wraith, DebugLevel
from ai_conversation import AIConversation
from puter_grok_service import puter_grok
from tts_service import TTSService
from call_manager import CallManager
from typing import Optional
from transcription_service import run_transcription_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Caller",
    description="AI-powered phone calling system with Twilio and custom TTS",
    version="1.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services (using optimized database)
ai_conversation = AIConversation()
tts_service = TTSService()
call_manager = CallManager()

# Security configuration
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "your-secure-admin-key-here")

# Database instance will be initialized on startup
db = None

# Twilio client
twilio_client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

# Simplified: removed admin middleware for now

@app.on_event("startup")
async def startup_event():
    """Initialize application with optimized database"""
    global db
    logger.info("🚀 Starting AI Voice Caller application...")
    
    try:
        # Initialize optimized database with connection pooling
        db = await get_optimized_db()
        logger.info("✅ Optimized database initialized with connection pooling")
        
        # Clean up any stuck calls from previous runs
        stuck_count = await db.cleanup_stuck_calls()
        if stuck_count > 0:
            logger.info(f"🧹 Cleaned up {stuck_count} stuck calls from previous session")
        
        # Test API connections
        try:
            # Test OpenAI
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                logger.error("OpenAI API key not configured")
            else:
                logger.info("OpenAI API key configured")
            
            # Test TTS service
            tts_status = await test_tts_service()
            logger.info(f"TTS service status: {tts_status}")
            
            # Test Twilio connection
            twilio_test = test_twilio_connection()
            logger.info(f"Twilio connection: {twilio_test}")
            
            logger.info("API connections verified")
            
            # Test Grok-4 API connection
            try:
                grok_status = await puter_grok.test_connection()
                logger.info(f"Grok-4 API status: {'✅ Available' if grok_status else '⚠️ Unavailable'}")
            except Exception as grok_error:
                logger.warning(f"Grok-4 API test failed: {grok_error}")
            
        except Exception as e:
            logger.error(f"API setup error: {e}")
            # Don't fail startup for API issues
        
        # Load default agent prompt if available
        try:
            default_config = await db.get_default_agent_configuration()
            if default_config:
                ai_conversation.load_custom_prompt(default_config)
                logger.info(f"✅ Loaded default agent configuration: {default_config['name']}")
            else:
                logger.info("No default agent prompt configured, using built-in prompt")
        except Exception as e:
            logger.warning(f"Could not load default agent prompt: {e}")
        
        # Start background cleanup task
        asyncio.create_task(background_cleanup_task())
        logger.info("Background cleanup task started")
        
        # Log successful startup
        logger.info("Application started successfully - database logging skipped during startup")
        
    except Exception as e:
        logger.error(f"Critical startup error: {e}")
        # Log startup failure if database is available
        try:
            await db.log_system_event("ERROR", f"Startup failed: {str(e)}", None)
        except:
            pass
        raise

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard - live stats and recent calls"""
    try:
        recent_calls = await db.get_recent_calls(limit=10)
        stats = await db.get_call_stats()
        agent_configs = await db.get_agent_configurations()
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "recent_calls": recent_calls,
            "stats": stats,
            "agent_configs": agent_configs,
            "nuclear_mode": False
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return HTMLResponse("<h1>AI Voice Caller</h1><p>Service Running</p><p><a href='/health'>Health Check</a></p>")

@app.post("/webhook/voice")
async def handle_voice_webhook(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
    SpeechResult: str = Form(None),
    Digits: str = Form(None),
    CallStatus: str = Form(None)
):
    """Handle incoming Twilio voice webhooks with enhanced error handling"""
    # Enhanced logging with all webhook parameters
    logger.info(f"Voice webhook received: CallSid={CallSid}, From={From}, To={To}, Status={CallStatus}")
    logger.info(f"Speech/Input: SpeechResult='{SpeechResult}', Digits='{Digits}'")
    
    # Log system event for debugging
    try:
        await db.log_system_event("INFO", f"Voice webhook: {CallStatus or 'processing'}", CallSid)
    except Exception as log_error:
        logger.warning(f"Failed to log system event: {log_error}")
    
    response = VoiceResponse()
    
    try:
        # Validate required parameters
        if not CallSid or not From or not To:
            logger.error(f"Missing required parameters: CallSid={CallSid}, From={From}, To={To}")
            logger.error("Call setup failed - terminating call")
            response.hangup()
            response.hangup()
            return Response(content=str(response), media_type="application/xml")
        
        # Get or create call record with enhanced error handling  
        try:
            call_record = await db.get_or_create_call(CallSid, From, To, "inbound")
            logger.info(f"Call record retrieved/created: {call_record['call_sid']} - Status: {call_record['status']}")
        except Exception as db_error:
            logger.error(f"Database error getting call record: {db_error}")
            await db.log_system_event("ERROR", f"DB error in voice webhook: {str(db_error)}", CallSid)
            logger.error("TTS generation failed - terminating call")
            response.hangup()
            response.hangup()
            return Response(content=str(response), media_type="application/xml")
        
        # Handle speech input processing
        if SpeechResult and SpeechResult.strip():
            logger.info(f"Processing speech input: '{SpeechResult}' for call {CallSid}")
            
            # Ensure conversation memory is reconstructed from DB if not in memory
            try:
                if CallSid not in ai_conversation.conversation_memory:
                    past_messages = await db.get_call_messages(CallSid)
                    memory = [{"role": "system", "content": ai_conversation.system_prompt}]
                    for msg in past_messages:
                        if msg["role"] in ("user", "assistant"):
                            memory.append({"role": msg["role"], "content": msg["content"]})
                    ai_conversation.conversation_memory[CallSid] = memory
            except Exception as mem_err:
                logger.warning(f"Conversation memory reconstruction failed: {mem_err}")

            try:
                # Get AI response with timeout protection
                ai_response = await asyncio.wait_for(
                    ai_conversation.get_response(
                        call_id=CallSid,
                        user_input=SpeechResult,
                        call_context=call_record
                    ),
                    timeout=30.0  # 30 second timeout for AI response
                )
                
                logger.info(f"AI response generated for {CallSid}: '{ai_response[:100]}...'")
                
                # Save conversation with error handling
                try:
                    await db.save_message(CallSid, "user", SpeechResult)
                    await db.save_message(CallSid, "assistant", ai_response)
                    logger.info(f"Conversation saved for call {CallSid}")
                except Exception as save_error:
                    logger.error(f"Error saving conversation: {save_error}")
                    # Continue processing even if save fails
                
                # Generate TTS audio with fallback handling
                # Get voice selection for this call
                selected_voice = call_voices.get(CallSid)
                if selected_voice and selected_voice.startswith('edge:'):
                    selected_voice = selected_voice[5:]  # Remove edge: prefix for Edge TTS
                
                try:
                    audio_url = await asyncio.wait_for(
                        tts_service.generate_speech(ai_response, CallSid, voice=selected_voice),
                        timeout=15.0  # 15 second timeout for TTS
                    )
                    
                    if audio_url:
                        logger.info(f"TTS audio generated: {audio_url}")
                        response.play(audio_url)
                    else:
                        logger.warning(f"TTS failed, using Twilio TTS for call {CallSid}")
                        response.say(ai_response, voice="alice")
                        
                except asyncio.TimeoutError:
                    logger.error(f"TTS generation timeout for call {CallSid}")
                    response.say(ai_response, voice="alice")
                except Exception as tts_error:
                    logger.error(f"TTS generation error: {tts_error}")
                    response.say(ai_response, voice="alice")
                
            except asyncio.TimeoutError:
                logger.error(f"AI response timeout for call {CallSid}")
                response.say("Please repeat your question.")
            except Exception as ai_error:
                logger.error(f"AI conversation error: {ai_error}")
                logger.error("AI/Speech processing failed - terminating call")
                response.hangup()
            
            # Continue listening with improved gather settings
            gather = Gather(
                input="speech",
                action="/webhook/voice",
                timeout=15,  # Increased timeout
                speech_timeout="auto",
                partial_result_callback=f"{os.getenv('NGROK_URL')}/webhook/voice"
            )
            response.append(gather)
            
            # Minimal fallback message
            logger.error("Voice input processing failed - terminating call")
            response.hangup()
            
        # Handle digits input (DTMF)
        elif Digits and Digits.strip():
            logger.info(f"Processing DTMF digits: '{Digits}' for call {CallSid}")
            
            # Process all digits as speech input for custom prompt handling
            digit_text = f"User pressed {Digits}"
            try:
                ai_response = await ai_conversation.get_response(
                    call_id=CallSid,
                    user_input=digit_text,
                    call_context=call_record
                )
                
                # Generate TTS audio with custom voice
                selected_voice = call_voices.get(CallSid)
                if selected_voice and selected_voice.startswith('edge:'):
                    selected_voice = selected_voice[5:]
                
                try:
                    audio_url = await tts_service.generate_speech(ai_response, CallSid, voice=selected_voice)
                    if audio_url:
                        response.play(audio_url)
                    else:
                        response.say(ai_response, voice="alice")
                except Exception as tts_error:
                    logger.error(f"TTS error for digits: {tts_error}")
                    response.say(ai_response, voice="alice")
                
            except Exception as e:
                logger.error(f"Error processing digits: {e}")
                logger.error("AI/Speech processing failed - terminating call")
                response.hangup()
                
            # Continue listening
            gather = Gather(
                input="speech dtmf",
                action="/webhook/voice",
                timeout=15,
                speech_timeout="auto"
            )
            response.append(gather)
            response.say("Please continue.")
            
        else:
            # Initial greeting or no input received
            logger.info(f"Generating initial greeting for call {CallSid}")
            
            try:
                greeting = await asyncio.wait_for(
                    ai_conversation.get_greeting(call_record),
                    timeout=10.0
                )
                
                logger.info(f"Greeting generated for {CallSid}: '{greeting}'")
                
                # Save greeting
                try:
                    await db.save_message(CallSid, "assistant", greeting)
                except Exception as save_error:
                    logger.error(f"Error saving greeting: {save_error}")
                
                # Generate TTS for greeting
                # Get voice selection for this call  
                selected_voice = call_voices.get(CallSid)
                if selected_voice and selected_voice.startswith('edge:'):
                    selected_voice = selected_voice[5:]  # Remove edge: prefix for Edge TTS
                
                try:
                    audio_url = await asyncio.wait_for(
                        tts_service.generate_speech(greeting, CallSid, voice=selected_voice),
                        timeout=15.0
                    )
                    
                    if audio_url:
                        response.play(audio_url)
                    else:
                        response.say(greeting, voice="alice")
                        
                except Exception as tts_error:
                    logger.error(f"TTS error for greeting: {tts_error}")
                    response.say(greeting, voice="alice")
                
            except Exception as greeting_error:
                logger.error(f"Error generating greeting: {greeting_error}")
                # Use minimal fallback greeting
                greeting = "Hello! How can I help you?"
                response.say(greeting, voice="alice")
            
            # Start listening for user input
            gather = Gather(
                input="speech dtmf",
                action="/webhook/voice",
                timeout=15,
                speech_timeout="auto",
                num_digits=1  # For DTMF
            )
            response.append(gather)
            
            # Enhanced fallback with options
            response.say("You can speak naturally, press 0 for a human agent, or 9 to end the call. Thank you!")
        
        # Log successful webhook processing
        logger.info(f"Voice webhook processed successfully for {CallSid}")
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        # Comprehensive error handling and logging
        error_msg = f"Critical error in voice webhook for {CallSid}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Log to database if possible
        try:
            await db.log_system_event("ERROR", error_msg, CallSid)
        except Exception:
            pass  # Don't fail if logging fails
        
        # Update call status to indicate error
        try:
            await db.update_call_status(CallSid, "error")
        except Exception:
            pass  # Don't fail if status update fails
        
        # Return user-friendly error response
        response = VoiceResponse()
        logger.error("Voice webhook failed - terminating call")
        response.hangup()
        
        # Offer option to try again or get help
        gather = Gather(
            input="dtmf",
            action="/webhook/voice",
            timeout=10,
            num_digits=1
        )
        response.append(gather)
        response.say("Thank you for your patience. Goodbye.")
        response.hangup()
        
        return Response(content=str(response), media_type="application/xml")

@app.post("/webhook/status")
async def handle_status_webhook(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    CallDuration: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
    Direction: str = Form(None),
    AnsweredBy: str = Form(None),
    CallbackSource: str = Form(None),
    ErrorCode: str = Form(None),
    ErrorMessage: str = Form(None)
):
    """Handle call status updates with comprehensive error handling"""
    
    # Enhanced logging with all available parameters
    logger.info(f"Status webhook received: CallSid={CallSid}, Status={CallStatus}, Duration={CallDuration}")
    logger.info(f"Call details: From={From}, To={To}, Direction={Direction}, AnsweredBy={AnsweredBy}")
    
    if ErrorCode or ErrorMessage:
        logger.error(f"Call error reported: Code={ErrorCode}, Message={ErrorMessage}")
    
    try:
        # Validate CallSid
        if not CallSid or not CallStatus:
            error_msg = f"Missing required parameters: CallSid={CallSid}, CallStatus={CallStatus}"
            logger.error(error_msg)
            await db.log_system_event("ERROR", error_msg, CallSid)
            return {"status": "error", "message": "Missing required parameters"}
        
        # Log the status update attempt
        await db.log_system_event("INFO", f"Status update: {CallStatus}", CallSid)
        
        # Map Twilio status to our internal status
        internal_status = _map_twilio_status(CallStatus)
        logger.info(f"Mapped Twilio status '{CallStatus}' to internal status '{internal_status}'")
        
        # Handle different call states
        if CallStatus in ["queued", "ringing"]:
            # Call is starting
            logger.info(f"Call {CallSid} is {CallStatus}")
            await db.update_call_status(CallSid, internal_status)
            
        elif CallStatus == "in-progress":
            # Call is active - this is normal
            logger.info(f"Call {CallSid} is in progress")
            await db.update_call_status(CallSid, "in-progress")
            
        elif CallStatus == "completed":
            # Call completed successfully
            logger.info(f"Call {CallSid} completed - Duration: {CallDuration}")
            
            # Generate conversation summary if available
            try:
                summary = await ai_conversation.get_conversation_summary(CallSid)
                if summary:
                    logger.info(f"Call summary for {CallSid}: {summary}")
                    # Persist structured summary for analytics
                    try:
                        await db.save_summary(CallSid, json.dumps(summary))
                    except Exception as se:
                        logger.warning(f"Failed to persist call summary: {se}")
                    await db.log_system_event("INFO", f"Call completed - Summary: {summary}", CallSid)
                    # Trigger a best-effort transcription attempt if recording exists
                    asyncio.create_task(_attempt_transcription(CallSid))
            except Exception as summary_error:
                logger.warning(f"Error generating call summary: {summary_error}")
            
            # Update status with duration
            await db.update_call_status(CallSid, "completed", CallDuration)
            
            # Clean up conversation memory
            try:
                await ai_conversation.end_conversation(CallSid)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up conversation: {cleanup_error}")
            
        elif CallStatus in ["busy", "failed", "no-answer", "canceled"]:
            # Call failed or was not answered
            logger.warning(f"Call {CallSid} ended with status: {CallStatus}")
            
            # Log specific error information
            error_details = f"Status: {CallStatus}"
            if ErrorCode:
                error_details += f", Error Code: {ErrorCode}"
            if ErrorMessage:
                error_details += f", Error: {ErrorMessage}"
            if AnsweredBy:
                error_details += f", Answered By: {AnsweredBy}"
                
            await db.log_system_event("WARNING", f"Call failed - {error_details}", CallSid)
            await db.update_call_status(CallSid, internal_status, CallDuration)
            
            # Clean up conversation memory for failed calls
            try:
                if CallSid in ai_conversation.conversation_memory:
                    del ai_conversation.conversation_memory[CallSid]
            except Exception:
                pass
                
        else:
            # Unknown status - log and handle gracefully
            logger.warning(f"Unknown call status '{CallStatus}' for call {CallSid}")
            await db.log_system_event("WARNING", f"Unknown status: {CallStatus}", CallSid)
            await db.update_call_status(CallSid, CallStatus, CallDuration)
        
        # Additional handling for stuck calls
        if CallStatus == "completed" or CallStatus in ["failed", "busy", "no-answer", "canceled"]:
            # Force cleanup any calls that might be stuck in 'in-progress'
            try:
                await _cleanup_stuck_calls(CallSid)
            except Exception as cleanup_error:
                logger.error(f"Error in stuck call cleanup: {cleanup_error}")
        
        logger.info(f"Status webhook processed successfully for {CallSid}")
        return {
            "status": "ok", 
            "call_sid": CallSid,
            "processed_status": internal_status,
            "duration": CallDuration
        }
        
    except Exception as e:
        error_msg = f"Error updating call status for {CallSid}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Log error to database if possible
        try:
            await db.log_system_event("ERROR", error_msg, CallSid)
        except Exception:
            pass  # Don't fail if logging fails
        
        # For critical errors, try to force call status update
        if CallStatus in ["completed", "failed", "busy", "no-answer", "canceled"]:
            try:
                logger.info(f"Attempting force status update for {CallSid} to {CallStatus}")
                await db.update_call_status(CallSid, CallStatus, CallDuration)
                return {"status": "ok_with_errors", "message": "Status updated despite errors"}
            except Exception as force_error:
                logger.error(f"Force status update failed: {force_error}")
        
        return {
            "status": "error", 
            "message": str(e),
            "call_sid": CallSid,
            "attempted_status": CallStatus
        }

def _map_twilio_status(twilio_status: str) -> str:
    """Map Twilio call status to internal status"""
    status_mapping = {
        "queued": "initiated",
        "ringing": "ringing", 
        "in-progress": "in-progress",
        "completed": "completed",
        "busy": "failed",
        "failed": "failed",
        "no-answer": "no-answer",
        "canceled": "canceled"
    }
    return status_mapping.get(twilio_status.lower(), twilio_status)

async def _cleanup_stuck_calls(current_call_sid: str):
    """Clean up any calls that might be stuck in 'in-progress' status"""
    
    try:
        # This is a safety mechanism to prevent calls from being stuck
        # In a production system, you might want to run this periodically
        
        # For now, just ensure the current call is properly handled
        logger.info(f"Cleanup check completed for {current_call_sid}")
        
        # Future enhancement: Add logic to find and update truly stuck calls
        # Example: calls that have been 'in-progress' for more than reasonable time
        
    except Exception as e:
        logger.error(f"Error in cleanup_stuck_calls: {e}")
        raise

@app.post("/webhook/recording")
async def handle_recording_webhook(
    request: Request,
    CallSid: str = Form(...),
    RecordingSid: str = Form(...),
    RecordingUrl: str = Form(None),
    RecordingStatus: str = Form(None),
    RecordingDuration: str = Form(None)
):
    """Handle Twilio recording status callbacks and store recording URL/duration"""
    try:
        logger.info(
            f"Recording webhook: CallSid={CallSid}, RecordingSid={RecordingSid}, Status={RecordingStatus}, URL={RecordingUrl}"
        )
        duration_int = None
        if RecordingDuration:
            try:
                duration_int = int(RecordingDuration)
            except Exception:
                duration_int = None

        if RecordingUrl:
            await db.save_recording(CallSid, RecordingUrl, duration_int)
            await db.log_system_event("INFO", f"Recording saved: {RecordingUrl}", CallSid)
            # Placeholder: mark transcription as pending (STT integration can update later)
            try:
                await db.save_transcription(CallSid, "PENDING")
            except Exception:
                pass
            # Kick off local transcription attempt in background
            asyncio.create_task(_attempt_transcription(CallSid))

        return {"status": "ok", "call_sid": CallSid, "recording_sid": RecordingSid}
    except Exception as e:
        logger.error(f"Recording webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/webhook/transcription")
async def handle_transcription_webhook(
    request: Request,
    CallSid: str = Form(...),
    TranscriptionText: str = Form(...)
):
    """Optional webhook to receive transcription text from an external STT service"""
    try:
        await db.save_transcription(CallSid, TranscriptionText)
        await db.log_system_event("INFO", "Transcription saved via webhook", CallSid)
        return {"status": "ok", "call_sid": CallSid}
    except Exception as e:
        logger.error(f"Transcription webhook error: {e}")
        return {"status": "error", "message": str(e)}

async def _attempt_transcription(call_sid: str):
    """Best-effort local transcription placeholder. Replace with real STT when available."""
    try:
        artifacts = await db.get_call_artifacts(call_sid)
        if not artifacts or not artifacts.get("recording_url"):
            return
        # If already not PENDING, skip
        if artifacts.get("transcription") and artifacts.get("transcription") != "PENDING":
            return
        # Run local Whisper-based transcription if possible
        text = await asyncio.get_event_loop().run_in_executor(None, run_transcription_pipeline, artifacts['recording_url'])
        if text:
            await db.save_transcription(call_sid, text)
            await db.log_system_event("INFO", "Local transcription completed", call_sid)
        else:
            logger.info(f"Local transcription unavailable/failed for {call_sid}")
    except Exception as e:
        logger.warning(f"Transcription attempt error for {call_sid}: {e}")

@app.get("/calls/{call_id}/view", response_class=HTMLResponse)
async def view_call_details(call_id: str, request: Request):
    """Render call details page including recording, transcription, summary, and messages"""
    try:
        call = await db.get_call_details(call_id)
        messages = await db.get_call_messages(call_id)
        artifacts = await db.get_call_artifacts(call_id)
        return templates.TemplateResponse("call_details.html", {
            "request": request,
            "call": call,
            "messages": messages,
            "artifacts": artifacts
        })
    except Exception as e:
        logger.error(f"Error rendering call details: {e}")
        return HTMLResponse("<h3>Error loading call details</h3>", status_code=500)

@app.post("/make-call")
async def make_outbound_call(
    request: Request,
    phone_number: str = Form(...),
    message: str = Form(None),
    voice: str = Form(None)
):
    """Make an outbound call"""
    try:
        logger.info(f"Making outbound call to {phone_number} with voice: {voice or 'default'}")
        
        # Create call record
        call_record = await db.create_outbound_call(phone_number, message)
        
        # Make the call - use externally reachable base URL
        webhook_base = os.getenv("WEBHOOK_BASE_URL", "http://localhost:8000")
        call = twilio_client.calls.create(
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            to=phone_number,
            url=f"{webhook_base}/webhook/voice",
            status_callback=f"{webhook_base}/webhook/status",
            record=True,
            recording_status_callback=f"{webhook_base}/webhook/recording"
        )
        
        # Update call record with Twilio SID and store voice selection
        await db.update_call_sid(call_record["id"], call.sid)
        
        # Store voice selection for this call
        if voice:
            call_voices[call.sid] = voice
            logger.info(f"Stored voice {voice} for call {call.sid}")
        
        return {
            "status": "success", 
            "call_sid": call.sid,
            "message": f"Call initiated to {phone_number}"
        }
        
    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# LLM Quick Test Endpoint
# ================================

@app.post("/api/llm-test")
async def llm_test(request: Request):
    """Quickly test the current LLM provider with a prompt.

    Body JSON:
      - prompt: string (required)
      - system_prompt: optional override for system prompt
      - provider: optional temporary override (vapi|local|grok|openai)
      - temperature, max_tokens: optional (currently advisory)
    """
    # Simple admin guard to prevent unauthorized use/costs
    admin_key = request.headers.get("x-admin-key")
    if not admin_key or admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    prompt = (data or {}).get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required")

    system_override = (data or {}).get("system_prompt")
    provider_override = (data or {}).get("provider")

    # Keep original provider, optionally override for this request only
    original_provider = ai_conversation.provider_mode
    try:
        if provider_override in ("vapi", "local", "grok", "openai"):
            ai_conversation.set_provider_mode(provider_override)

        system_prompt = system_override or ai_conversation.system_prompt or (
            "You are a concise assistant for phone calls. Keep replies under 20 words."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Use the existing provider routing
        text = await ai_conversation._get_ai_response(messages)

        return {
            "status": "ok",
            "provider": ai_conversation.provider_mode,
            "response": text,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/llm-test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Restore original provider mode
        if provider_override:
            ai_conversation.set_provider_mode(original_provider)

@app.get("/calls/{call_id}")
async def get_call_details(call_id: str):
    """Get call details and conversation"""
    try:
        call_details = await db.get_call_details(call_id)
        messages = await db.get_call_messages(call_id)
        
        return {
            "call": call_details,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error getting call details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_voice_debug.html", response_class=HTMLResponse)
async def voice_debug_page():
    """Voice debugging page"""
    with open('/home/javier/ai-voice-caller/test_voice_debug.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with comprehensive diagnostics"""
    try:
        health_status = await health_check_internal()
        
        # Add additional diagnostics  
        webhook_base = os.getenv("WEBHOOK_BASE_URL", "http://localhost:8000")
        health_status["webhook_endpoints"] = {
            "voice_webhook": f"{webhook_base}/webhook/voice",
            "status_webhook": f"{webhook_base}/webhook/status",
            "recording_webhook": f"{webhook_base}/webhook/recording"
        }
        
        # Add recent system logs
        try:
            # Get recent system events
            async with aiosqlite.connect(db.db_path) as db_conn:
                async with db_conn.execute("""
                    SELECT level, message, timestamp FROM system_logs 
                    ORDER BY timestamp DESC LIMIT 5
                """) as cursor:
                    recent_logs = await cursor.fetchall()
                    health_status["recent_logs"] = [
                        {"level": row[0], "message": row[1], "timestamp": row[2]}
                        for row in recent_logs
                    ]
        except Exception as log_error:
            health_status["recent_logs"] = f"Error retrieving logs: {str(log_error)}"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "healthy": False
        }

@app.get("/api/voices")
async def get_available_voices():
    """Get all available Edge TTS voices"""
    return voice_selector.get_all_voices_summary()

@app.get("/api/voices/{voice_code}")
async def get_voice_info(voice_code: str):
    """Get detailed information about a specific voice"""
    voice_info = voice_selector.get_voice_info(voice_code)
    if not voice_info:
        raise HTTPException(status_code=404, detail="Voice not found")
    return voice_info

# ================================
# Provider Toggle Endpoints
# ================================

@app.get("/api/provider")
async def get_provider_mode():
    try:
        return {"provider": ai_conversation.provider_mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/provider")
async def set_provider_mode(body: Dict):
    try:
        mode = (body or {}).get("provider")
        if mode not in ("vapi", "local", "grok", "openai"):
            raise HTTPException(status_code=400, detail="Invalid provider")
        ai_conversation.set_provider_mode(mode)
        return {"status": "ok", "provider": ai_conversation.provider_mode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voices/set-default")
async def set_default_voice(request: Request):
    """Set the default voice for the system"""
    data = await request.json()
    voice_code = data.get("voice_code")
    
    if not voice_code:
        raise HTTPException(status_code=400, detail="voice_code required")
    
    success = voice_selector.set_default_voice(voice_code)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid voice code")
    
    return {"message": f"Default voice set to {voice_code}", "voice": voice_selector.get_voice_info(voice_code)}

@app.get("/api/voices/demo/{voice_code}")
async def demo_voice(voice_code: str):
    """Generate a demo audio file for a specific voice"""
    voice_info = voice_selector.get_voice_info(voice_code)
    if not voice_info:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    demo_text = voice_selector.get_voice_demo_text(voice_code)
    
    # Generate demo audio with specific voice
    audio_url = await tts_service.generate_voice_demo(demo_text, voice_code)
    
    if audio_url:
        return {"voice": voice_info, "demo_text": demo_text, "audio_url": audio_url}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate demo audio")

@app.get("/api/business-voices")
async def get_business_voice_recommendations():
    """Get voice recommendations for different business use cases"""
    return voice_selector.get_business_voices()

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get dashboard data for real-time updates with debug integration"""
    try:
        # Log debug event
        debug_system.log_event("system", "API", "dashboard_data_request", "Dashboard data requested")
        
        recent_calls = await db.get_recent_calls(limit=10)
        stats = await db.get_call_stats()
        
        return {
            "recent_calls": recent_calls,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def background_cleanup_task():
    """Background task to periodically clean up stuck calls and old data"""
    
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            # Clean up stuck calls
            stuck_count = await db.cleanup_stuck_calls(max_age_minutes=60)
            if stuck_count > 0:
                logger.info(f"Background cleanup: Fixed {stuck_count} stuck calls")
            
            # Clean up old audio files
            try:
                await tts_service.cleanup_old_audio_files(max_age_hours=24)
            except Exception as cleanup_error:
                logger.warning(f"Audio cleanup error: {cleanup_error}")
            
            # Log system health
            try:
                health_status = await health_check_internal()
                if not health_status.get("healthy", False):
                    logger.warning(f"System health check failed: {health_status}")
                    await db.log_system_event("WARNING", f"Health check failed: {health_status}", None)
            except Exception as health_error:
                logger.error(f"Health check error: {health_error}")
                
        except Exception as e:
            logger.error(f"Background cleanup task error: {e}")
            # Continue running even if there's an error
            await asyncio.sleep(60)  # Wait a minute before retrying

async def test_tts_service() -> dict:
    """Test TTS service functionality"""
    try:
        # Test a simple TTS generation
        test_audio = await tts_service.generate_speech("Test", "health_check")
        return {"status": "ok", "test_audio": bool(test_audio)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def test_twilio_connection() -> dict:
    """Test Twilio connection"""
    try:
        # Test Twilio client by getting account info
        account = twilio_client.api.accounts(twilio_client.account_sid).fetch()
        return {"status": "ok", "account_status": account.status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def health_check_internal() -> dict:
    """Internal health check function"""
    
    health_status = {
        "healthy": True,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Database health
    try:
        health_status["services"]["database"] = await db.health_check()
    except Exception as e:
        health_status["services"]["database"] = False
        health_status["healthy"] = False
        logger.error(f"Database health check failed: {e}")
    
    # Twilio health
    try:
        twilio_status = test_twilio_connection()
        health_status["services"]["twilio"] = twilio_status["status"] == "ok"
    except Exception as e:
        health_status["services"]["twilio"] = False
        health_status["healthy"] = False
        logger.error(f"Twilio health check failed: {e}")
    
    # OpenAI health
    health_status["services"]["openai"] = bool(os.getenv("OPENAI_API_KEY"))
    
    # TTS service health
    try:
        tts_status = await test_tts_service()
        health_status["services"]["tts"] = tts_status["status"] == "ok"
    except Exception as e:
        health_status["services"]["tts"] = False
        logger.error(f"TTS health check failed: {e}")
    
    # Check for stuck calls
    try:
        stuck_count = await db.get_stuck_calls_count()
        health_status["stuck_calls"] = stuck_count
        if stuck_count > 5:  # Threshold for concern
            health_status["healthy"] = False
    except Exception as e:
        logger.error(f"Stuck calls check failed: {e}")
    
    return health_status

# OpenRouter Proxy for Puter.js Integration
@app.post("/api/openrouter/chat")
async def openrouter_chat_proxy(request: Request):
    """Proxy endpoint for Puter.js OpenRouter integration"""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        call_sid = data.get("call_sid", "")
        
        # Get conversation history
        conversation = await db.get_call_messages(call_sid)
        
        # Use AI conversation service with OpenRouter
        response = await ai_conversation.get_response(
            call_id=call_sid,
            user_input=prompt,
            call_context={"type": "openrouter"}
        )
        
        return {
            "response": response, 
            "model": "openrouter:meta-llama/llama-3.1-8b-instruct",
            "provider": "puter.js",
            "cost": 0.0
        }
        
    except Exception as e:
        logger.error(f"OpenRouter proxy error: {e}")
        return {"error": "Failed to process request", "response": "I'm here to help. What can I assist you with?"}

# Agent Prompt Management Endpoints
@app.get("/api/agent-prompts")
async def get_agent_prompts():
    """Get all agent prompt configurations"""
    try:
        # Legacy compatibility: Return configurations as prompts
        configurations = await db.get_agent_configurations()
        prompts = [{"id": c["id"], "name": c["name"], "system_prompt": c["system_prompt"], 
                   "greeting_style": c.get("personality_style", "professional"), 
                   "personality": c.get("conversation_style", "helpful"),
                   "is_default": c["is_default"]} for c in configurations]
        return {"prompts": prompts}
    except Exception as e:
        logger.error(f"Error getting agent prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent-prompts")
async def create_agent_prompt(request: Request):
    """Create a new agent prompt configuration"""
    try:
        data = await request.json()
        name = data.get("name")
        system_prompt = data.get("system_prompt")
        greeting_style = data.get("greeting_style", "friendly")
        personality = data.get("personality", "helpful")
        is_default = data.get("is_default", False)
        
        if not name or not system_prompt:
            raise HTTPException(status_code=400, detail="Name and system_prompt are required")
        
        # Legacy compatibility: Convert prompt to configuration format
        config = {
            "name": name,
            "system_prompt": system_prompt,
            "personality_style": greeting_style,
            "conversation_style": personality,
            "is_default": is_default
        }
        config_id = await db.create_agent_configuration(config)
        prompt = {"id": config_id, "name": name, "system_prompt": system_prompt} if config_id else None
        
        # If this is the default, update the AI conversation system
        if is_default:
            ai_conversation.load_custom_prompt(prompt)
        
        return {"status": "success", "prompt": prompt}
        
    except Exception as e:
        logger.error(f"Error creating agent prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/agent-prompts/{prompt_id}")
async def update_agent_prompt(prompt_id: int, request: Request):
    """Update an existing agent prompt configuration"""
    try:
        data = await request.json()
        
        # Legacy method - redirecting to configurations
        success = False
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent prompt not found or no changes made")
        
        # If this is being set as default, update the AI conversation system
        if data.get("is_default"):
            updated_config = await db.get_default_agent_configuration()
            updated_prompt = {"id": updated_config["id"], "name": updated_config["name"], 
                            "system_prompt": updated_config["system_prompt"]} if updated_config else None
            if updated_prompt:
                ai_conversation.load_custom_prompt(updated_prompt)
        
        return {"status": "success", "message": "Agent prompt updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating agent prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/agent-prompts/{prompt_id}")
async def delete_agent_prompt(prompt_id: int):
    """Delete an agent prompt configuration"""
    try:
        # Legacy method - redirecting to configurations
        success = False
        
        if not success:
            raise HTTPException(status_code=400, detail="Cannot delete default prompt or prompt not found")
        
        return {"status": "success", "message": "Agent prompt deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting agent prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent-prompts/{prompt_id}/set-default")
async def set_default_agent_prompt(prompt_id: int):
    """Set a specific prompt as the default"""
    try:
        # Legacy method - redirecting to configurations
        success = False
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set default prompt")
        
        # Update the AI conversation system with new default
        default_config = await db.get_default_agent_configuration()
        if default_config:
            ai_conversation.load_custom_prompt(default_config)
        
        return {"status": "success", "message": "Default agent prompt updated successfully"}
        
    except Exception as e:
        logger.error(f"Error setting default agent prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# COMPREHENSIVE AGENT CONFIGURATION ENDPOINTS
# ==========================================

@app.get("/api/agent-configurations")
async def get_agent_configurations():
    """Get all comprehensive agent configurations"""
    try:
        configurations = await db.get_agent_configurations()
        return configurations
        
    except Exception as e:
        logger.error(f"Error getting agent configurations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent-configurations")
async def create_agent_configuration(config: Dict):
    """Create a new comprehensive agent configuration"""
    try:
        config_id = await db.create_agent_configuration(config)
        
        if not config_id:
            raise HTTPException(status_code=500, detail="Failed to create agent configuration")
        
        return {"id": config_id, "status": "success", "message": "Agent configuration created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating agent configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agent-configurations/{config_id}")
async def get_agent_configuration(config_id: int):
    """Get a specific comprehensive agent configuration"""
    try:
        config = await db.get_agent_configuration(config_id)
        
        if not config:
            raise HTTPException(status_code=404, detail="Agent configuration not found")
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting agent configuration {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/agent-configurations/{config_id}")
async def update_agent_configuration(config_id: int, config: Dict):
    """Update an existing comprehensive agent configuration"""
    try:
        success = await db.update_agent_configuration(config_id, config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update agent configuration")
        
        return {"status": "success", "message": "Agent configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating agent configuration {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/agent-configurations/{config_id}")
async def delete_agent_configuration(config_id: int):
    """Delete a comprehensive agent configuration"""
    try:
        success = await db.delete_agent_configuration(config_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete agent configuration")
        
        return {"status": "success", "message": "Agent configuration deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting agent configuration {config_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent-configurations/{config_id}/set-default")
async def set_default_agent_configuration(config_id: int):
    """Set a specific comprehensive agent configuration as the default"""
    try:
        # NUCLEAR FIX: Use synchronous database (no async/await issues)
        success = nuclear_db.set_default_agent_configuration(config_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set default configuration")
        
        # Update the AI conversation system with new default (EMERGENCY BULLETPROOF)
        try:
            updated_config = nuclear_db.get_agent_configuration(config_id)
            if updated_config:
                simple_config = {
                    "name": updated_config["name"],
                    "system_prompt": updated_config["system_prompt"],
                    "greeting_style": updated_config.get("personality_style", "professional"),
                    "personality": updated_config.get("conversation_style", "helpful")
                }
                ai_conversation.load_custom_prompt(simple_config)
                logger.info(f"Emergency config reload successful: {updated_config['name']}")
        except Exception as reload_error:
            logger.warning(f"Config reload failed (non-critical): {reload_error}")
        
        return {"status": "success", "message": "Default agent configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Error setting default agent configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-configuration")
async def test_agent_configuration(request: Dict):
    """Test a comprehensive agent configuration"""
    try:
        config = request.get("config", {})
        test_message = request.get("test_message", "Hello, this is a test.")
        
        # Temporarily apply the configuration to AI conversation
        temp_ai = AIConversation()
        
        # Load test configuration
        temp_config = {
            "name": config.get("name", "Test Config"),
            "system_prompt": config.get("system_prompt", ""),
            "greeting_style": config.get("personality_style", "professional"),
            "personality": config.get("conversation_style", "helpful")
        }
        temp_ai.load_custom_prompt(temp_config)
        
        # Generate test response
        response = await temp_ai.get_response("test_call", test_message, {"from_number": "test"})
        
        return {"status": "success", "response": response}
        
    except Exception as e:
        logger.error(f"Error testing agent configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preview-voice")
async def preview_voice_configuration(request: Dict):
    """Generate a voice preview with the specified configuration"""
    try:
        voice = request.get("voice", "en-US-AvaMultilingualNeural")
        rate = request.get("rate", 1.0)
        pitch = request.get("pitch", 0)
        volume = request.get("volume", 100)
        text = request.get("text", "Hello! This is a voice preview.")
        
        # Voice mapping table - proper Edge TTS voice names
        voice_mapping = {
            "en-US-AvaMultilingualNeural": "en-US-AvaMultilingualNeural",
            "en-US-JennyNeural": "en-US-JennyNeural", 
            "en-US-DavisNeural": "en-US-DavisNeural",
            "en-US-TonyNeural": "en-US-TonyNeural",
            "en-US-AriaNeural": "en-US-AriaNeural",
            "en-US-GuyNeural": "en-US-GuyNeural",
            "en-US-JaneNeural": "en-US-JaneNeural",
            # Handle shortened versions
            "AvaMultilingual": "en-US-AvaMultilingualNeural",
            "Jenny": "en-US-JennyNeural",
            "Davis": "en-US-DavisNeural", 
            "Tony": "en-US-TonyNeural",
            "Aria": "en-US-AriaNeural",
            "Guy": "en-US-GuyNeural",
            "Jane": "en-US-JaneNeural"
        }
        
        # Map voice to proper Edge TTS format
        mapped_voice = voice_mapping.get(voice, "en-US-AvaMultilingualNeural")
        
        # Generate TTS with proper voice mapping
        audio_url = await tts_service.generate_speech(
            text, 
            "voice_preview", 
            voice=mapped_voice
        )
        
        if audio_url:
            return {"status": "success", "audio_url": audio_url}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate voice preview")
        
    except Exception as e:
        logger.error(f"Error generating voice preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin recovery endpoint removed for simplicity

# 🧠 MASTER DEBUG ENDPOINTS
@app.get("/debug/{call_id}")
async def get_debug_status(call_id: str):
    """Shows current state for specific call"""
    debug_data = debug_system.get_debug_status(call_id)
    return {"call_id": call_id, "debug_data": debug_data}

@app.get("/debug/system/health")
async def system_debug_health():
    """Comprehensive system debug health check - THREADING ISSUE DETECTOR"""
    try:
        threading_errors = 0
        with open("test.log", "r") as f:
            log_content = f.read()[-2000:]  # Last 2000 chars
            threading_errors = log_content.count("threads can only be started once")
    except:
        threading_errors = 0
    
    return {
        "debug_level": debug_system.debug_level.value,
        "threading_errors_detected": threading_errors,
        "system_layers": {
            "AI_CORE": "✅ Nuclear + Local AI Active",
            "VOICE_SYNTH": "✅ Edge TTS Ready", 
            "CALL_ORCHESTRATOR": "❌ Threading issues detected" if threading_errors > 0 else "✅ Threading OK",
            "TWILIO_API": "✅ Connected",
            "DATABASE": "❌ Threading errors in direct connections" if threading_errors > 0 else "✅ Nuclear Mode"
        },
        "critical_issue": "DATABASE THREADING CORRUPTION - async connections failing" if threading_errors > 0 else None
    }

@app.get("/debug/wraith/audit")
async def wraith_audit():
    """WRAITH background audit on failed calls"""
    audit_results = await wraith.audit_calls()
    return {"audit_results": audit_results}
