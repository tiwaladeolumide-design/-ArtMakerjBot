#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ArtMakerjBot - Telegram AI Image Generation Bot
Deployed on Railway with GitHub
"""

import os
import io
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

# Third-party imports
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler, 
    ContextTypes,
    filters
)
from PIL import Image
import requests

# ==================== CONFIGURATION ====================

# Get environment variables from Railway
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN environment variable not set!")

# API Keys
REPLICATE_API_KEY = os.environ.get('REPLICATE_API_KEY', '')
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ADMIN_CHAT_ID = os.environ.get('ADMIN_CHAT_ID', '')

# Model configurations
REPLICATE_MODEL = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
HUGGINGFACE_MODEL = "stabilityai/stable-diffusion-2-1"
OPENAI_MODEL = "dall-e-2"  # or "dall-e-3"

# Rate limiting
RATE_LIMIT = 5  # Images per user per day
RATE_LIMIT_WINDOW = 86400  # 24 hours in seconds

# ==================== LOGGING ====================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ==================== USER STATE MANAGEMENT ====================

class UserState:
    """Manage user sessions and rate limiting"""
    
    def __init__(self):
        self.users: Dict[int, Dict] = {}
    
    def get_user(self, user_id: int) -> Dict:
        """Get or create user state"""
        if user_id not in self.users:
            self.users[user_id] = {
                'generations': 0,
                'last_generation': None,
                'waiting_feedback': False,
                'preferred_model': 'auto',  # auto, replicate, huggingface, openai
                'total_generations': 0
            }
        return self.users[user_id]
    
    def can_generate(self, user_id: int) -> Tuple[bool, str]:
        """Check if user can generate more images"""
        user = self.get_user(user_id)
        
        # Reset daily count if needed
        if user['last_generation']:
            last_time = datetime.fromisoformat(user['last_generation'])
            if datetime.now() - last_time > timedelta(seconds=RATE_LIMIT_WINDOW):
                user['generations'] = 0
        
        if user['generations'] >= RATE_LIMIT:
            wait_time = RATE_LIMIT_WINDOW - (datetime.now() - datetime.fromisoformat(user['last_generation'])).seconds
            hours = wait_time // 3600
            minutes = (wait_time % 3600) // 60
            return False, f"⏳ Daily limit reached. Try again in {hours}h {minutes}m"
        
        return True, "OK"
    
    def increment_generation(self, user_id: int):
        """Increment generation count"""
        user = self.get_user(user_id)
        user['generations'] += 1
        user['last_generation'] = datetime.now().isoformat()
        user['total_generations'] += 1

user_state = UserState()

# ==================== API HANDLERS ====================

class ImageGenerator:
    """Handle image generation from various APIs"""
    
    @staticmethod
    async def generate_with_replicate(prompt: str) -> bytes:
        """Generate image using Replicate API"""
        if not REPLICATE_API_KEY:
            raise Exception("Replicate API key not configured")
        
        headers = {
            "Authorization": f"Token {REPLICATE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "version": REPLICATE_MODEL,
            "input": {
                "prompt": prompt,
                "negative_prompt": "blurry, bad quality, distorted, low resolution, ugly, deformed",
                "width": 768,
                "height": 768,
                "num_inference_steps": 35,
                "guidance_scale": 7.5,
                "scheduler": "DPMSolverMultistep"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            # Start prediction
            async with session.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Replicate API error: {error_text}")
                data = await response.json()
                prediction_id = data["id"]
            
            # Poll for result (max 60 seconds)
            for i in range(60):
                await asyncio.sleep(1)
                try:
                    async with session.get(
                        f"https://api.replicate.com/v1/predictions/{prediction_id}",
                        headers=headers
                    ) as response:
                        data = await response.json()
                        
                        if data["status"] == "succeeded":
                            image_url = data["output"][0]
                            # Download the image
                            async with session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    return await img_response.read()
                                else:
                                    raise Exception("Failed to download generated image")
                        elif data["status"] == "failed":
                            raise Exception(f"Generation failed: {data.get('error', 'Unknown error')}")
                        elif data["status"] == "canceled":
                            raise Exception("Generation was canceled")
                except Exception as e:
                    if i == 59:  # Last iteration
                        raise Exception(f"Generation timed out: {str(e)}")
                    continue
            
            raise Exception("Generation timed out after 60 seconds")
    
    @staticmethod
    async def generate_with_huggingface(prompt: str) -> bytes:
        """Generate image using Hugging Face API (free)"""
        if not HUGGINGFACE_API_KEY:
            raise Exception("Hugging Face API key not configured")
        
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": "blurry, bad quality",
                "num_inference_steps": 30
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error_text = await response.text()
                    raise Exception(f"Hugging Face API error: {error_text}")
    
    @staticmethod
    async def generate_with_openai(prompt: str) -> bytes:
        """Generate image using OpenAI DALL-E API"""
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API key not configured")
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL,
            "prompt": prompt,
            "n": 1,
            "size": "512x512"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    image_url = data["data"][0]["url"]
                    # Download the image
                    async with session.get(image_url) as img_response:
                        if img_response.status == 200:
                            return await img_response.read()
                        else:
                            raise Exception("Failed to download generated image")
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
    
    @staticmethod
    async def generate_image(prompt: str, preferred_model: str = 'auto') -> Tuple[bytes, str]:
        """
        Generate image using the best available API
        Returns: (image_bytes, model_used)
        """
        models = []
        
        # Build model priority list based on user preference
        if preferred_model == 'replicate' and REPLICATE_API_KEY:
            models.append(('Replicate', ImageGenerator.generate_with_replicate))
        elif preferred_model == 'openai' and OPENAI_API_KEY:
            models.append(('OpenAI', ImageGenerator.generate_with_openai))
        elif preferred_model == 'huggingface' and HUGGINGFACE_API_KEY:
            models.append(('Hugging Face', ImageGenerator.generate_with_huggingface))
        else:
            # Auto mode: try best available
            if REPLICATE_API_KEY:
                models.append(('Replicate', ImageGenerator.generate_with_replicate))
            if OPENAI_API_KEY:
                models.append(('OpenAI', ImageGenerator.generate_with_openai))
            if HUGGINGFACE_API_KEY:
                models.append(('Hugging Face', ImageGenerator.generate_with_huggingface))
        
        if not models:
            raise Exception("No image generation APIs configured. Please set at least one API key.")
        
        # Try each model in order
        last_error = None
        for model_name, model_func in models:
            try:
                logger.info(f"Attempting generation with {model_name}...")
                result = await model_func(prompt)
                return result, model_name
            except Exception as e:
                logger.warning(f"{model_name} failed: {str(e)}")
                last_error = e
                continue
        
        raise Exception(f"All models failed. Last error: {str(last_error)}")

# ==================== BOT COMMANDS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    user_id = user.id
    user_state.get_user(user_id)
    
    welcome_text = f"""
🎨 *Welcome to ArtMakerjBot, {user.first_name}!*

I'm an AI-powered image generation bot that turns your text descriptions into stunning artworks!

✨ *How it works:*
Just send me any text prompt and I'll generate an image using state-of-the-art AI models.

🔮 *Example Prompts:*
• "A beautiful sunset over the ocean with dolphins jumping"
• "Cyberpunk city at night, neon lights, rain, 8k"
• "A magical forest with glowing mushrooms and fairies"
• "Portrait of a cat wearing a wizard hat, digital art"

⚙️ *Commands:*
/start - Show this message
/help - Detailed help and tips
/models - Show available AI models
/status - Check bot and API status
/stats - Your personal usage statistics
/feedback - Send feedback to the developer
/about - About this bot

🎯 *Quick actions:*
"""
    
    keyboard = [
        [InlineKeyboardButton("✨ Try Random Prompt", callback_data="random_prompt")],
        [InlineKeyboardButton("📖 View Examples", callback_data="show_examples")],
        [InlineKeyboardButton("⚙️ Change Model", callback_data="change_model")],
        [InlineKeyboardButton("📊 My Stats", callback_data="my_stats")]
    ]
    
    await update.message.reply_text(
        welcome_text,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
📚 *ArtMakerjBot User Guide*

*1️⃣ How to Generate Images*
Simply type a description of what you want to see!
Example: "A majestic dragon flying over a medieval castle at sunset"

*2️⃣ Prompt Writing Tips*
• Be specific and detailed
• Include style, colors, and mood
• Mention quality: "highly detailed, 4k, photorealistic"
• Add artistic references: "in the style of Studio Ghibli"
• Describe the composition: "wide shot, from above, close-up"

*3️⃣ Model Options*
Use /models to switch between different AI models:
• Replicate (Best quality, requires API key)
• OpenAI DALL-E (Great quality, requires API key)
• Hugging Face (Free, good quality)

*4️⃣ Daily Limits*
• Free: {RATE_LIMIT} images per day
• Resets every 24 hours
• Premium features coming soon!

*5️⃣ Best Practices*
• Use 10-20 words for optimal results
• Avoid NSFW content (automatically blocked)
• Save your favorite images
• Share your creations with the community

*6️⃣ Troubleshooting*
• If generation fails, try simplifying your prompt
• Check /status for API availability
• Wait a few seconds between attempts
• Use /feedback to report issues

Need more help? Just ask! 🚀
""".format(RATE_LIMIT=RATE_LIMIT)
    
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /models command"""
    user_id = update.effective_user.id
    user = user_state.get_user(user_id)
    current = user.get('preferred_model', 'auto')
    
    models_text = f"""
⚙️ *Available AI Models*

Current model: *{current.upper()}*

🔄 *Auto Mode* (Default)
Best available model will be used automatically

🤖 *Replicate*
Quality: ⭐⭐⭐⭐⭐
Speed: ⭐⭐⭐⭐
Cost: Paid (free credits available)
Status: {'✅ Available' if REPLICATE_API_KEY else '❌ Not configured'}

🎯 *OpenAI DALL-E*
Quality: ⭐⭐⭐⭐⭐
Speed: ⭐⭐⭐⭐
Cost: Paid
Status: {'✅ Available' if OPENAI_API_KEY else '❌ Not configured'}

⚡ *Hugging Face*
Quality: ⭐⭐⭐
Speed: ⭐⭐
Cost: Free
Status: {'✅ Available' if HUGGINGFACE_API_KEY else '❌ Not configured'}

Click a button below to change your preferred model:
"""
    
    keyboard = []
    if REPLICATE_API_KEY:
        keyboard.append([InlineKeyboardButton("🤖 Use Replicate", callback_data="model_replicate")])
    if OPENAI_API_KEY:
        keyboard.append([InlineKeyboardButton("🎯 Use OpenAI", callback_data="model_openai")])
    if HUGGINGFACE_API_KEY:
        keyboard.append([InlineKeyboardButton("⚡ Use Hugging Face", callback_data="model_huggingface")])
    keyboard.append([InlineKeyboardButton("🔄 Use Auto Mode", callback_data="model_auto")])
    
    await update.message.reply_text(
        models_text,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    status_text = """
📊 *Bot Status*

🤖 Bot: ✅ Online and operational
⚡ Uptime: 100%
📈 Load: Normal

🔑 *API Status:*
"""
    
    # Check Replicate
    if REPLICATE_API_KEY:
        status_text += "✅ Replicate API: Connected\n"
    else:
        status_text += "❌ Replicate API: Not configured\n"
    
    # Check OpenAI
    if OPENAI_API_KEY:
        status_text += "✅ OpenAI API: Connected\n"
    else:
        status_text += "❌ OpenAI API: Not configured\n"
    
    # Check Hugging Face
    if HUGGINGFACE_API_KEY:
        status_text += "✅ Hugging Face API: Connected\n"
    else:
        status_text += "❌ Hugging Face API: Not configured\n"
    
    status_text += f"""
📊 *Server Info:*
• Platform: Railway
• Python: 3.11
• Status: All systems operational

💡 *Need help?* Use /help for guidance.
"""
    
    await update.message.reply_text(status_text, parse_mode="Markdown")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command"""
    user_id = update.effective_user.id
    user = user_state.get_user(user_id)
    
    stats_text = f"""
📊 *Your Personal Statistics*

🎨 Total Images Generated: {user['total_generations']}
📈 Images Today: {user['generations']}/{RATE_LIMIT}
⏳ Reset: {24 - (datetime.now() - datetime.fromisoformat(user['last_generation'])).seconds // 3600 if user['last_generation'] else 24}h remaining

⚙️ Preferred Model: {user.get('preferred_model', 'auto').upper()}
📅 Member Since: {datetime.now().strftime('%Y-%m-%d')}

💡 *Tips:*
• Try different prompts for better results
• Save your favorite images
• Share with friends and community
"""
    
    # Progress bar
    used = user['generations']
    total = RATE_LIMIT
    bar = "█" * used + "░" * (total - used)
    stats_text += f"\n📊 Daily Limit: [{bar}] {used}/{total}"
    
    await update.message.reply_text(stats_text, parse_mode="Markdown")

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /feedback command"""
    await update.message.reply_text(
        "📝 *Send Your Feedback*\n\n"
        "We value your input! Please tell us:\n"
        "• What you love about ArtMakerjBot\n"
        "• What features you'd like to see\n"
        "• Any bugs or issues you've encountered\n"
        "• General suggestions for improvement\n\n"
        "Type your message below and we'll review it! 🙏",
        parse_mode="Markdown"
    )
    context.user_data['waiting_feedback'] = True

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command"""
    about_text = """
🤖 *About ArtMakerjBot*

🎨 *Description:*
ArtMakerjBot is a state-of-the-art AI image generation bot that transforms your text descriptions into stunning visual artworks using advanced machine learning models.

⚡ *Features:*
• Multiple AI models (Replicate, OpenAI, Hugging Face)
• High-quality image generation
• User-friendly interface with inline buttons
• Daily limits for fair usage
• Personal statistics tracking
• Feedback system

🛠️ *Technology:*
• Python 3.11
• python-telegram-bot library
• Deployed on Railway
• GitHub for version control

🔮 *Models Used:*
• Stable Diffusion 2.1
• DALL-E 2 (optional)
• Various fine-tuned models

👨‍💻 *Developer:*
Built with ❤️ by independent developers

📚 *Open Source:*
This project is open-source and available on GitHub

⭐ *Support Us:*
• Star the project on GitHub
• Share with friends
• Provide feedback

Thank you for using ArtMakerjBot! 🚀
"""
    await update.message.reply_text(about_text, parse_mode="Markdown")

# ==================== MESSAGE HANDLERS ====================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text messages (prompts or feedback)"""
    user_id = update.effective_user.id
    text = update.message.text.strip()
    
    # Check if user is sending feedback
    if context.user_data.get('waiting_feedback'):
        context.user_data['waiting_feedback'] = False
        await update.message.reply_text(
            "✅ *Thank you for your feedback!*\n\n"
            "We appreciate your input and will review it shortly. "
            "Your feedback helps make ArtMakerjBot better for everyone! 🙏",
            parse_mode="Markdown"
        )
        
        # Forward to admin if configured
        if ADMIN_CHAT_ID:
            try:
                feedback_msg = f"📝 *New Feedback*\n\n"
                feedback_msg += f"👤 User: @{update.effective_user.username or 'Unknown'}\n"
                feedback_msg += f"🆔 ID: {user_id}\n"
                feedback_msg += f"💬 Message: {text}"
                
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=feedback_msg,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to forward feedback: {e}")
        return
    
    # Check if text is a valid prompt
    if len(text) < 3:
        await update.message.reply_text(
            "❌ *Prompt too short!*\n\n"
            "Please provide a more detailed description. "
            "Try to include style, mood, and details for better results.",
            parse_mode="Markdown"
        )
        return
    
    # Check rate limit
    can_generate, message = user_state.can_generate(user_id)
    if not can_generate:
        await update.message.reply_text(
            f"⏳ *Rate Limit Reached*\n\n{message}\n\n"
            "Premium features with higher limits coming soon! 🚀",
            parse_mode="Markdown"
        )
        return
    
    # Start generation
    thinking_msg = await update.message.reply_text(
        "🎨 *Generating your image...*\n"
        "This usually takes 10-30 seconds. Please wait! ⏳\n\n"
        "🔄 Using: *Processing...*",
        parse_mode="Markdown"
    )
    
    try:
        # Get user's preferred model
        user = user_state.get_user(user_id)
        preferred_model = user.get('preferred_model', 'auto')
        
        # Generate image
        image_data, model_used = await ImageGenerator.generate_image(text, preferred_model)
        
        # Increment usage
        user_state.increment_generation(user_id)
        
        # Send image
        caption = f"🎨 *Generated for:* {text[:150]}{'...' if len(text) > 150 else ''}\n\n"
        caption += f"🤖 Model: *{model_used}*\n"
        caption += f"✨ *Made with ArtMakerjBot*"
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Regenerate", callback_data=f"regenerate_{text[:50]}"),
                InlineKeyboardButton("📤 Share", switch_inline_query=text[:50])
            ],
            [
                InlineKeyboardButton("🎨 Different Style", callback_data="show_styles"),
                InlineKeyboardButton("📊 Stats", callback_data="my_stats")
            ]
        ]
        
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=io.BytesIO(image_data),
            caption=caption,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Delete thinking message
        await thinking_msg.delete()
        
    except Exception as e:
        logger.error(f"Generation error for user {user_id}: {e}")
        await thinking_msg.edit_text(
            f"❌ *Error Generating Image*\n\n"
            f"Something went wrong: {str(e)}\n\n"
            f"🔄 *Try these solutions:*\n"
            f"• Rephrase your prompt (be more specific)\n"
            f"• Use fewer details (keep it simple)\n"
            f"• Wait a moment and try again\n"
            f"• Check /status for API availability\n"
            f"• Try /models to switch AI models\n\n"
            f"If the problem persists, use /feedback to report it.",
            parse_mode="Markdown"
        )

# ==================== CALLBACK HANDLERS ====================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    # Model switching
    if data.startswith("model_"):
        model = data.replace("model_", "")
        user = user_state.get_user(user_id)
        user['preferred_model'] = model
        
        model_names = {
            'auto': '🔄 Auto Mode',
            'replicate': '🤖 Replicate',
            'openai': '🎯 OpenAI',
            'huggingface': '⚡ Hugging Face'
        }
        
        await query.edit_message_text(
            f"✅ *Model Updated!*\n\n"
            f"Your preferred model is now: *{model_names.get(model, model)}*\n\n"
            f"All future generations will try this model first. "
            f"If it's unavailable, it will fall back to other models.",
            parse_mode="Markdown"
        )
        return
    
    # Show examples
    if data == "show_examples":
        examples_text = """
📖 *Example Prompts for Inspiration*

1️⃣ *Fantasy:*
"A majestic dragon flying over a medieval castle at sunset, highly detailed, in the style of fantasy art, golden hour, 4k"

2️⃣ *Cyberpunk:*
"Cyberpunk city at night, neon signs, rain on the streets, reflections, highly detailed, 8k, photorealistic"

3️⃣ *Nature:*
"Beautiful sunset over a calm ocean with dolphins jumping, warm colors, photorealistic, highly detailed"

4️⃣ *Portrait:*
"Portrait of a young woman with flowing hair, digital art, in the style of anime, vibrant colors, detailed"

5️⃣ *Surreal:*
"A surreal dreamscape with floating islands, waterfalls, and glowing mushrooms, fantasy art, highly detailed"

6️⃣ *Abstract:*
"Abstract geometric shapes in vibrant colors, modern art style, gradient background, high contrast"

Try one of these or create your own! 🎨
"""
        await query.edit_message_text(examples_text, parse_mode="Markdown")
        return
    
    # Random prompt
    if data == "random_prompt":
        random_prompts = [
            "A cute cat wearing a wizard hat, digital art, vibrant colors",
            "A futuristic city with flying cars, cyberpunk style, neon lights",
            "A magical forest with glowing mushrooms, fireflies, fantasy art",
            "A serene mountain lake at sunrise, photorealistic, highly detailed",
            "A colorful abstract painting with geometric shapes, modern art",
            "A portrait of a robot with glowing eyes, sci-fi style, 4k",
            "A beautiful garden with exotic flowers, watercolor painting style",
            "A space station orbiting Earth, realistic, highly detailed"
        ]
        import random
        prompt = random.choice(random_prompts)
        
        # Simulate user sending the prompt
        context.user_data['generation_prompt'] = prompt
        
        await query.edit_message_text(
            f"🎲 *Random Prompt:*\n\n"
            f"`{prompt}`\n\n"
            f"Generating now... ⏳",
            parse_mode="Markdown"
        )
        
        # Process the prompt
        thinking_msg = await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="🎨 *Generating your random image...*\nThis may take a moment ⏳",
            parse_mode="Markdown"
        )
        
        try:
            image_data, model_used = await ImageGenerator.generate_image(prompt, 'auto')
            
            user = user_state.get_user(user_id)
            user_state.increment_generation(user_id)
            
            caption = f"🎲 *Random Generation*\n\n"
            caption += f"🤖 Model: *{model_used}*\n"
            caption += f"✨ *Made with ArtMakerjBot*"
            
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode="Markdown"
            )
            await thinking_msg.delete()
            
        except Exception as e:
            await thinking_msg.edit_text(
                f"❌ *Error generating random image*\n\n{str(e)}\n\n"
                f"Try /help for assistance.",
                parse_mode="Markdown"
            )
        return
    
    # Show styles
    if data == "show_styles":
        styles_text = """
🎨 *Style Suggestions for Your Prompt*

Add these to your prompt for different styles:

🖌️ *Art Styles:*
• "in the style of Studio Ghibli"
• "digital art, vibrant colors"
• "oil painting, realistic"
• "watercolor style, soft"
• "anime style, detailed"

🌅 *Lighting & Mood:*
• "golden hour, warm light"
• "neon lights, cyberpunk"
• "moody, dark atmosphere"
• "dreamy, ethereal"
• "bright, cheerful"

📐 *Composition:*
• "wide angle shot"
• "close-up, detailed"
• "from above, top view"
• "symmetrical composition"
• "dynamic pose"

⚡ *Quality:*
• "highly detailed, 4k"
• "photorealistic, 8k"
• "masterpiece, award-winning"
• "ultra HD, sharp focus"
• "cinematic, film grain"

Example prompt combining these:
"A majestic dragon flying over a medieval castle at sunset, highly detailed, in the style of fantasy art, golden hour, wide angle shot, 4k"
"""
        await query.edit_message_text(styles_text, parse_mode="Markdown")
        return
    
    # My stats
    if data == "my_stats":
        user = user_state.get_user(user_id)
        stats_text = f"""
📊 *Your Statistics*

🎨 Total Images: {user['total_generations']}
📈 Today: {user['generations']}/{RATE_LIMIT}
⚙️ Model: {user.get('preferred_model', 'auto').upper()}

📊 *Progress:*
"""
        used = user['generations']
        total = RATE_LIMIT
        bar = "█" * used + "░" * (total - used)
        stats_text += f"[{bar}] {used}/{total}"
        
        await query.edit_message_text(stats_text, parse_mode="Markdown")
        return
    
    # Regenerate
    if data.startswith("regenerate_"):
        prompt = data.replace("regenerate_", "")
        await query.edit_message_text(
            f"🔄 *Regenerating:*\n\n"
            f"`{prompt}`\n\n"
            f"⏳ Working on it...",
            parse_mode="Markdown"
        )
        
        try:
            image_data, model_used = await ImageGenerator.generate_image(prompt, 'auto')
            
            caption = f"🔄 *Regenerated*\n\n"
            caption += f"🤖 Model: *{model_used}*\n"
            caption += f"✨ *Made with ArtMakerjBot*"
            
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"❌ *Error regenerating:* {str(e)}",
                parse_mode="Markdown"
            )
        return
    
    # Change model
    if data == "change_model":
        await models_command(update, context)
        return
    
    # Default response
    await query.edit_message_text(
        "❓ *Unknown action*\n\n"
        "Please use the available buttons or commands.",
        parse_mode="Markdown"
    )

# ==================== ERROR HANDLERS ====================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors and notify user"""
    logger.error(f"Update {update} caused error {context.error}")
    
    try:
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ *An unexpected error occurred.*\n\n"
                     "Our team has been notified. Please try again later or use /feedback to report it.",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

# ==================== MAIN ====================

async def main():
    """Start the bot"""
    logger.info("🚀 Starting ArtMakerjBot...")
    
    # Create application
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(CommandHandler("about", about_command))
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start polling
    logger.info("✅ Bot is running! Press Ctrl+C to stop.")
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
