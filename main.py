import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

# Database helpers
from database import db, create_document, get_documents

app = FastAPI(title="AI Business Assistant Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic Models (Schemas)
# -----------------------------
class Contact(BaseModel):
    name: Optional[str] = Field(None)
    email: Optional[EmailStr] = Field(None)
    phone: Optional[str] = Field(None)

class BusinessInput(BaseModel):
    business_name: str = Field(..., description="Company name")
    industry: str = Field(...)
    services: List[str] = Field(..., description="List of services or products")
    location: str = Field(...)
    target_audience: str = Field(...)
    goals: List[str] = Field(default_factory=list)
    tone: str = Field("professional")
    brand_colors: List[str] = Field(default_factory=lambda: ["#6d28d9", "#0ea5e9"])  # hex colors
    brand_voice: Optional[str] = Field(None, description="Extra brand voice notes")
    faqs: List[Dict[str, str]] = Field(default_factory=list, description="[{question, answer}]")
    examples: List[str] = Field(default_factory=list, description="Example customer interactions or phrases")
    subscription_tier: str = Field(..., description="starter | standard | premium")
    website_url: Optional[str] = None
    contact: Optional[Contact] = None

class GenerationResult(BaseModel):
    business_summary: str
    brand_identity: Dict[str, Any]
    chatbot_persona: Dict[str, Any]
    website_structure: Dict[str, Any]
    social_media_plan: Dict[str, Any]
    booking_tools: Dict[str, Any]
    sales_and_ads: Dict[str, Any]
    sops: List[str]
    automations: Dict[str, Any]
    dashboard: Dict[str, Any]
    subscription_tier: str
    created_at: datetime


# -----------------------------
# Helper generation utilities
# -----------------------------

def sentence_list(items: List[str]) -> str:
    return ", ".join(items)


def make_business_summary(data: BusinessInput) -> str:
    return (
        f"{data.business_name} is a {data.tone} {data.industry} brand based in {data.location}. "
        f"They offer {sentence_list(data.services)} to {data.target_audience}. "
        f"Primary goals: {sentence_list(data.goals) or 'brand growth and customer satisfaction'}."
    )


def make_brand_identity(data: BusinessInput) -> Dict[str, Any]:
    return {
        "colors": data.brand_colors,
        "typography": ["Inter", "Manrope", "Geist"],
        "voice": data.brand_voice or f"{data.tone}, clear, helpful, conversion-focused",
        "keywords": [data.industry, data.location, *data.services],
        "value_props": [
            "Fast response times",
            "Clear pricing",
            "Expert support",
        ],
    }


def tier_includes(feature: str, tier: str) -> bool:
    order = {"starter": 0, "standard": 1, "premium": 2}
    # map features to min tier
    gates = {
        "basic": 0,
        "website": 1,
        "ads": 1,
        "booking": 1,
        "caller_bot": 2,
        "crm": 2,
        "full_builder": 2,
        "voice_support": 2,
        "analytics_full": 2,
    }
    return order.get(tier, 0) >= gates.get(feature, 0)


def make_chatbot_persona(data: BusinessInput) -> Dict[str, Any]:
    system_prompt = f"You are {data.business_name}'s AI assistant. Tone: {data.tone}. Industry: {data.industry}. Target audience: {data.target_audience}."  # noqa
    knowledge = {
        "business_description": make_business_summary(data),
        "services": data.services,
        "policies": ["Be polite", "Never promise unavailable offers", "Escalate complex issues"],
        "goals": data.goals,
        "faqs": data.faqs,
        "examples": data.examples,
        "capabilities": [
            "Customer service",
            "Sales qualification",
            "Lead capture",
            "FAQ handling",
            "Product explanation",
        ],
    }
    outputs = {
        "greeting": f"Hi! You\'re speaking with the {data.business_name} AI assistant — how can I help today?",
        "lead_form_fields": ["name", "email", "phone", "need"],
    }
    return {"system_prompt": system_prompt, "knowledge": knowledge, "outputs": outputs}


def make_website_structure(data: BusinessInput, tier: str) -> Dict[str, Any]:
    base = {
        "pages": [
            {"path": "/", "title": "Home"},
            {"path": "/about", "title": "About"},
            {"path": "/services", "title": "Services", "items": data.services},
            {"path": "/pricing", "title": "Pricing"},
            {"path": "/contact", "title": "Contact"},
        ],
        "seo": {
            "title": f"{data.business_name} | {data.industry} in {data.location}",
            "description": f"{data.business_name} offers {sentence_list(data.services)} in {data.location}.",
            "keywords": [data.business_name, data.industry, data.location, *data.services],
        },
        "theme": {"colors": data.brand_colors, "tone": data.tone},
    }
    if tier_includes("website", tier):
        base["integrations"] = {"chat_widget": True, "booking": True}
    return base


def make_social_plan(data: BusinessInput, tier: str) -> Dict[str, Any]:
    base_calendar = [
        {"day": i + 1, "theme": theme}
        for i, theme in enumerate([
            "Brand story",
            "How it works",
            "Customer testimonial",
            "Service spotlight",
            "Behind the scenes",
            "FAQ of the week",
            "Offer/CTA",
        ] * 4)
    ][:30]

    base = {
        "calendar_30_day": base_calendar,
        "captions_style": f"{data.tone} with clear CTAs",
        "hashtags": [f"#{data.industry.replace(' ', '')}", f"#{data.location.replace(' ', '')}", "#SmallBusiness", "#Tips"],
    }
    if tier_includes("ads", tier):
        base["ad_angles"] = [
            "Pain-Agitate-Solve for top problem",
            "Time-saving benefit angle",
            "Local authority angle",
        ]
        base["platforms"] = ["Instagram", "Facebook", "Google", "TikTok"]
    else:
        base["platforms"] = ["Instagram", "Facebook"]
    return base


def make_booking_tools(data: BusinessInput, tier: str) -> Dict[str, Any]:
    base = {
        "booking_link": f"https://book.{data.business_name.lower().replace(' ', '')}.ai",
        "reminders": ["email"],
        "confirmation_messages": {
            "email": "Thanks for booking with us!",
        },
    }
    if tier_includes("booking", tier):
        base["reminders"] = ["email", "sms", "whatsapp"]
        base["calendar_sync"] = True
    return base


def make_sales_ads(data: BusinessInput, tier: str) -> Dict[str, Any]:
    base = {
        "funnels": [
            {
                "name": "Lead magnet → Nurture → Consult",
                "stages": ["Offer", "Capture", "Email nurture", "Consult call"],
            }
        ],
        "ad_ideas": [
            {
                "platform": "Facebook",
                "headline": f"{data.business_name}: {data.services[0]} in {data.location}",
                "copy": f"Tired of guessing? Try our {data.services[0]} — trusted by locals.",
            }
        ],
    }
    if tier_includes("ads", tier):
        base["generators"] = ["FB/IG primary text", "Google RSA headlines", "TikTok hooks"]
    return base


def make_sops(data: BusinessInput, tier: str) -> List[str]:
    sops = [
        "Inbound chat triage and escalation",
        "Lead capture and CRM entry",
        "Daily social posting routine",
    ]
    if tier_includes("website", tier):
        sops.append("Appointment scheduling and no-show follow-up")
    if tier_includes("caller_bot", tier):
        sops.append("Automated call reminders and voicemail drop")
    return sops


def make_automations(data: BusinessInput, tier: str) -> Dict[str, Any]:
    base = {
        "triggers": [
            "New website chat",
            "Form submission",
            "New booking",
        ],
        "actions": [
            "Send confirmation email",
            "Create CRM contact",
            "Notify team Slack",
        ],
    }
    if tier_includes("crm", tier):
        base["integrations"] = ["CRM", "WhatsApp", "Facebook", "Instagram", "Calendar"]
    else:
        base["integrations"] = ["Email", "Calendar"]
    return base


def make_dashboard(data: BusinessInput, tier: str) -> Dict[str, Any]:
    roles = [
        {"name": "AI Receptionist", "status": "ready"},
        {"name": "AI Support Agent", "status": "ready"},
        {"name": "AI Content Creator", "status": "ready"},
    ]
    if tier_includes("ads", tier):
        roles.append({"name": "AI Ad Expert", "status": "ready"})
        roles.append({"name": "AI Social Media Manager", "status": "ready"})
    if tier_includes("voice_support", tier):
        roles.extend([
            {"name": "AI Sales Agent", "status": "ready"},
            {"name": "AI Booking & Scheduling Assistant", "status": "ready"},
            {"name": "AI Financial Assistant", "status": "ready"},
            {"name": "24/7 Multi-platform Assistant", "status": "ready"},
        ])
    analytics = {
        "level": "basic" if not tier_includes("analytics_full", tier) else "full",
        "widgets": ["Leads", "Bookings", "Messages"],
    }
    return {"roles": roles, "analytics": analytics}


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "AI Business Assistant Generator Backend"}


@app.post("/generate", response_model=GenerationResult)
def generate_assistant(data: BusinessInput):
    tier = data.subscription_tier.lower()
    if tier not in {"starter", "standard", "premium"}:
        raise HTTPException(status_code=400, detail="subscription_tier must be one of: starter, standard, premium")

    result = GenerationResult(
        business_summary=make_business_summary(data),
        brand_identity=make_brand_identity(data),
        chatbot_persona=make_chatbot_persona(data),
        website_structure=make_website_structure(data, tier),
        social_media_plan=make_social_plan(data, tier),
        booking_tools=make_booking_tools(data, tier),
        sales_and_ads=make_sales_ads(data, tier),
        sops=make_sops(data, tier),
        automations=make_automations(data, tier),
        dashboard=make_dashboard(data, tier),
        subscription_tier=tier,
        created_at=datetime.utcnow(),
    )

    # Persist both input and output
    try:
        create_document("businessinput", data.model_dump())
        create_document("generationresult", result.model_dump())
    except Exception as e:
        # Don't fail generation if DB not available
        print("DB save error:", e)

    return result


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
