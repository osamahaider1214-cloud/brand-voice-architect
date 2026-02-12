import json
import re
from datetime import datetime
from typing import Any, Dict

import streamlit as st
from openai import OpenAI


# -----------------------------
# Helpers
# -----------------------------
def extract_json(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction if the model accidentally wraps output with text.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def call_llm_json(client: OpenAI, model: str, instructions: str, user_input: str) -> Dict[str, Any]:
    """
    Call OpenAI API with proper chat completions endpoint and JSON mode.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return extract_json(content)
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        raise


def pretty_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

# -----------------------------
# Prompts
# -----------------------------
VOICE_CARD_INSTRUCTIONS = """You are a senior brand strategist and creative director.
Return VALID JSON ONLY. No markdown. No extra commentary.

Hard rules:
- Use ONLY facts from the provided product description and user inputs.
- Never invent numbers, claims, certifications, pricing, partnerships, or availability.
- If a key detail is missing, use a placeholder like: [INSERT PRICE], [INSERT PROOF], [INSERT DATE].
- You MUST return valid JSON that matches the requested schema.
"""

VOICE_CARD_USER_TEMPLATE = """Create a Brand Voice Card for the following.

Brand name: {brand_name}
Target audience: {audience}
Primary objective: {objective}
Product description (source of truth): {product_description}

Output JSON schema:
{{
  "brand": {{
    "name": "...",
    "positioning_one_liner": "...",
    "audience": "...",
    "objective": "..."
  }},
  "voice": {{
    "tone_traits": ["...", "...", "..."],
    "formality": "low|medium|high",
    "sentence_style": "short|mixed|long",
    "humor_level": "none|light|playful",
    "emoji_policy": "none|sparingly|allowed",
    "pov": "we|you|third_person"
  }},
  "lexicon": {{
    "use": ["...", "..."],
    "avoid": ["...", "..."]
  }},
  "style_rules": [
    "..."
  ],
  "compliance_guardrails": [
    "..."
  ]
}}
"""

ASSETS_INSTRUCTIONS = """You are a performance marketer and brand copy lead.
Return VALID JSON ONLY. No markdown. No extra commentary.

Hard rules:
- Use ONLY facts from the provided product description + voice card.
- Do not invent metrics, awards, logos, customer names, compliance claims, or prices.
- If needed, use placeholders like [INSERT PRICE], [INSERT PROOF], [INSERT LINK].
- Keep voice consistent with the Voice Card.
- You MUST return valid JSON that matches the requested schema.
"""

ASSETS_USER_TEMPLATE = """Generate a multi-channel marketing plan and copy.

Inputs:
Product description (source of truth): {product_description}

Brand Voice Card (must follow):
{voice_card_json}

Output JSON schema:
{{
  "campaign_core": {{
    "big_idea": "...",
    "key_messages": ["...", "...", "..."],
    "primary_cta": "..."
  }},
  "email_sequence": {{
    "email_1": {{
      "goal": "...",
      "subject": "...",
      "preheader": "...",
      "body": "...",
      "cta": "..."
    }},
    "email_2": {{
      "goal": "...",
      "subject": "...",
      "preheader": "...",
      "body": "...",
      "cta": "..."
    }},
    "email_3": {{
      "goal": "...",
      "subject": "...",
      "preheader": "...",
      "body": "...",
      "cta": "..."
    }}
  }},
  "social": {{
    "linkedin": [
      {{
        "post": "...",
        "creative_direction": "...",
        "hashtags": ["...","..."]
      }}
    ],
    "instagram": [
      {{
        "caption": "...",
        "reel_script": "...",
        "on_screen_text": ["...","..."],
        "hashtags": ["...","..."]
      }}
    ],
    "x": [
      {{
        "post": "..."
      }}
    ]
  }},
  "web_landing_page": {{
    "hero_headline": "...",
    "hero_subhead": "...",
    "sections": [
      {{
        "title": "...",
        "copy": "..."
      }}
    ],
    "faq": [
      {{
        "q": "...",
        "a": "..."
      }}
    ],
    "meta_title": "...",
    "meta_description": "..."
  }}
}}
"""

AUDIT_INSTRUCTIONS = """You are a meticulous brand QA reviewer.
Return VALID JSON ONLY. No markdown. No extra commentary.

Task:
Score each asset for voice consistency against the Voice Card and message consistency across channels.

Scoring:
- score from 1 to 5 (5 = perfect alignment)
- Provide a short why + an actionable fix suggestion.
- Do NOT invent product facts in fixes. Use placeholders if needed.
- You MUST return valid JSON that matches the requested schema.
"""

AUDIT_USER_TEMPLATE = """Audit the following assets.

Voice Card:
{voice_card_json}

Assets:
{assets_json}

Return JSON schema:
{{
  "overall": {{
    "average_score": 0,
    "top_drift_themes": ["...","..."],
    "global_fixes": ["...","..."]
  }},
  "items": [
    {{
      "asset_id": "email_sequence.email_1.body",
      "channel": "email",
      "score": 1,
      "why": "...",
      "fix_suggestion": "..."
    }}
  ]
}}
"""


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Brand-Voice Campaign Architect", layout="wide")
st.title("🎨 Brand-Voice Campaign Architect")

with st.sidebar:
    st.header("Inputs")
    brand_name = st.text_input("Brand name", "Acme")
    audience = st.text_input("Target audience", "Busy professionals who value quality and convenience")
    objective = st.text_input("Primary objective", "Drive qualified leads and trials")
    model = st.text_input("Model", "gpt-4o")

    st.caption("Tip: store OPENAI_API_KEY in Streamlit secrets for deployment.")
    st.divider()

product_description = st.text_area(
    "Product description (single source of truth)",
    height=180,
    value=(
        "Example: A lightweight, reusable smart bottle with UV self-cleaning cap. "
        "Keeps water fresh, tracks hydration via app, USB-C charging, BPA-free. "
        "Targeted at commuters and gym-goers. No pricing provided."
    ),
)

# OpenAI client
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif "OPENAI_API_KEY" in st.session_state:
    api_key = st.session_state["OPENAI_API_KEY"]

if not api_key:
    api_key = st.text_input("OPENAI_API_KEY (local only)", type="password")
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key

client = OpenAI(api_key=api_key) if api_key else None

col_a, col_b = st.columns([1, 1])

with col_a:
    generate = st.button("🚀 Generate Voice Card + Assets", type="primary", disabled=not bool(client))
with col_b:
    run_audit = st.button("🔍 Run Consistency Audit", disabled=not bool(client))

if generate and client:
    try:
        with st.spinner("Generating Brand Voice Card..."):
            voice_card = call_llm_json(
                client=client,
                model=model,
                instructions=VOICE_CARD_INSTRUCTIONS,
                user_input=VOICE_CARD_USER_TEMPLATE.format(
                    brand_name=brand_name,
                    audience=audience,
                    objective=objective,
                    product_description=product_description,
                ),
            )
        st.session_state["voice_card"] = voice_card
        st.success("✅ Voice Card generated!")

        with st.spinner("Generating multi-channel assets..."):
            assets = call_llm_json(
                client=client,
                model=model,
                instructions=ASSETS_INSTRUCTIONS,
                user_input=ASSETS_USER_TEMPLATE.format(
                    product_description=product_description,
                    voice_card_json=pretty_json(voice_card),
                ),
            )
        st.session_state["assets"] = assets
        st.success("✅ Assets generated!")
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

if run_audit and client:
    if "voice_card" not in st.session_state or "assets" not in st.session_state:
        st.warning("⚠️ Generate the Voice Card + Assets first.")
    else:
        try:
            with st.spinner("Auditing consistency..."):
                audit = call_llm_json(
                    client=client,
                    model=model,
                    instructions=AUDIT_INSTRUCTIONS,
                    user_input=AUDIT_USER_TEMPLATE.format(
                        voice_card_json=pretty_json(st.session_state["voice_card"]),
                        assets_json=pretty_json(st.session_state["assets"]),
                    ),
                )
            st.session_state["audit"] = audit
            st.success("✅ Audit complete!")
        except Exception as e:
            st.error(f"Audit failed: {str(e)}")

# Display outputs
tabs = st.tabs(["📋 Brand Voice Card", "📝 Assets", "🔍 Consistency Audit", "📥 Export"])

with tabs[0]:
    st.subheader("Brand Voice Card")
    if "voice_card" in st.session_state:
        st.code(pretty_json(st.session_state["voice_card"]), language="json")
    else:
        st.info("👆 Click **Generate Voice Card + Assets** to begin.")

with tabs[1]:
    st.subheader("Multi-channel Assets")
    if "assets" in st.session_state:
        st.code(pretty_json(st.session_state["assets"]), language="json")
    else:
        st.info("👆 Click **Generate Voice Card + Assets** to begin.")

with tabs[2]:
    st.subheader("Voice Consistency Audit")
    if "audit" in st.session_state:
        audit_data = st.session_state["audit"]
        
        # Show overall score
        if "overall" in audit_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Score", f"{audit_data['overall'].get('average_score', 0)}/5")
            with col2:
                st.write("**Top Drift Themes:**")
                for theme in audit_data['overall'].get('top_drift_themes', []):
                    st.write(f"- {theme}")
        
st.divider()
        st.code(pretty_json(audit_data), language="json")
    else:
        st.info("👆 Click **Run Consistency Audit** after generation.")

with tabs[3]:
    st.subheader("Export Campaign")
    
    if "voice_card" in st.session_state and "assets" in st.session_state:
        # Build complete markdown export
        export_md = f"""# Brand Voice Card\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n## Brand Information\n- **Name:** {st.session_state["voice_card"].get("brand", {}).get("name", "N/A")}\n- **Audience:** {st.session_state["voice_card"].get("brand", {}).get("audience", "N/A")}\n- **Objective:** {st.session_state["voice_card"].get("brand", {}).get("objective", "N/A")}\n\n## Voice Card (JSON)\n```json\n{pretty_json(st.session_state["voice_card"])}\n```\n\n---\n\n# Multi-Channel Assets\n\n## Campaign Core\n```json\n{pretty_json(st.session_state["assets"].get("campaign_core", {}))}\n```\n\n## Email Sequence\n```json\n{pretty_json(st.session_state["assets"].get("email_sequence", {}))}\n```\n\n## Social Media\n```json\n{pretty_json(st.session_state["assets"].get("social", {}))}\n```\n\n## Landing Page\n```json\n{pretty_json(st.session_state["assets"].get("web_landing_page", {}))}\n```\n"""

        if "audit" in st.session_state:
            export_md += f"""\n---\n\n# Consistency Audit\n```json\n{pretty_json(st.session_state["audit"])}\n```\n"""

        # Display preview
        st.markdown("### Preview")
        with st.expander("Show markdown preview"):
            st.code(export_md, language="markdown")
        
        # Download button
        st.download_button(
            label="📥 Download Campaign (.md)",
            data=export_md,
            file_name=f"brand_campaign_{brand_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
    else:
        st.info("👆 Generate Voice Card + Assets first to enable export.")

# Footer
st.divider()
st.caption("Built with Streamlit + OpenAI | Ensures brand consistency across all marketing channels")
