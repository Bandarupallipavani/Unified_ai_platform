"""
ai/copilot.py — AI Co-Pilot v3
Fixes:
  - History is restored from DB before __init__ (done in main.py)
  - History trimmed to last 20 exchanges to avoid context overflow
  - Domain-aware system prompt
  - Handles empty pipeline_context gracefully
"""
import os
import json
import logging
from anthropic import AsyncAnthropic

logger     = logging.getLogger(__name__)
client     = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
MAX_HISTORY = 40   # max message objects kept in memory (20 exchanges)


DOMAIN_HINTS = {
    "healthcare": "Focus on patient outcomes, clinical metrics (AUC-ROC, sensitivity, specificity), HIPAA data considerations, and avoiding biased predictions in medical contexts.",
    "finance":    "Focus on risk metrics (Gini, KS statistic), regulatory compliance (fair lending), feature importance transparency, and time-series considerations.",
    "hr":         "Focus on fairness/bias in hiring predictions, GDPR for employee data, and interpretable models for HR decision-makers.",
    "retail":     "Focus on customer segmentation, demand forecasting, churn prediction, and recommendation systems.",
    "general":    "Provide balanced, practical ML advice suitable for any domain.",
}

SYSTEM_PROMPT = """You are an expert AI Co-Pilot embedded in a unified ML/DL/NLP platform.
Your role is to guide users through the entire ML lifecycle — from data profiling to deployment.

You have full context of the user's current pipeline state (dataset profile, model type, training
results, best model metrics). Use this context to give precise, actionable advice.

Domain context: {domain_hint}

Guidelines:
- Be concise and practical. Avoid generic advice.
- When a user asks "what should I do next?", look at pipeline_context.status and recommend the specific next step.
- When metrics are available, comment on them specifically (e.g. "Your XGBoost achieved 0.923 AUC — that's strong, but check for overfitting given the gap between train and CV scores").
- Suggest hyperparameter ranges, preprocessing strategies, and feature engineering ideas relevant to the data profile.
- Flag data quality issues (high null rates, class imbalance, feature leakage) proactively.
- For DL/NLP models, advise on architecture selection, learning rate schedules, and fine-tuning strategies.
- Always respond in a friendly, expert tone suitable for both beginners and advanced users.
"""


class CoPilot:
    def __init__(self, session_id: str, mode: str = "beginner", domain: str = "general"):
        self.session_id = session_id
        self.mode       = mode
        self.domain     = domain
        self.history    = []   # populated from DB in main.py before chat()

    async def chat(self, user_message: str, pipeline_context: dict, db=None) -> str:
        domain_hint = DOMAIN_HINTS.get(self.domain, DOMAIN_HINTS["general"])
        system = SYSTEM_PROMPT.format(domain_hint=domain_hint)

        # Inject pipeline context into the system message for this turn
        ctx_block = f"""
Current pipeline state:
- Status: {pipeline_context.get('status', 'unknown')}
- Model type: {pipeline_context.get('model_type', 'ml')}
- User mode: {self.mode}
- Dataset profile summary: {json.dumps(pipeline_context.get('profile', {}), indent=2)[:1000]}
- Best model so far: {json.dumps(pipeline_context.get('best_model', {}), indent=2)}
"""
        full_system = system + "\n" + ctx_block

        # Build message list — trim to last MAX_HISTORY messages to avoid token overflow
        messages = self.history[-MAX_HISTORY:] + [{"role": "user", "content": user_message}]

        try:
            response = await client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                system=full_system,
                messages=messages,
            )
            reply = response.content[0].text
        except Exception as e:
            logger.exception(f"CoPilot API error: {e}")
            reply = (
                "I'm having trouble connecting to the AI service right now. "
                "Please check your API key and try again. "
                f"Error: {type(e).__name__}"
            )

        return reply
