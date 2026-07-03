import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from openai import OpenAI
from backend.app.core.config import settings

logger = logging.getLogger("phishing_platform")

SYSTEM_PROMPT = """You are an Enterprise Cybersecurity Phishing Analyst. 
You will receive structured findings from a phishing analysis engine. 
Your job is to explain, summarize, and generate a security intelligence report. 

You MUST return your output in strict JSON format. Do NOT write any conversational text, markdown formatting blocks (like ```json), or explanations outside of the JSON structure.

The output JSON MUST follow this exact schema:
{
  "threat_type": "The classified threat type, e.g., 'Credential Harvesting', 'Brand Impersonation', 'Financial Wire Fraud', 'Invoice Scam', 'Malware/Ransomware Distribution', 'Safe Email'",
  "severity": "Low, Medium, High, or Critical",
  "summary": "A technical description of why this email is flagged, referencing specific rules triggered and technical findings.",
  "indicators": ["A list of indicator strings, e.g. domain names, emails, attachment names, headers, urgent phrases"],
  "recommendations": "Actionable security advice for the recipient and corporate security teams.",
  "executive_summary": "A 1-2 sentence non-technical summary explaining the threat to a business executive."
}
"""

class LLMAnalyst:
    def __init__(self):
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("LLM Analyst initialized using OpenAI Client.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

    async def analyze_threat(self, 
                             verdict: str, 
                             risk_score: float, 
                             rules: List[Dict[str, Any]], 
                             urls: List[Dict[str, Any]], 
                             headers: Dict[str, Any], 
                             email_text: str) -> Dict[str, Any]:
        """Queries local Ollama or hosted OpenAI API to construct a threat report."""
        
        # Prepare content payload
        payload = {
            "prediction_verdict": verdict,
            "composite_risk_score": risk_score,
            "rules_triggered": [
                {"name": r.get("rule_name"), "severity": r.get("severity"), "reason": r.get("reason")}
                for r in rules
            ],
            "urls_analyzed": [
                {"url": u.get("url"), "flags": u.get("flags"), "is_suspicious": u.get("is_suspicious")}
                for u in urls
            ],
            "headers_analysis": {
                "findings": headers.get("findings", []),
                "spf_pass": headers.get("spf_pass"),
                "dkim_pass": headers.get("dkim_pass"),
                "dmarc_pass": headers.get("dmarc_pass")
            },
            "email_body_snippet": email_text[:500] + ("..." if len(email_text) > 500 else "")
        }

        user_content = f"Please analyze this security scan data and generate the JSON report:\n{json.dumps(payload, indent=2)}"

        # 1. Try OpenAI if API Key exists
        if self.openai_client:
            try:
                logger.info("Sending request to OpenAI API...")
                response = self.openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2
                )
                raw_result = response.choices[0].message.content
                return self._parse_json_safely(raw_result, payload)
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}. Falling back to Ollama...")

        # 2. Try Ollama (Local LLM)
        try:
            logger.info(f"Sending request to local Ollama at {settings.OLLAMA_HOST} using {settings.OLLAMA_MODEL}...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_HOST}/api/chat",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.2}
                    }
                )
                if response.status_code == 200:
                    raw_result = response.json()["message"]["content"]
                    return self._parse_json_safely(raw_result, payload)
                else:
                    logger.warning(f"Ollama returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")

        # 3. Deterministic Static Fallback (If both APIs are down/fail)
        logger.warning("All LLM connections failed. Generating deterministic fallback report.")
        return self._generate_fallback_report(payload)

    def _parse_json_safely(self, text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans and parses the LLM output safely."""
        text = text.strip()
        # Strip markdown syntax if the model ignored our system prompt instructions
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
        
        try:
            parsed = json.loads(text)
            # Ensure all required keys exist
            required_keys = ["threat_type", "severity", "summary", "indicators", "recommendations", "executive_summary"]
            for key in required_keys:
                if key not in parsed:
                    parsed[key] = "Not provided by model"
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}. Raw response: {text}")
            return self._generate_fallback_report(payload)

    def _generate_fallback_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Produces a clean threat intelligence summary when LLMs are offline."""
        risk_score = payload["composite_risk_score"]
        verdict = payload["prediction_verdict"]
        rules = payload["rules_triggered"]
        urls = payload["urls_analyzed"]
        headers = payload["headers_analysis"]
        
        threat_type = "Safe Email"
        severity = "Low"
        
        if risk_score >= 70.0:
            severity = "High"
            threat_type = "Potential Phishing Scam"
            if any("harvest" in r["name"].lower() for r in rules):
                threat_type = "Credential Harvesting"
            elif any("imperson" in r["name"].lower() for r in rules):
                threat_type = "Brand Impersonation"
            elif any("finance" in r["name"].lower() for r in rules):
                threat_type = "Financial Wire Fraud"
        elif risk_score >= 40.0:
            severity = "Medium"
            threat_type = "Suspicious Message"

        indicators = []
        for r in rules:
            indicators.append(f"Rule: {r['name']}")
        for u in urls:
            if u["is_suspicious"]:
                indicators.append(f"URL: {u['url']}")
        for f in headers["findings"]:
            indicators.append(f"Header: {f}")

        summary = f"Automated scan triggered a composite risk score of {risk_score}%. "
        if rules:
            summary += f"The rules engine detected {len(rules)} phishing pattern(s): {', '.join([r['name'] for r in rules])}. "
        if urls:
            susp_count = sum(1 for u in urls if u["is_suspicious"])
            summary += f"Found {len(urls)} URLs in the body, with {susp_count} flagged as suspicious. "
        if headers["findings"]:
            summary += f"Email header verification raised discrepancies: {', '.join(headers['findings'])}."
        if not rules and not any(u["is_suspicious"] for u in urls) and not headers["findings"]:
            summary += "No phishing rules, suspicious URLs, or header anomalies were detected."

        recommendations = "No immediate phishing action is required based on the available indicators. "
        if severity in ["High", "Critical"]:
            recommendations = "Verify sender domain legitimacy before responding. Do not click links or submit login credentials. "
            recommendations += "Forward this email to your organization's security operations center (SOC) immediately."
        elif severity == "Medium":
            recommendations = "Verify sender domain legitimacy before responding. Do not click links or submit login credentials. "
            recommendations += "Exercise caution when interacting with links and attachments."
        else:
            recommendations += "Continue normal caution for unknown senders."

        if "safe" in verdict.lower():
            exec_summary = f"This message is classified as SAFE with a low risk score of {risk_score}%. No phishing indicators were detected."
        else:
            exec_summary = f"This email has been marked as {severity.upper()} risk ({risk_score}%) due to detected phishing indicators. Users should not interact with it."

        return {
            "threat_type": threat_type,
            "severity": severity,
            "summary": summary.strip(),
            "indicators": indicators,
            "recommendations": recommendations,
            "executive_summary": exec_summary
        }
