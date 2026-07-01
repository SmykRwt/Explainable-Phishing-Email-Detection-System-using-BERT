import re
from typing import Dict, List, Any, Optional
from backend.app.schemas.analysis import RuleResult

class RuleEngine:
    @staticmethod
    def evaluate_rules(email_data: Dict[str, Any]) -> List[RuleResult]:
        """Evaluates all rules on the parsed email data and returns triggered rules."""
        triggered = []
        
        body_lower = email_data.get("body", "").lower()
        subject_lower = email_data.get("subject", "").lower()
        full_text = f"{subject_lower}\n{body_lower}"
        attachments = email_data.get("attachments", [])
        urls = email_data.get("urls", [])

        # 1. Credential Harvesting Rule
        cred_pattern = r"\b(login|sign in|password|username|credential|credentials|verify your account|auth|confirm details)\b"
        if re.search(cred_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="Credential Harvesting Lure",
                severity="Critical",
                confidence=0.85,
                reason="Detected text prompts requesting account credentials, login credentials, or verification details."
            ))

        # 2. Urgency Rule
        urgency_pattern = r"\b(urgent|immediately|act now|expires?|within \d+ hours|limited time|don't delay|asap|now)\b"
        if re.search(urgency_pattern, full_text):
            triggered.append(RuleResult(
                rule_name="Urgency / Pressure Tactics",
                severity="High",
                confidence=0.75,
                reason="Detected urgency keywords prompting immediate action or expressing limited-time expiration."
            ))

        # 3. Impersonation Rule
        impersonation_pattern = r"\b(dear customer|dear user|dear member|account holder|official notification|customer support)\b"
        if re.search(impersonation_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="Impersonation Patterns",
                severity="Medium",
                confidence=0.60,
                reason="Detected generic greetings or brand support vocabulary commonly used to spoof identities."
            ))

        # 4. Financial Request Rule
        financial_pattern = r"\b(wire transfer|bank account|bank details|routing number|credit card|billing details|payment method|routing number|fund transfer|transfer the amount|transfer the funds|send me your bank)\b"
        if re.search(financial_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="Financial Request Bait",
                severity="High",
                confidence=0.80,
                reason="Found phrasing asking for bank accounts, bank details, routing details, wire transfers, or credit card updates."
            ))

        # 4b. Prize / Reward Bait Rule
        prize_pattern = r"\b(prize|winner|won|congratulations?|reward|gift|claim|lucky|lottery|draw|jackpot)\b"
        if re.search(prize_pattern, full_text):
            triggered.append(RuleResult(
                rule_name="Prize / Reward Bait",
                severity="High",
                confidence=0.85,
                reason="Detected vocabulary offering winnings, prizes, lotteries, or congratulations that are typically used to lure users into scams."
            ))

        # 5. Password Reset Rule
        pwd_reset_pattern = r"\b(password reset|reset password|forgotten password|recovery link|reset your password)\b"
        if re.search(pwd_reset_pattern, full_text):
            triggered.append(RuleResult(
                rule_name="Password Reset Bait",
                severity="High",
                confidence=0.70,
                reason="Found indicators mimicking a requested password reset or account recovery process."
            ))

        # 6. Account Suspension Rule
        suspension_pattern = r"\b(suspended|restricted|blocked|unauthorized access|deactivated|close your account|account warning)\b"
        if re.search(suspension_pattern, full_text):
            triggered.append(RuleResult(
                rule_name="Account Suspension Threat",
                severity="Critical",
                confidence=0.90,
                reason="Detected keywords suggesting the user's account is suspended, blocked, or has experienced unauthorized access."
            ))

        # 7. Invoice Scam Rule
        invoice_pattern = r"\b(invoice|unpaid balance|receipt|billing invoice|purchase confirmation|order confirmation|payment due)\b"
        if re.search(invoice_pattern, full_text):
            triggered.append(RuleResult(
                rule_name="Invoice Scam / False Billing",
                severity="Medium",
                confidence=0.65,
                reason="Found vocabulary pertaining to billing invoices, payments due, or order receipts."
            ))

        # 8. Gift Card Scam Rule
        gift_card_pattern = r"\b(gift card|gift cards|buy steam|buy amazon card|purchase vanilla|itunes card)\b"
        if re.search(gift_card_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="Gift Card Purchase Solicitation",
                severity="High",
                confidence=0.85,
                reason="Found solicitations asking to purchase, scan, or verify gift card codes."
            ))

        # 9. QR Code Present Rule
        qr_pattern = r"\b(qr code|scan the code|scan qr|camera scanner|scan below)\b"
        if re.search(qr_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="QR Code Phishing (Quishing) Prompt",
                severity="High",
                confidence=0.70,
                reason="Text requests scanning a QR code, which is a common vector to bypass email URL screening."
            ))

        # 9b. Call to Action Prompt Rule
        cta_pattern = r"\b(click|reply|visit|subscribe|register|wap link|call this|text back|click here)\b"
        if re.search(cta_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="Call to Action Prompt",
                severity="Medium",
                confidence=0.65,
                reason="Detected direct commands prompting the user to click links, reply, or register."
            ))

        # 9c. Financial Lure Rule
        lure_pattern = r"\b(free|cash|money|earn|profit|credit|credits|win|won|cashback)\b"
        if re.search(lure_pattern, body_lower):
            triggered.append(RuleResult(
                rule_name="Financial Lure Bait",
                severity="Medium",
                confidence=0.70,
                reason="Detected keywords offering credits, free items, cash back, or financial winnings typical of lures."
            ))

        # 10. Attachment Risk Rule
        high_risk_ext = {".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".msi", ".zip", ".rar", ".7z", ".pdf", ".docm", ".xlsm"}
        dangerous_attachment = False
        reasons = []
        for att in attachments:
            fname = att.get("filename", "").lower()
            for ext in high_risk_ext:
                if fname.endswith(ext):
                    dangerous_attachment = True
                    reasons.append(f"attachment '{fname}' has risky extension ({ext})")
        
        # Also check if text mentions "attachment" or "attached" when there are no attachments
        if "attached" in body_lower or "attachment" in body_lower:
            if not attachments:
                # Often scammers fake attachments or prompt the user about missing items
                pass
        
        if dangerous_attachment:
            triggered.append(RuleResult(
                rule_name="Dangerous / Attachment Risk",
                severity="Critical",
                confidence=0.90,
                reason=f"Dangerous attachment detected: {', '.join(reasons)}."
            ))
        elif attachments:
            triggered.append(RuleResult(
                rule_name="Attachment Present",
                severity="Low",
                confidence=0.50,
                reason=f"Email contains {len(attachments)} attachment(s)."
            ))

        # Extra Check: URL mismatch rule
        suspicious_urls = [url for url in urls if "suspicious" in url or len(url) > 100]
        if len(urls) > 5:
            triggered.append(RuleResult(
                rule_name="High URL Density",
                severity="Medium",
                confidence=0.70,
                reason="The body contains an unusually high number of URLs, typical of spam newsletters or redirect campaigns."
            ))

        return triggered
