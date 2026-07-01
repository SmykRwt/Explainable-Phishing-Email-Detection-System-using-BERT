import pytest
from backend.app.rules.engine import RuleEngine

def test_credential_harvesting_rule():
    # Phishing email asking for sign-in details
    email_data = {
        "body": "Please login to your account using the credentials below.",
        "subject": "Security notification",
        "urls": [],
        "attachments": [],
        "sender": "sender@attacker.com",
        "reply_to": ""
    }
    rules = RuleEngine.evaluate_rules(email_data)
    rule_names = [r.rule_name for r in rules]
    assert "Credential Harvesting Lure" in rule_names

def test_urgency_rule():
    # Urgent phrasing
    email_data = {
        "body": "Your bank account will be deactivated within 24 hours. Please act now.",
        "subject": "URGENT NOTICE",
        "urls": [],
        "attachments": [],
        "sender": "bank@secure.com",
        "reply_to": ""
    }
    rules = RuleEngine.evaluate_rules(email_data)
    rule_names = [r.rule_name for r in rules]
    assert "Urgency / Pressure Tactics" in rule_names
    assert "Account Suspension Threat" in rule_names

def test_attachment_risk():
    # dangerous attachment
    email_data = {
        "body": "See attached document for invoice details.",
        "subject": "Invoice details",
        "urls": [],
        "attachments": [{"filename": "invoice.pdf.exe", "size": 1024}],
        "sender": "accounts@corp.com",
        "reply_to": ""
    }
    rules = RuleEngine.evaluate_rules(email_data)
    rule_names = [r.rule_name for r in rules]
    assert "Dangerous / Attachment Risk" in rule_names
