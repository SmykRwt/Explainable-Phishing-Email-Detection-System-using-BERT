from backend.app.services.email_parser import EmailParser

def test_raw_text_parsing():
    raw_text = "From: security@brand.com\nSubject: Account Verification\n\nPlease log in here: http://brand-login.com"
    parsed = EmailParser.parse_raw_text(raw_text)
    assert parsed["sender"] == "security@brand.com"
    assert parsed["subject"] == "Account Verification"
    assert "http://brand-login.com" in parsed["urls"]
    assert "log in" in parsed["body"]

def test_eml_parsing():
    eml_data = """From: billing@corp.com
Reply-To: support@gmail.com
Subject: Overdue Invoice
Content-Type: text/plain; charset="utf-8"

Hi, please check the invoice on https://corp-invoice-download.com/123
"""
    parsed = EmailParser.parse_eml(eml_data)
    assert parsed["sender"] == "billing@corp.com"
    assert parsed["reply_to"] == "support@gmail.com"
    assert parsed["subject"] == "Overdue Invoice"
    assert "https://corp-invoice-download.com/123" in parsed["urls"]
