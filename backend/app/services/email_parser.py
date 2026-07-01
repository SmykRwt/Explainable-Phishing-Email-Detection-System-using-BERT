import email
from email import policy
from email.parser import BytesParser
import re
from typing import Dict, List, Any, Optional

class EmailParser:
    @staticmethod
    def parse_eml(eml_content: str) -> Dict[str, Any]:
        """Parses a raw EML string and extracts relevant fields."""
        try:
            msg = email.message_from_string(eml_content, policy=policy.default)
        except Exception as e:
            # Fallback if parser fails
            return EmailParser.parse_raw_text(eml_content)
        
        headers = {k: v for k, v in msg.items()}
        
        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        reply_to = msg.get("Reply-To", "")
        to = msg.get("To", "")
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors="ignore")
                elif content_type == "text/html" and not body and "attachment" not in content_disposition:
                    # Simple html plain-text extraction fallback
                    payload = part.get_payload(decode=True)
                    if payload:
                        html = payload.decode(errors="ignore")
                        # Strip simple html tags
                        body += re.sub(r"<[^>]+>", "", html)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors="ignore")
        
        if not body:
            # If still empty, use the main body as a string
            body = msg.get_payload() or ""
            if not isinstance(body, str):
                body = str(body)

        # Extract attachments
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                filename = part.get_filename()
                if filename:
                    attachments.append({
                        "filename": filename,
                        "content_type": part.get_content_type(),
                        "size": len(part.get_payload(decode=True) or "")
                    })

        # Find URLs in body
        urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", body)

        return {
            "sender": sender,
            "reply_to": reply_to,
            "to": to,
            "subject": subject,
            "body": body,
            "urls": list(set(urls)),
            "attachments": attachments,
            "headers": headers,
            "is_eml": True
        }

    @staticmethod
    def parse_raw_text(text: str) -> Dict[str, Any]:
        """Parses a plain text email if no EML format is detected."""
        # Simple extraction of headers if present in raw text format
        sender = ""
        subject = ""
        body = text
        
        # Look for simple lines like "From: ...", "Subject: ..." at the top
        lines = text.split("\n")
        header_lines_processed = 0
        for line in lines[:5]:
            if line.lower().startswith("from:"):
                sender = line[5:].strip()
                header_lines_processed += 1
            elif line.lower().startswith("subject:"):
                subject = line[8:].strip()
                header_lines_processed += 1
        
        # If we found headers, we can slice body
        if header_lines_processed > 0:
            body = "\n".join(lines[header_lines_processed:]).strip()

        urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", body)

        return {
            "sender": sender,
            "reply_to": "",
            "to": "",
            "subject": subject,
            "body": body,
            "urls": list(set(urls)),
            "attachments": [],
            "headers": {},
            "is_eml": False
        }
