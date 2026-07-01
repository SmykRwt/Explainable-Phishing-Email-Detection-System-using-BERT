import re
from typing import Dict, List, Any, Optional

FREE_MAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "protonmail.com", "icloud.com"}

class HeaderAnalyzer:
    @staticmethod
    def parse_email_address(raw_header: str) -> Dict[str, str]:
        """Parses a display name and email address from standard headers like From."""
        if not raw_header:
            return {"display_name": "", "email": "", "domain": ""}
        
        # Match pattern "Display Name" <email@domain.com>
        match = re.match(r'^(.*?)\s*<([^>]+)>', raw_header)
        if match:
            display_name = match.group(1).strip().replace('"', '').replace("'", "")
            email_addr = match.group(2).strip()
        else:
            display_name = ""
            email_addr = raw_header.strip()
        
        domain = email_addr.split("@")[-1].lower() if "@" in email_addr else ""
        return {
            "display_name": display_name,
            "email": email_addr,
            "domain": domain
        }

    @staticmethod
    def analyze_headers(headers: Dict[str, Any], sender_field: str = "", reply_to_field: str = "") -> Dict[str, Any]:
        """Inspects email authentication headers and validates field alignments."""
        findings = []
        is_suspicious = False
        
        # 1. Parse From Address
        from_data = HeaderAnalyzer.parse_email_address(sender_field)
        
        # 2. Reply-To Mismatch
        if reply_to_field:
            reply_data = HeaderAnalyzer.parse_email_address(reply_to_field)
            if reply_data["email"] and from_data["email"] and reply_data["email"] != from_data["email"]:
                # If domain is different
                if reply_data["domain"] != from_data["domain"]:
                    findings.append(f"Reply-To Mismatch: replies go to {reply_data['email']} but sender is {from_data['email']}")
                    is_suspicious = True

        # 3. Display Name Mismatch / Spoofing
        # E.g., From: "PayPal Security <billing@paypal.com>" <hackers@gmail.com>
        display_name = from_data["display_name"]
        if display_name:
            email_in_display = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', display_name)
            if email_in_display:
                found_email = email_in_display.group(0).lower()
                if found_email != from_data["email"].lower():
                    findings.append(f"Display Name Spoofing: email address in Display Name '{found_email}' hides actual sender '{from_data['email']}'")
                    is_suspicious = True

            # E.g., From: "Netflix Support" <support@free-netflix-site.com> or <free-netflix@gmail.com>
            known_brands = ["paypal", "netflix", "google", "apple", "microsoft", "amazon", "chase", "bankofamerica"]
            for brand in known_brands:
                if brand in display_name.lower():
                    if brand not in from_data["domain"]:
                        findings.append(f"Sender Mismatch: Display Name claims to be '{brand}' but email domain is '{from_data['domain']}'")
                        is_suspicious = True

        # 4. Authentication Check (SPF, DKIM, DMARC)
        auth_results = headers.get("Authentication-Results", "")
        received_spf = headers.get("Received-SPF", "")
        
        spf_pass = True
        dkim_pass = True
        dmarc_pass = True

        # Convert to single string for scanning
        auth_str = (str(auth_results) + " " + str(received_spf)).lower()

        if auth_str:
            # Check SPF
            if "spf=fail" in auth_str or "spf=softfail" in auth_str:
                findings.append("SPF Authentication failed")
                spf_pass = False
                is_suspicious = True
            
            # Check DKIM
            if "dkim=fail" in auth_str:
                findings.append("DKIM Signature authentication failed")
                dkim_pass = False
                is_suspicious = True
            
            # Check DMARC
            if "dmarc=fail" in auth_str:
                findings.append("DMARC Compliance policy failed")
                dmarc_pass = False
                is_suspicious = True

        return {
            "findings": findings,
            "spf_pass": spf_pass,
            "dkim_pass": dkim_pass,
            "dmarc_pass": dmarc_pass,
            "is_suspicious": is_suspicious,
            "sender_parsed": from_data
        }
