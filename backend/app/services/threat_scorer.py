from typing import List, Dict, Any
from backend.app.schemas.analysis import RuleResult, URLFindingBase

class ThreatScorer:
    @staticmethod
    def calculate_rules_score(rules: List[RuleResult]) -> float:
        """Returns a score from 0 to 100 based on triggered rules."""
        if not rules:
            return 0.0
        
        # Mapping severity to severity values
        severity_mapping = {
            "critical": 95.0,
            "high": 75.0,
            "medium": 45.0,
            "low": 15.0
        }
        
        scores = []
        for rule in rules:
            sev = rule.severity.lower()
            scores.append(severity_mapping.get(sev, 10.0))
            
        # Max-pooling: if a critical rule is found, the rule threat score is high.
        # But we add a small premium if multiple rules are triggered.
        max_score = max(scores)
        count_bonus = min(len(scores) - 1, 5) * 5.0  # Up to 25 points bonus for multiple triggers
        
        return min(max_score + count_bonus, 100.0)

    @staticmethod
    def calculate_urls_score(url_findings: List[Any]) -> float:
        """Returns a score from 0 to 100 based on parsed URL vulnerabilities."""
        if not url_findings:
            return 0.0
        
        # Check if any URL is suspicious (handles both dict and DB model object)
        suspicious_urls = []
        for u in url_findings:
            is_susp = u.get("is_suspicious") if isinstance(u, dict) else getattr(u, "is_suspicious", False)
            if is_susp:
                suspicious_urls.append(u)

        if not suspicious_urls:
            return 0.0
            
        flag_scores = {
            "insecure http protocol": 30.0,
            "url shortener service used": 60.0,
            "suspicious tld": 65.0,
            "high domain name entropy (dga indicator)": 70.0,
            "ip-based url host": 75.0,
            "brand impersonation": 80.0,
            "idn homograph attack indicator (non-ascii characters)": 85.0,
            "typosquatting detected": 90.0,
            "unusually long url": 15.0
        }
        
        scores = []
        for u in suspicious_urls:
            flags = u.get("flags", []) if isinstance(u, dict) else getattr(u, "flags", [])
            # Handle DB comma-separated string format
            if isinstance(flags, str):
                flags = [f.strip() for f in flags.split(",") if f.strip()]
                
            for flag in flags:
                matched = False
                for k, v in flag_scores.items():
                    if k in flag.lower():
                        scores.append(v)
                        matched = True
                if not matched:
                    scores.append(20.0)
                    
        if not scores:
            return 10.0
            
        return float(min(max(scores) + min(len(scores) - 1, 3) * 5.0, 100.0))

    @staticmethod
    def calculate_header_score(header_findings: Dict[str, Any]) -> float:
        """Returns a score from 0 to 100 based on header analysis."""
        if not header_findings.get("is_suspicious", False):
            return 0.0
            
        score = 0.0
        findings = header_findings.get("findings", [])
        
        finding_scores = {
            "spf authentication failed": 40.0,
            "dkim signature authentication failed": 40.0,
            "dmarc compliance policy failed": 50.0,
            "reply-to mismatch": 60.0,
            "display name spoofing": 75.0,
            "sender mismatch": 80.0
        }
        
        scores = []
        for finding in findings:
            matched = False
            for k, v in finding_scores.items():
                if k in finding.lower():
                    scores.append(v)
                    matched = True
            if not matched:
                scores.append(25.0)
                
        if not scores:
            return 10.0
            
        return float(min(max(scores) + min(len(scores) - 1, 3) * 5.0, 100.0))

    @staticmethod
    def compute_risk_score(
        bert_spam_prob: float,
        rules: List[RuleResult],
        url_findings: List[Any],
        header_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combines predictions, rules, URLs, and headers into a composite risk score (0-100).
        Weights dynamically re-allocated if URLs or headers are not present/applicable.
        """
        rules_score = ThreatScorer.calculate_rules_score(rules)
        urls_score = ThreatScorer.calculate_urls_score(url_findings)
        header_score = ThreatScorer.calculate_header_score(header_findings)
        
        # Determine applicability
        url_applicable = len(url_findings) > 0
        
        sender_email = header_findings.get("sender_parsed", {}).get("email", "") if isinstance(header_findings, dict) else ""
        header_applicable = bool(sender_email) or len(header_findings.get("findings", [])) > 0
        
        # Default weights
        weights = {
            "bert": 0.45,
            "rules": 0.25,
            "urls": 0.15,
            "headers": 0.15
        }
        
        # Build list of active components
        active = ["bert", "rules"]
        if url_applicable:
            active.append("urls")
        if header_applicable:
            active.append("headers")
            
        total_active_weight = sum(weights[c] for c in active)
        
        # Calculate normalized score
        weighted_score = 0.0
        if total_active_weight > 0:
            norm_bert = weights["bert"] / total_active_weight
            norm_rules = weights["rules"] / total_active_weight
            
            weighted_score += (bert_spam_prob * norm_bert) + (rules_score * norm_rules)
            
            if "urls" in active:
                norm_urls = weights["urls"] / total_active_weight
                weighted_score += urls_score * norm_urls
            if "headers" in active:
                norm_headers = weights["headers"] / total_active_weight
                weighted_score += header_score * norm_headers
        else:
            # Fallback
            weighted_score = (bert_spam_prob + rules_score) / 2.0
            
        risk_score = float(round(min(max(weighted_score, 0.0), 100.0), 2))
        
        # If BERT is highly confident, ensure the risk score reflects that high confidence
        if bert_spam_prob >= 85.0:
            risk_score = max(risk_score, float(round(bert_spam_prob * 0.85, 2)))
            
        # Check for Trusted Sender Discount (cryptographic authentication whitelist)
        is_trusted_sender = False
        if header_applicable and isinstance(header_findings, dict):
            spf = header_findings.get("spf_pass", False)
            dkim = header_findings.get("dkim_pass", False)
            dmarc = header_findings.get("dmarc_pass", False)
            sender_domain = header_findings.get("sender_parsed", {}).get("domain", "").lower()
            
            trusted_domains = {"microsoft.com", "google.com", "paypal.com", "apple.com", "amazon.com", "netflix.com"}
            if sender_domain in trusted_domains and spf and dkim and dmarc:
                is_trusted_sender = True
                
        # Determine classification label from the composite score
        is_soft_whitelisted = False
        sender_email = header_findings.get("sender_parsed", {}).get("email", "") if isinstance(header_findings, dict) else ""
        sender_domain = ""
        if "@" in sender_email:
            sender_domain = sender_email.split("@")[-1].lower()
            
        trusted_domains = {"microsoft.com", "google.com", "paypal.com", "apple.com", "amazon.com", "netflix.com"}
        has_suspicious_urls = any(u.get("is_suspicious") if isinstance(u, dict) else getattr(u, "is_suspicious", False) for u in url_findings)
        
        # If it claims to be from a trusted brand but lacks cryptographic proof (not is_trusted_sender)
        # and has no suspicious links, we downgrade it to Suspicious (capped at 55.0%) instead of High Risk
        if sender_domain in trusted_domains and not is_trusted_sender and not has_suspicious_urls:
            is_soft_whitelisted = True

        if is_trusted_sender:
            risk_score = float(round(min(risk_score * 0.05, 5.0), 2)) # reduce by 95%, max 5%
            verdict = "✅ Safe"
        elif is_soft_whitelisted:
            risk_score = float(round(min(risk_score, 55.0), 2)) # cap at 55.0%, Suspicious
            verdict = "🟡 Suspicious"
        elif risk_score >= 70.0:
            verdict = "⚠️ High Phishing Risk"
        elif risk_score >= 40.0:
            verdict = "🟡 Suspicious"
        else:
            verdict = "✅ Safe"
            
        return {
            "risk_score": risk_score,
            "verdict": verdict,
            "breakdown": {
                "bert_score": float(round(bert_spam_prob, 2)),
                "rules_score": float(round(rules_score, 2)),
                "urls_score": float(round(urls_score, 2)),
                "header_score": float(round(header_score, 2))
            }
        }
