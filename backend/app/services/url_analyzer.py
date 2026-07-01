import re
import math
from urllib.parse import urlparse
import idna
from typing import Dict, List, Any

# Target brands to test for typosquatting / brand impersonation
POPULAR_BRANDS = [
    "paypal", "google", "apple", "microsoft", "amazon", "netflix",
    "facebook", "bankofamerica", "chase", "wellsfargo", "yahoo",
    "linkedin", "instagram", "github", "twitter"
]

SUSPICIOUS_TLDS = {
    "zip", "mov", "click", "link", "xyz", "top", "info", "gq", "tk", "ml",
    "cf", "ga", "work", "date", "download", "bid", "country", "stream", "loan"
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "rebrand.ly", "goo.gl", "bit.do", "mcaf.ee", "su.pr"
}

class URLAnalyzer:
    @staticmethod
    def calculate_entropy(domain: str) -> float:
        """Calculates Shannon entropy of the domain name to detect random DGA domains."""
        if not domain:
            return 0.0
        # Count letter frequencies
        frequencies = {}
        for char in domain:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        for count in frequencies.values():
            p = count / len(domain)
            entropy -= p * math.log2(p)
        return float(round(entropy, 3))

    @staticmethod
    def LevenshteinDistance(s1: str, s2: str) -> int:
        """Calculates edit distance between two strings."""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    @staticmethod
    def analyze_url(url: str) -> Dict[str, Any]:
        """Performs check-by-check intelligence on a single URL."""
        if not url.startswith(("http://", "https://")):
            # Prepend protocol for parsing if missing
            parsed = urlparse("http://" + url)
        else:
            parsed = urlparse(url)

        domain = parsed.netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]  # strip port

        flags = []
        is_suspicious = False
        
        # 1. Scheme checks
        if parsed.scheme == "http":
            flags.append("Insecure HTTP protocol")
            is_suspicious = True

        # 2. IP address host check
        ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
        if re.match(ip_pattern, domain):
            flags.append("IP-based URL host")
            is_suspicious = True

        # 3. URL shorteners
        if domain in SHORTENERS:
            flags.append("URL shortener service used")
            is_suspicious = True

        # 4. Suspicious TLD check
        tld = domain.split(".")[-1] if "." in domain else ""
        if tld in SUSPICIOUS_TLDS:
            flags.append(f"Suspicious TLD (.{tld})")
            is_suspicious = True

        # 5. Entropy check
        entropy = URLAnalyzer.calculate_entropy(domain)
        if entropy > 3.8 and len(domain) > 12:
            flags.append("High domain name entropy (DGA indicator)")
            is_suspicious = True

        # 6. Typosquatting / Brand Impersonation check
        domain_without_tld = ".".join(domain.split(".")[:-1]) if "." in domain else domain
        
        # Look for homoglyphs / IDN attacks
        try:
            punycode = domain.encode("ascii").decode("ascii")
            is_idn = False
        except UnicodeEncodeError:
            try:
                punycode = idna.encode(domain).decode("ascii")
                is_idn = punycode.startswith("xn--")
            except Exception:
                is_idn = True
                
        if is_idn:
            flags.append("IDN Homograph attack indicator (non-ASCII characters)")
            is_suspicious = True

        # Brand check
        for brand in POPULAR_BRANDS:
            tokens = re.split(r"[.\-_]", domain_without_tld)
            for token in tokens:
                if token == brand:
                    if domain_without_tld != brand and not domain_without_tld.endswith("." + brand):
                        flags.append(f"Brand impersonation (referencing '{brand}' outside official domain)")
                        is_suspicious = True
                else:
                    if len(token) >= len(brand) - 1 and len(token) <= len(brand) + 1:
                        dist = URLAnalyzer.LevenshteinDistance(token, brand)
                        if 0 < dist <= 2:
                            flags.append(f"Typosquatting detected ({token} mimicking {brand})")
                            is_suspicious = True

        # 7. Length check
        if len(url) > 75:
            flags.append("Unusually long URL (>75 chars)")
            is_suspicious = True

        return {
            "url": url,
            "flags": flags,
            "entropy": entropy,
            "is_suspicious": is_suspicious
        }
