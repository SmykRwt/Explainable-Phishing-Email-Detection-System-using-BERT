from backend.app.services.url_analyzer import URLAnalyzer

def test_url_tld_and_scheme():
    # Insecure and suspicious TLD
    res = URLAnalyzer.analyze_url("http://example.xyz")
    assert res["is_suspicious"] is True
    assert "Insecure HTTP protocol" in res["flags"]
    assert "Suspicious TLD (.xyz)" in res["flags"]

def test_url_typosquatting():
    # Typosquatting paypal
    res = URLAnalyzer.analyze_url("https://paypa1-update.com/login")
    assert res["is_suspicious"] is True
    assert any("Typosquatting" in f or "impersonation" in f.lower() for f in res["flags"])

def test_url_shorteners():
    # Shortener link
    res = URLAnalyzer.analyze_url("https://bit.ly/34sfw")
    assert res["is_suspicious"] is True
    assert "URL shortener service used" in res["flags"]
