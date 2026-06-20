import re
from pathlib import Path

HTML = Path(__file__).parent.parent / "eujeno" / "ui" / "static" / "index.html"

def read_html():
    return HTML.read_text(encoding="utf-8")

def test_navigator_language_present():
    assert "navigator.language" in read_html()

def test_eujeno_lang_key_present():
    assert "'eujeno_lang'" in read_html()

def test_begin_translations_marker():
    assert "// BEGIN_TRANSLATIONS" in read_html()

def test_end_translations_marker():
    assert "// END_TRANSLATIONS" in read_html()

def test_native_language_names():
    html = read_html()
    for name in ["Italiano", "English", "Français", "Deutsch", "Español"]:
        assert name in html, f"Native name '{name}' not found in HTML"
