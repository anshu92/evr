"""Shared inline-email styling helpers for HTML reports."""


def html_escape(s: str) -> str:
    """Escape text for HTML body content."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def card_wrap(title: str, inner_html: str, *, accent: str = "#111827") -> str:
    """Outer card matching daily email visual tokens."""
    return f"""
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);border-left:4px solid {accent};">
    <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:10px;">{title}</div>
    {inner_html}
  </div>"""
