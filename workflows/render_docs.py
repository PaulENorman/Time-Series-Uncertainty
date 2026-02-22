from pathlib import Path
import html
import re


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "index.md"
OUT = ROOT / "docs" / "index.html"

IMG_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
OL_RE = re.compile(r"^\d+\.\s+(.*)$")
UL_RE = re.compile(r"^[-*]\s+(.*)$")
LINK_RE = re.compile(r"(?<!\\)\[(.+?)\]\((.+?)\)")
CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def inline_format(text: str) -> str:
    # Keep TeX syntax untouched for MathJax.
    text = CODE_RE.sub(lambda m: f"<code>{html.escape(m.group(1))}</code>", text)
    text = BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)
    text = LINK_RE.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        text,
    )
    return text


def flush_paragraph(buf: list[str], parts: list[str]) -> None:
    if not buf:
        return
    joined = " ".join(s.strip() for s in buf if s.strip())
    if joined:
        parts.append(f"<p>{inline_format(joined)}</p>")
    buf.clear()


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Uncertainty in the Mean for Correlated Signals</title>",
        """<style>
:root { --bg:#f7f8fa; --panel:#ffffff; --ink:#17202a; --line:#d8dee9; }
body { margin:0; background:linear-gradient(180deg,#eef2f7 0%,#f8fafc 50%,#f3f6fb 100%); color:var(--ink); font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif; line-height:1.6; }
main { max-width: 980px; margin: 24px auto; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding: 28px 36px; box-shadow:0 10px 30px rgba(16,24,40,.06); }
h1,h2,h3,h4 { line-height:1.25; margin:1.1em 0 .5em; }
h1 { font-size: 2.1rem; border-bottom:2px solid var(--line); padding-bottom:.4rem; }
h2 { font-size: 1.45rem; color:#0b5563; }
p { margin:.7em 0; }
hr { border:0; border-top:1px solid var(--line); margin:1.2em 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { background:#eef4ff; border:1px solid #d8e3ff; border-radius:4px; padding:1px 4px; font-size:.95em; }
pre { background:#111827; color:#f9fafb; border-radius:10px; padding:14px; overflow:auto; border:1px solid #1f2937; }
ul,ol { margin:.5em 0 .9em 1.25em; }
img { display:block; max-width:100%; height:auto; margin:.6em auto 1.1em; border:1px solid var(--line); border-radius:8px; }
a { color:#0f766e; text-decoration:none; border-bottom:1px dotted #0f766e; }
.math { margin: .8em 0; }
</style>""",
        """<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }
};
</script>""",
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>',
        "</head><body><main>",
        "<p>Rendered from <code>docs/index.md</code>.</p>",
    ]

    in_code = False
    in_ul = False
    in_ol = False
    para_buf: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Display math block \[ ... \]
        if line.strip() == r"\[":
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            i += 1
            math_lines = []
            while i < n and lines[i].strip() != r"\]":
                math_lines.append(lines[i])
                i += 1
            parts.append('<div class="math">\\[' + "\n".join(math_lines) + r"\]</div>")
            if i < n and lines[i].strip() == r"\]":
                i += 1
            continue

        if line.startswith("```"):
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_code:
                in_code = True
                parts.append("<pre><code>")
            else:
                in_code = False
                parts.append("</code></pre>")
            i += 1
            continue

        if in_code:
            parts.append(html.escape(line))
            i += 1
            continue

        if not line.strip():
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            i += 1
            continue

        if line.strip() == "---":
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            parts.append("<hr>")
            i += 1
            continue

        m = HEAD_RE.match(line)
        if m:
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            lvl = len(m.group(1))
            parts.append(f"<h{lvl}>{inline_format(m.group(2))}</h{lvl}>")
            i += 1
            continue

        m = IMG_RE.match(line)
        if m:
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            alt = html.escape(m.group(1), quote=True)
            src = html.escape(m.group(2), quote=True)
            parts.append(f'<img src="{src}" alt="{alt}">')
            i += 1
            continue

        m = UL_RE.match(line)
        if m:
            flush_paragraph(para_buf, parts)
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{inline_format(m.group(1))}</li>")
            i += 1
            continue

        m = OL_RE.match(line)
        if m:
            flush_paragraph(para_buf, parts)
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if not in_ol:
                parts.append("<ol>")
                in_ol = True
            parts.append(f"<li>{inline_format(m.group(1))}</li>")
            i += 1
            continue

        para_buf.append(line)
        i += 1

    flush_paragraph(para_buf, parts)
    if in_ul:
        parts.append("</ul>")
    if in_ol:
        parts.append("</ol>")
    if in_code:
        parts.append("</code></pre>")
    parts.append("</main></body></html>")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT.resolve())


if __name__ == "__main__":
    main()
