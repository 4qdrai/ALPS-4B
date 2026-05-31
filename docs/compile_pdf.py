import os
import re
import time
import markdown
from playwright.sync_api import sync_playwright

def compile_markdown_to_pdf():
    print("=== ALPS-4B: Compiling Scientific Paper PDF (Playwright) ===")
    
    md_path = "docs/scientific_paper.md"
    html_path = "docs/scientific_paper_temp.html"
    pdf_path = "docs/scientific_paper.pdf"
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # --- Protect Math Blocks from Markdown Parser ---
    # We extract LaTeX formulas before Markdown parsing and re-inject them afterwards 
    # to prevent underscores (_) or backslashes (\) from being corrupted by HTML tags.
    display_math_placeholders = []
    inline_math_placeholders = []
    
    # 1. Extract Fenced Math Blocks: ```math ... ```
    def save_fenced_math(match):
        formula = match.group(1).strip()
        placeholder = f"<!--DISPLAY_MATH_{len(display_math_placeholders)}-->"
        display_math_placeholders.append(f"<div class='equation-container'>$$\\begin{{equation}}{formula}\\end{{equation}}$$</div>")
        return placeholder
        
    md_protected = re.sub(r'```math\s*(.*?)\s*```', save_fenced_math, md_content, flags=re.DOTALL)
    
    # 2. Extract Double Dollar Math Blocks: $$ ... $$
    def save_display_math(match):
        formula = match.group(1).strip()
        placeholder = f"<!--DISPLAY_MATH_{len(display_math_placeholders)}-->"
        display_math_placeholders.append(f"<div class='equation-container'>$$\\begin{{equation}}{formula}\\end{{equation}}$$</div>")
        return placeholder
        
    md_protected = re.sub(r'\$\$\s*(.*?)\s*\$\$', save_display_math, md_protected, flags=re.DOTALL)
    
    # 3. Extract Single Dollar Inline Math Blocks: $ ... $
    # (Matches single $ but not double $$)
    def save_inline_math(match):
        formula = match.group(1).strip()
        placeholder = f"<!--INLINE_MATH_{len(inline_math_placeholders)}-->"
        inline_math_placeholders.append(f"${formula}$")
        return placeholder
        
    md_protected = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inline_math, md_protected)
    
    # --- Convert Protected Markdown to HTML ---
    html_body = markdown.markdown(md_protected, extensions=['extra', 'codehilite'])
    
    # --- Re-inject Math Blocks ---
    for idx, formula in enumerate(display_math_placeholders):
        html_body = html_body.replace(f"<!--DISPLAY_MATH_{idx}-->", formula)
        
    for idx, formula in enumerate(inline_math_placeholders):
        html_body = html_body.replace(f"<!--INLINE_MATH_{idx}-->", formula)
        
    # --- Wrap in beautiful scientific paper CSS styling ---
    html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ALPS-4B: Adaptive Latent Prediction System</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/latin-modern-webfonts@1.0.0/style.css">
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
      processEscapes: true
    }},
    startup: {{
      pageReady: () => {{
        return MathJax.startup.defaultPageReady().then(() => {{
          window.mathjaxReady = true;
        }});
      }}
    }}
  }};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    @page {{
        size: A4;
        margin: 2.54cm;
    }}
    body {{
        font-family: 'Latin Modern Roman', 'Times New Roman', serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #000000;
        margin: 0;
        padding: 0;
        background-color: #ffffff;
        text-align: justify;
    }}
    .paper-title {{
        font-size: 16pt;
        font-weight: bold;
        text-align: center;
        margin-top: 1cm;
        margin-bottom: 0.6cm;
        line-height: 1.25;
    }}
    .author-box {{
        text-align: center;
        font-size: 11pt;
        margin-bottom: 0.8cm;
        line-height: 1.4;
    }}
    .abstract-box {{
        margin: 0 1.2cm 0.8cm 1.2cm;
        font-size: 9.5pt;
        line-height: 1.45;
        background-color: #fdfdfd;
        border: 0.5px solid #cccccc;
        padding: 15px;
        border-radius: 4px;
    }}
    .abstract-title {{
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
        font-size: 10pt;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .abstract-text {{
        font-style: italic;
        margin: 0;
        text-align: justify;
    }}
    h2 {{
        font-size: 12pt;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 10px;
        border-bottom: 0.5px solid #000000;
        padding-bottom: 3px;
    }}
    h3 {{
        font-size: 11pt;
        font-weight: bold;
        margin-top: 18px;
        margin-bottom: 8px;
    }}
    p {{
        margin: 0 0 10px 0;
        text-indent: 1.5em;
    }}
    /* First paragraph after headings should not have text indentation */
    h2 + p, h3 + p, .abstract-box + p, div + p {{
        text-indent: 0 !important;
    }}
    ul, ol {{
        margin: 6px 0;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 5px;
    }}
    code {{
        font-family: 'Courier New', monospace;
        background-color: #f5f5f5;
        font-size: 8.5pt;
        padding: 1px 3px;
    }}
    pre {{
        font-family: 'Courier New', monospace;
        background-color: #fafafa;
        border: 0.5px solid #dddddd;
        padding: 10px;
        font-size: 8pt;
        margin-bottom: 15px;
        white-space: pre-wrap;
    }}
    .equation-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 14px 0;
        width: 100%;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 9pt;
    }}
    th {{
        border-top: 1.5px solid #000000;
        border-bottom: 1px solid #000000;
        padding: 6px 8px;
        font-weight: bold;
        text-align: left;
    }}
    td {{
        border-bottom: 1px solid #dddddd;
        padding: 5px 8px;
        text-align: left;
    }}
    tr:last-child td {{
        border-bottom: 1.5px solid #000000;
    }}
    hr {{
        border: none;
        border-top: 0.5px solid #cccccc;
        margin: 20px 0;
    }}
</style>
</head>
<body>
    <div class="paper-title">ALPS-4B: Adaptive Latent Prediction System with Hierarchical Latent Predictive Architectures and Reflexive Safety Watchdogs</div>
    <div class="author-box">
        <strong>Dr-Ing. M. Essayed Bouzouraa and 4QDR.AI Labs</strong><br/>
        <em>4qdr.ai@gmail.com</em><br/>
        <a href="https://github.com/4qdrai/ALPS-4B">GitHub Repository</a>
    </div>
    
    {html_body}
</body>
</html>
"""

    # --- Publication adjustments in compiled HTML ---
    # 1. Format Abstract Block
    html_template = html_template.replace(
        "<h2>Abstract</h2>\n<p>",
        "<div class='abstract-box'><div class='abstract-title'>Abstract</div><p class='abstract-text'>"
    )
    # End the abstract-box div at the end of the abstract paragraph
    split_str = "<h2>1. Introduction</h2>"
    if split_str in html_template:
        parts = html_template.split(split_str)
        # Add closing div to abstract section (part 1)
        parts[0] = parts[0] + "</div>\n"
        html_template = split_str.join(parts)
        
    # Remove redundant title and author text if parsed from md
    html_template = html_template.replace(
        "<h1>ALPS-4B: Adaptive Latent Prediction System with Hierarchical Latent Predictive Architectures and Reflexive Safety Watchdogs</h1>",
        ""
    )
    html_template = html_template.replace(
        "<p><strong>Dr-Ing. M. Essayed Bouzouraa and 4QDR.AI Labs</strong><br />\n<em>4qdr.ai@gmail.com</em><br />\n<a href=\"https://github.com/4qdrai/ALPS-4B\">GitHub Repository</a></p>",
        ""
    )
    
    # Write temporary HTML file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Successfully generated HTML draft at: {html_path}")
    
    # --- Launch Playwright to compile PDF ---
    print("Launching Playwright to compile PDF...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 800, "height": 600},
            device_scale_factor=2
        )
        page = context.new_page()
        file_url = f"file:///{os.path.abspath(html_path).replace(os.sep, '/')}"
        page.goto(file_url)
        page.emulate_media(media="print")
        
        print("Waiting for MathJax typesetter...")
        page.wait_for_function("window.mathjaxReady === true", timeout=30000)
        time.sleep(1.5)  # Let layout settle
        
        print("Printing to A4 PDF...")
        page.pdf(
            path=pdf_path,
            print_background=True,
            format="A4",
            margin={"top": "2.54cm", "bottom": "2.54cm", "left": "2.54cm", "right": "2.54cm"}
        )
        browser.close()
        
    print(f"PDF successfully generated and saved to: {pdf_path}")
    
    # Clean up intermediate HTML
    if os.path.exists(html_path):
        os.remove(html_path)

if __name__ == "__main__":
    compile_markdown_to_pdf()
