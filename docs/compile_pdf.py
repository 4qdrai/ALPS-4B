import os
import markdown
from xhtml2pdf import pisa

def compile_markdown_to_pdf():
    print("=== ALPS-4B: Compiling Scientific Paper PDF ===")
    
    md_path = "docs/scientific_paper.md"
    html_path = "docs/scientific_paper_temp.html"
    pdf_path = "docs/scientific_paper.pdf"
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
    
    # Wrap in beautiful publication CSS styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: letter;
            margin: 1in;
        }}
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 10.5pt;
            line-height: 1.6;
            color: #333333;
        }}
        h1 {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 22pt;
            font-weight: bold;
            text-align: center;
            margin-top: 0;
            margin-bottom: 5px;
            color: #111111;
        }}
        h2 {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 14pt;
            font-weight: bold;
            margin-top: 25px;
            margin-bottom: 10px;
            border-bottom: 0.5pt solid #cccccc;
            padding-bottom: 3px;
            color: #222222;
        }}
        h3 {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 5px;
            color: #333333;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 10px;
            text-align: justify;
        }}
        .author-box {{
            text-align: center;
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            margin-bottom: 25px;
            color: #444444;
        }}
        .abstract-box {{
            background-color: #f8f9fa;
            border: 0.5pt solid #dddddd;
            padding: 12px;
            margin-bottom: 30px;
            border-radius: 4px;
        }}
        .abstract-title {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #111111;
        }}
        .abstract-text {{
            font-size: 9.5pt;
            line-height: 1.5;
            margin: 0;
            text-align: justify;
        }}
        code {{
            font-family: 'Courier New', monospace;
            background-color: #f1f1f1;
            font-size: 9pt;
            padding: 1px 3px;
        }}
        pre {{
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            border: 0.5pt solid #dddddd;
            padding: 10px;
            font-size: 8.5pt;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 9pt;
        }}
        th {{
            background-color: #f1f1f1;
            font-weight: bold;
            border: 0.5pt solid #dddddd;
            padding: 6px;
            text-align: left;
        }}
        td {{
            border: 0.5pt solid #dddddd;
            padding: 6px;
        }}
        hr {{
            border: none;
            border-top: 0.5pt solid #cccccc;
            margin: 20px 0;
        }}
        .math {{
            text-align: center;
            font-style: italic;
            margin: 15px 0;
            font-size: 11pt;
        }}
    </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    # Make custom adjustments for neat styling
    # 1. Format Author Block
    html_template = html_template.replace(
        "<p><strong>Dr-Ing. M. Essayed Bouzouraa and 4QDR.AI Labs</strong><br />\n<em>4qdr.ai@gmail.com</em><br />\n<a href=\"https://github.com/4qdrai/ALPS-4B\">GitHub Repository</a></p>",
        "<div class='author-box'><strong>Dr-Ing. M. Essayed Bouzouraa and 4QDR.AI Labs</strong><br/><em>4qdr.ai@gmail.com</em><br/><a href='https://github.com/4qdrai/ALPS-4B'>GitHub Repository</a></div>"
    )
    
    # 2. Format Abstract Block
    html_template = html_template.replace(
        "<h2>Abstract</h2>\n<p>",
        "<div class='abstract-box'><div class='abstract-title'>Abstract</div><p class='abstract-text'>"
    )
    # End the abstract-box div at the end of abstract section
    split_str = "<h2>1. Introduction</h2>"
    if split_str in html_template:
        parts = html_template.split(split_str)
        # Add closing div to abstract section (part 1)
        parts[0] = parts[0] + "</div>\n"
        html_template = split_str.join(parts)
        
    # Write temporary HTML file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    # Compile to PDF
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_template, dest=pdf_file)
        
    if pisa_status.err:
        print("Error compiling PDF.")
    else:
        print(f"Successfully compiled PDF and saved to {pdf_path}!")
        
    # Clean up temp html
    if os.path.exists(html_path):
        os.remove(html_path)

if __name__ == "__main__":
    compile_markdown_to_pdf()
