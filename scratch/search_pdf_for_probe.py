import os
import sys

# Configure standard output to handle UTF-8 characters on Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def search_pdf_for_probe():
    pdf_path = "2603.19312v2.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    try:
        import pypdf
    except ImportError:
        print("Installing pypdf...")
        os.system("pip install pypdf")
        import pypdf

    reader = pypdf.PdfReader(pdf_path)
    print(f"Successfully loaded PDF. Pages: {len(reader.pages)}")

    keywords = ["probe", "decoder", "linear", "regression", "freeze", "frozen", "planning", "mpc", "cross entropy", "cem", "receding", "horizon", "trajectory", "path"]
    
    matches = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        # Split into sentences or lines
        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            found_kws = [kw for kw in keywords if kw in line_lower]
            if found_kws:
                matches.append((idx + 1, found_kws, line.strip()))

    print(f"\nTotal matches found: {len(matches)}")
    print("\n--- Match Details (grouped by page) ---")
    
    # Group matches by page and print
    current_page = None
    count = 0
    for page_num, kws, line in matches:
        if page_num != current_page:
            current_page = page_num
            print(f"\n--- Page {page_num} ---")
        print(f"  [KWs: {', '.join(kws)}] {line}")
        count += 1
        if count > 150: # Limit output length
            print("\n... Truncated due to length ...")
            break

if __name__ == "__main__":
    search_pdf_for_probe()
