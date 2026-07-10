"""Inline screenshots as base64 into report.html -> report_final.html.

Then render to PDF with:
  "Google Chrome" --headless --print-to-pdf=Image_Extractor_Agent_Report.pdf report_final.html
"""
import base64
import io
import pathlib

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent


def datauri(path, maxw=1100, q=84, crop_bottom_frac=None):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if crop_bottom_frac:
        im = im.crop((0, 0, w, int(h * crop_bottom_frac)))
        w, h = im.size
    if w > maxw:
        im = im.resize((maxw, int(h * maxw / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    search = datauri(HERE / "assets/search.png", maxw=1100, q=84)
    atlas = datauri(HERE / "assets/atlas.png", maxw=1100, q=86, crop_bottom_frac=0.52)
    html = (HERE / "report.html").read_text()
    html = html.replace('src="assets/search.png"', f'src="{search}"')
    html = html.replace('src="assets/atlas.png"', f'src="{atlas}"')
    (HERE / "report_final.html").write_text(html)
    print("wrote report_final.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
