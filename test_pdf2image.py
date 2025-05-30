from pdf2image import convert_from_path

# Path to your test image-based PDF
pdf_path = "uploads/test_image_pdf.pdf"  # 👈 Replace this with your actual PDF file name
poppler_path = r"C:\poppler-24.08.0\Library\bin"  # ✅ Correct poppler path

try:
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    print("✅ Total pages extracted:", len(pages))
    pages[0].save("test_page_1.png", "PNG")
    print("✅ First page saved as test_page_1.png")
except Exception as e:
    print("❌ Error:", e)
