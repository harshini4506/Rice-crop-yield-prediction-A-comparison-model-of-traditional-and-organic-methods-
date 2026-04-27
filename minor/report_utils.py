from fpdf import FPDF

def generate_pdf(normal_yield, organic_yield, improvement, fertilizer):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Yield Optimization Report", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Traditional Yield: {normal_yield:.4f} tons/ha", ln=True)
    pdf.cell(0, 10, f"Organic ML Yield: {organic_yield:.4f} tons/ha", ln=True)
    pdf.cell(0, 10, f"Yield Improvement: {improvement:.4f} tons/ha", ln=True)
    pdf.cell(0, 10, f"Recommended Organic Fertilizer: {fertilizer:.4f} kg/ha", ln=True)

    file_path = "Yield_Report.pdf"
    pdf.output(file_path)

    return file_path
