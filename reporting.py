"""
Reporting Module for Learning Pattern Analysis System

This module handles:
1. Generating PDF reports for individual students
2. Exporting class summaries (optional expansion)
"""

from fpdf import FPDF
import datetime
import os

class StudentReport(FPDF):
    def header(self):
        # Logo placeholder (optional)
        self.set_font('Arial', 'B', 20)
        self.set_text_color(102, 126, 234) # LearnIQ blue
        self.cell(0, 10, 'LearnIQ - Student Learning Profile', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Generated on: {datetime.date.today().strftime("%B %d, %Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(169, 169, 169)
        self.cell(0, 10, '© 2026 MADTech - For Educational Use Only', 0, 0, 'C')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')

def generate_pdf_report(student_data):
    """
    Generate a PDF report for a single student.
    
    Args:
        student_data (dict or pd.Series): Student info including pattern and risk
        
    Returns:
        bytearray: PDF content
    """
    pdf = StudentReport()
    pdf.add_page()
    
    # 1. Student Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, f'Student ID: {student_data.name if hasattr(student_data, "name") else "N/A"}', 1, 1, 'L', True)
    pdf.ln(5)
    
    # 2. Key Metrics Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(95, 10, 'Attribute', 1)
    pdf.cell(95, 10, 'Value', 1, 1)
    
    pdf.set_font('Arial', '', 11)
    metrics = [
        ('Learning Pattern', student_data['persona']),
        ('Risk Level', student_data['risk_level']),
        ('Engagement Score', f"{student_data['engagement_score']:.1f}/100"),
        ('Consistency Index', f"{student_data['consistency_index']:.1f}/100"),
        ('Participation Stability', f"{student_data['participation_stability']:.1f}/100"),
        ('Performance Trend', f"{student_data['performance_trend']:.3f}"),
        ('Recent Grades', f"G1: {student_data['G1']}, G2: {student_data['G2']}, G3: {student_data['G3']}")
    ]
    
    for attr, val in metrics:
        pdf.cell(95, 10, attr, 1)
        pdf.cell(95, 10, str(val), 1, 1)
    
    pdf.ln(10)
    
    # 3. Interpretation & Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Interpretive Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    # Extract explanation and recommendations text
    explanation = student_data.get('explanation', "No detailed assessment available.")
    # Strip some markdown characters for PDF
    clean_explanation = explanation.replace('**', '').replace('•', '-')
    pdf.multi_cell(0, 7, clean_explanation)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommended Teaching Strategies', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    recs_text = student_data.get('recommendations_text', "")
    if recs_text:
        # Just take the strategies part
        parts = recs_text.split('Recommended Strategies:')
        if len(parts) > 1:
            strategies = parts[1].split('Additional Context-Specific Actions:')[0].strip()
            clean_strategies = strategies.replace('**', '').replace('•', '-')
            pdf.multi_cell(0, 7, clean_strategies)
    
    return bytes(pdf.output())
