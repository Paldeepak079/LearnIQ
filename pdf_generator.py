"""
PDF Report Generator for LearnIQ
Creates professional student profile reports
"""

from fpdf import FPDF
import datetime


class StudentReportPDF(FPDF):
    """Custom PDF class for student reports"""
    
    def header(self):
        """Page header"""
        self.set_fill_color(26, 28, 44)
        self.rect(0, 0, 210, 40, 'F')
        
        self.set_font('Arial', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, 'LearnIQ', 0, 1, 'C')
        
        self.set_font('Arial', 'I', 12)
        self.set_text_color(168, 162, 255)
        self.cell(0, 10, 'Learning Pattern Analysis Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        """Page footer"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} | Page {self.page_no()}', 0, 0, 'C')
    
    def section_title(self, title):
        """Add a section title"""
        self.set_font('Arial', 'B', 16)
        self.set_text_color(76, 81, 191)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def info_row(self, label, value):
        """Add an information row"""
        self.set_font('Arial', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.cell(60, 8, f'{label}:', 0, 0)
        
        self.set_font('Arial', '', 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, str(value), 0, 1)


def generate_student_report(student_data, pattern_info, prediction_info=None):
    """
    Generate a PDF report for a student
    
    Args:
        student_data: Student record (pandas Series)
        pattern_info: Dictionary with pattern analysis
        prediction_info: Optional prediction results
    
    Returns:
        PDF bytes
    """
    pdf = StudentReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Student Information
    pdf.section_title('Student Profile')
    pdf.info_row('Age', f"{student_data.get('age', 'N/A')} years")
    pdf.info_row('Study Time', f"{student_data.get('studytime', 'N/A')} hours/week")
    pdf.info_row('Absences', f"{student_data.get('absences', 'N/A')} days")
    pdf.info_row('Previous Failures', student_data.get('failures', 'N/A'))
    pdf.ln(5)
    
    # Academic Performance
    pdf.section_title('Academic Performance')
    pdf.info_row('Period 1 Grade (G1)', student_data.get('G1', 'N/A'))
    pdf.info_row('Period 2 Grade (G2)', student_data.get('G2', 'N/A'))
    pdf.info_row('Final Grade (G3)', student_data.get('G3', 'N/A'))
    pdf.ln(5)
    
    # Learning Pattern
    pdf.section_title('Learning Pattern Analysis')
    pdf.info_row('Assigned Pattern', pattern_info.get('pattern', 'N/A'))
    pdf.info_row('Risk Level', pattern_info.get('risk_level', 'N/A'))
    
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, pattern_info.get('explanation', ''))
    pdf.ln(5)
    
    # Behavioral Metrics
    pdf.section_title('Behavioral Indicators')
    pdf.info_row('Engagement Score', f"{pattern_info.get('engagement_score', 'N/A'):.2f}")
    pdf.info_row('Consistency Index', f"{pattern_info.get('consistency_index', 'N/A'):.2f}")
    pdf.info_row('Performance Trend', f"{pattern_info.get('performance_trend', 'N/A'):.2f}")
    pdf.ln(5)
    
    # Prediction (if available)
    if prediction_info:
        pdf.section_title('Grade Forecast')
        pdf.info_row('Predicted Final Grade', prediction_info.get('predicted_grade', 'N/A'))
        pdf.info_row('Confidence Range', 
                     f"{prediction_info.get('confidence_lower', 'N/A')} - {prediction_info.get('confidence_upper', 'N/A')}")
        pdf.info_row('Pass Probability', f"{prediction_info.get('pass_probability', 'N/A')}%")
        pdf.ln(5)
    
    # Recommendations
    pdf.section_title('Recommended Interventions')
    recommendations = pattern_info.get('recommendations', [])
    
    pdf.set_font('Arial', '', 10)
    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 6, f"{i}. {rec}")
        pdf.ln(2)
    
    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')
