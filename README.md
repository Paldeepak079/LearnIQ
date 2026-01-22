# Learning Pattern Analysis & Teaching Guidance System

**An interpretable system for educators to understand student learning behaviors and provide personalized teaching interventions**

Built for the **Education | Behavioral Analytics | Personalization** hackathon theme.

---

## Problem Statement

Students differ widely in how they absorb information, engage in class, and respond to instruction. Teachers rarely have tools that reveal these differences at scale or suggest targeted interventions.

This system enables educators to:
- **Identify** distinct learning behavior patterns
- **Explain** why each student fits a specific pattern
- **Detect** early warning signs of disengagement
- **Recommend** evidence-based teaching strategies

---

## Key Features

### 1. Pattern Discovery
- Unsupervised clustering (K-Means) to identify natural learning groupings
- Automatic optimal cluster determination using Silhouette Score
- Human-readable personas: *Consistent Achievers*, *At-Risk Learners*, *Emerging Stars*, etc.

### 2. Transparent Classification
- Feature importance calculation for each student
- Plain English explanations (not just predictions!)
- "Why does this student belong to this pattern?"

### 3. Early Disengagement Detection
- Multi-factor risk assessment system
- 3 risk levels: Normal, Watchlist, High Risk
- Flags based on attendance, engagement, performance trends

### 4. Teaching Guidance Engine
- Pattern-specific evidence-based strategies
- Personalized, actionable recommendations
- Non-punitive, growth-oriented approach

### 5. Interactive Dashboard
- Class overview with visualizations
- Individual student profiles
- Risk alert management
- Teaching strategy browser

---

## Dataset

Uses: `student_dataset.csv`

**Features** (34 total):
- **Academic**: Grades (G1, G2, G3), study time, failures
- **Behavioral**: Absences, activities, paid classes
- **Support**: School support, family support
- **Lifestyle**: Free time, going out, romantic relationships

**Students**: 6,395 across multiple schools

---

## Technology Stack

- **Python 3.x**
- **Machine Learning**: scikit-learn (KMeans, StandardScaler)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Analysis**: scipy

---

## Quick Start

### Installation

```bash
# Clone or download this repository
cd "Google Developer Groups on Campus Praxis 2.0"

# Install dependencies
pip install -r requirements.txt
```

### Running the System

#### Option 1: interactive Dashboard (Recommended)
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

#### Option 2: Test Individual Modules

**Data Processing:**
```bash
python data_processor.py
```

**Pattern Discovery:**
```bash
python pattern_discovery.py
```

**Risk Detection:**
```bash
python risk_detection.py
```

**Teaching Guidance:**
```bash
python teaching_guidance.py
```

---

## 📁 Project Structure

```
├── student_dataset.csv           # Student data
├── app.py                        # Streamlit dashboard (main app)
├── data_processor.py              # Data loading & feature engineering
├── pattern_discovery.py           # Clustering & persona assignment
├── explainability.py              # SHAP-style explanations
├── risk_detection.py              # Early warning system
├── teaching_guidance.py           # Intervention recommendations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## How It Works

### 1. Data Processing
- Load student data
- Handle missing values (median/mode imputation)
- Engineer 4 behavioral features:
  - **Engagement Score** (0-100): Study time + activities + extra classes
  - **Consistency Index** (0-100): Performance stability across G1, G2, G3
  - **Performance Trend** (-1 to 1): Improving/stable/declining
  - **Participation Stability** (0-100): Attendance + support systems

### 2. Pattern Discovery
- K-Means clustering on normalized features
- Optimal k selection (Elbow + Silhouette methods)
- Persona assignment based on cluster profiles:
  - **Consistent Achievers**: High engagement, stable performance
  - **Silent Performers**: Good grades, low visible engagement
  - **At-Risk Learners**: Low engagement, poor performance
  - **Irregular Learners**: High variability
  - **Emerging Stars**: Positive improvement trend
  - **Declining Performers**: Negative trend

### 3. Explainability
- For each student, calculate feature importance
- Identify top 3 factors influencing cluster assignment
- Generate plain English explanations
- Example: *"This student is a Consistent Achiever because of high engagement (85/100) and minimal absences"*

### 4. Risk Detection
- Multi-factor flagging system:
  - High absences (>10 days)
  - Low engagement (<30/100)
  - Performance volatility (consistency <50)
  - Declining trend (<-0.15)
  - Low participation (<35/100)
- Risk classification:
  - **Normal**: 0 flags
  - **Watchlist**: 1-2 flags
  - **High Risk**: 3+ flags

### 5. Teaching Recommendations
- Evidence-based strategies mapped to each pattern
- Contextual recommendations based on individual student data
- Priority levels based on risk
- Examples:
  - For *At-Risk*: Immediate intervention, study skills workshop, counseling referral
  - For *Consistent Achievers*: Advanced challenges, peer mentorship opportunities

---

## 📸 Dashboard Screenshots

The interactive dashboard includes:

1. **Class Overview**: Pattern distribution, risk metrics, behavioral charts
2. **Student Profiles**: Individual analysis with explanations and recommendations
3. **Risk Alerts**: Filterable list of students needing intervention
4. **Teaching Strategies**: Pattern-specific guidance
5. **About & Ethics**: System documentation and ethical guidelines

---

## Ethical Considerations

**This system is designed to ASSIST teacher judgment, not replace it.**

- **What We Do:**
- Focus on observable behaviors and performance
- Provide transparent explanations
- Suggest non-punitive, growth-oriented strategies
- Treat patterns as dynamic, not permanent labels

- **What We Don't Do:**
- Use demographic attributes (gender, address, family background) for clustering
- Make final decisions about students
- Replace teacher expertise
- Create permanent student categorizations

**Privacy**: No personally identifiable information (names, IDs) required for analysis.

---

## Educational Research Basis

This system draws on:
- **Differentiated Instruction** (Tomlinson, 2001)
- **Response to Intervention (RTI)** frameworks
- **Growth Mindset** principles (Dweck, 2006)
- **Early Warning Systems** in education (Allensworth & Easton, 2007)
- **Behavioral Analytics** for learning (Baker & Inventado, 2014)

---

## Hackathon Alignment

### Theme: Education | Behavioral Analytics | Personalization

- **Identification of learning tendencies**: Unsupervised clustering discovers 4-6 natural patterns
- **Explanation of pattern assignment**: Feature importance + plain English explanations
- **Early disengagement signals**: Multi-factor risk detection system
- **Teaching strategies aligned to patterns**: Evidence-based recommendations per pattern
- **Class & school-level summaries**: Dashboard provides aggregated insights

### Evaluation Criteria Met:

- **Practical usefulness**: Direct actionable guidance for teachers
- **Interpretability**: Every decision explained in plain language
- **Fairness**: No demographic bias, dynamic patterns
- **Creativity**: Novel combination of clustering + explainability + interventions

---

## 🔮 Future Enhancements

- **Temporal tracking**: Monitor pattern transitions over semesters
- **Intervention effectiveness**: A/B testing of recommended strategies
- **Teacher feedback loop**: Allow educators to rate recommendation usefulness
- **Multi-school comparison**: Benchmarking across institutions
- **Mobile app**: Accessible on tablets for classroom use

---

## Support

For questions or issues:
1. Review the **About & Ethics** section in the dashboard
2. Check module documentation in code comments
3. Run individual test scripts for debugging

---

## 📄 License

Educational use only. Created for hackathon demo purposes.

---

## Acknowledgments

Built for educators who want to better understand and support their students.

**Dataset**: Student Performance Data Set (UCI Machine Learning Repository)

---

**Remember**: Every student has potential. This tool helps you discover and nurture it.

© 2026 MADTech. All rights reserved.
