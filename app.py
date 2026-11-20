"""Bilingual Streamlit app for supportive student-performance screening."""

from __future__ import annotations

import json
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, normalize_frame

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
DATA_PATH = ROOT / "data" / "xAPI-Edu-Data.csv"

FIELD_LABELS = {
    "en": {
        "StageID": "Education stage",
        "GradeID": "Grade",
        "SectionID": "Section",
        "Topic": "Subject",
        "Semester": "Semester",
        "Relation": "Parent or guardian",
        "ParentAnsweringSurvey": "Parent answered the survey?",
        "ParentschoolSatisfaction": "Parent satisfaction with school",
        "StudentAbsenceDays": "Absence category this term",
        "raisedhands": "Times raised hand",
        "VisITedResources": "Learning resources visited",
        "AnnouncementsView": "Announcements viewed",
        "Discussion": "Discussion activity",
    },
    "ar": {
        "StageID": "المرحلة الدراسية",
        "GradeID": "الصف",
        "SectionID": "الشعبة",
        "Topic": "المادة",
        "Semester": "الفصل الدراسي",
        "Relation": "وليّ الأمر المسؤول",
        "ParentAnsweringSurvey": "هل أجاب وليّ الأمر على الاستبيان؟",
        "ParentschoolSatisfaction": "رضا وليّ الأمر عن المدرسة",
        "StudentAbsenceDays": "الغياب خلال الفصل",
        "raisedhands": "مرات رفع اليد",
        "VisITedResources": "مرات زيارة موارد التعلّم",
        "AnnouncementsView": "مشاهدات الإعلانات",
        "Discussion": "المشاركة في النقاش",
    },
}

OPTION_LABELS = {
    "en": {
        "lowerlevel": "Primary school",
        "MiddleSchool": "Middle school",
        "HighSchool": "High school",
        "F": "First semester",
        "S": "Second semester",
        "Father": "Father",
        "Mum": "Mother",
        "Yes": "Yes",
        "No": "No",
        "Good": "Satisfied",
        "Bad": "Dissatisfied",
        "Under-7": "Fewer than 7 days",
        "Above-7": "More than 7 days",
    },
    "ar": {
        "lowerlevel": "المرحلة الابتدائية",
        "MiddleSchool": "المرحلة الإعدادية",
        "HighSchool": "المرحلة الثانوية",
        "F": "الفصل الأول",
        "S": "الفصل الثاني",
        "Father": "الأب",
        "Mum": "الأم",
        "Yes": "نعم",
        "No": "لا",
        "Good": "راضٍ",
        "Bad": "غير راضٍ",
        "Under-7": "أقل من 7 أيام",
        "Above-7": "أكثر من 7 أيام",
        "Arabic": "اللغة العربية",
        "Biology": "الأحياء",
        "Chemistry": "الكيمياء",
        "English": "اللغة الإنجليزية",
        "French": "اللغة الفرنسية",
        "Geology": "الجيولوجيا",
        "History": "التاريخ",
        "IT": "تقنية المعلومات",
        "Math": "الرياضيات",
        "Quran": "القرآن الكريم",
        "Science": "العلوم",
        "Spanish": "اللغة الإسبانية",
    },
}

TEXT = {
    "en": {
        "page_caption": "Exploratory academic screening · Human support only",
        "tabs": ["Single student", "Batch CSV", "About the model"],
        "single_title": "Initial performance estimate",
        "single_intro": "Enter the available school and participation indicators. Use the estimate to start a supportive conversation, never as an automated decision.",
        "unknown": "Unknown",
        "engagement_title": "Participation indicators (0–100)",
        "submit": "Show estimate",
        "predicted": "Model estimate",
        "score": "Model score for this class",
        "prediction_caution": "This score is sigmoid-calibrated using training folds, but remains uncertain and is not a reliable probability of student success or difficulty. Review the situation with the student and family and use other evidence.",
        "suggestions_title": "Ideas for supportive follow-up",
        "below_median": "The {feature} indicator is below the sample median. Ask whether the student needs help accessing resources or participating.",
        "absence": "Review the reasons for absences with the student and family, then agree on a suitable follow-up step.",
        "no_suggestions": "No clear signal appears in these descriptive rules. Continue regular check-ins and listen to the student.",
        "suggestion_caveat": "These descriptive prompts come from this sample; they do not show that a factor caused the outcome.",
        "batch_title": "Score a CSV file",
        "batch_intro": "Upload a CSV with the school and participation fields. Do not include names or student ID numbers.",
        "upload": "CSV file to score",
        "read_error": "Could not read the file: {error}",
        "missing": "Required columns are missing: {columns}",
        "template": "Download input template",
        "invalid": "Review the numeric values: {invalid} rows contain missing or non-numeric values, and {range} rows fall outside 0–100.",
        "results": "Download predictions",
        "batch_caveat": "Scores are calibrated internally but remain uncertain. Do not use them alone to rank students or restrict educational opportunities.",
        "model_title": "About the model and its limits",
        "model_intro": "The model assigns a low (L), medium (M), or high (H) performance class using school stage, grade, subject, attendance category, parent survey responses, and participation indicators. Gender, nationality, and birthplace are excluded from prediction.",
        "selected_model": "Model selected by cross-validation",
        "holdout_f1": "Macro F1 on a separate test set",
        "rows": "Records used",
        "model_caution": "The dataset is small and comes from one learning-system context. Results may not generalize to another school or country. Other inputs can still reflect unequal circumstances. Never use this estimate as an automated decision.",
        "audit_title": "Descriptive review by gender on the test set",
        "audit_caveat": "Groups are small. This initial check is not proof of fairness or absence of bias.",
        "missing_model": "The trained model was not found. Run train.py first.",
        "language": "Language",
    },
    "ar": {
        "page_caption": "تصنيف استكشافي للأداء الدراسي · للمساندة البشرية فقط",
        "tabs": ["تقدير طالب", "ملف CSV", "عن النموذج"],
        "single_title": "تقدير أولي لمستوى الأداء",
        "single_intro": "أدخل بيانات الدراسة والمشاركة المتاحة. استخدم التقدير لبدء حوار داعم، وليس لاتخاذ قرار آلي بشأن الطالب.",
        "unknown": "غير معروف",
        "engagement_title": "مؤشرات المشاركة (من 0 إلى 100)",
        "submit": "عرض التقدير",
        "predicted": "تقدير النموذج للفئة",
        "score": "درجة النموذج لهذه الفئة",
        "prediction_caution": "الدرجة خضعت لمعايرة sigmoid على طيات التدريب، لكنها تظل غير يقينية ولا تمثل احتمالًا موثوقًا لنجاح الطالب أو تعثره. راجع الحالة مع الطالب والأسرة واستخدم مصادر أخرى للمعلومة.",
        "suggestions_title": "أفكار لمتابعة داعمة",
        "below_median": "مؤشر **{feature}** أقل من وسيط العينة؛ اسأل الطالب إن كان يحتاج مساعدة للوصول للموارد أو للمشاركة.",
        "absence": "راجع أسباب الغياب مع الطالب والأسرة، ثم اتفقوا على خطوة متابعة مناسبة.",
        "no_suggestions": "لا تظهر إشارة واضحة من هذه القواعد الوصفية؛ استمر في المتابعة المعتادة واستمع للطالب.",
        "suggestion_caveat": "هذه أفكار وصفية مستندة إلى العينة، ولا تثبت أن أي عامل سبب النتيجة.",
        "batch_title": "تقدير ملف CSV",
        "batch_intro": "ارفع ملف CSV يحتوي على بيانات الدراسة ومؤشرات المشاركة. لا تضع أسماء الطلاب أو أرقامهم.",
        "upload": "ملف CSV للتقدير",
        "read_error": "تعذّرت قراءة الملف: {error}",
        "missing": "الأعمدة المطلوبة غير موجودة: {columns}",
        "template": "تنزيل قالب الإدخال",
        "invalid": "راجع القيم الرقمية: {invalid} صفوف بها قيم ناقصة أو غير رقمية، و{range} صفوف خارج النطاق 0–100.",
        "results": "تنزيل نتائج التقدير",
        "batch_caveat": "الدرجات معايرة داخليًا لكنها غير يقينية. لا تستخدمها وحدها لترتيب الطلاب أو حرمانهم من فرصة تعليمية.",
        "model_title": "عن النموذج وحدوده",
        "model_intro": "يصنف النموذج الأداء إلى منخفض (L) أو متوسط (M) أو مرتفع (H)، باستخدام المرحلة والصف والمادة وفئة الغياب ومؤشرات المشاركة وإجابات استبيان وليّ الأمر. استُبعد النوع والجنسية ومكان الميلاد من التنبؤ.",
        "selected_model": "النموذج المختار بالتحقق المتقاطع",
        "holdout_f1": "Macro F1 على مجموعة اختبار منفصلة",
        "rows": "عدد السجلات",
        "model_caution": "العينة صغيرة ومصدرها سياق تعليمي واحد؛ قد لا تنطبق النتائج على مدرسة أو بلد آخر. وقد تعكس المدخلات ظروفًا أو فرصًا غير متساوية. لا تستخدم التقدير لاتخاذ قرار آلي.",
        "audit_title": "مراجعة وصفية حسب النوع في مجموعة الاختبار",
        "audit_caveat": "المجموعات صغيرة؛ هذه مراجعة أولية وليست إثباتًا للإنصاف أو خلوّ النموذج من التحيّز.",
        "missing_model": "لم يتم العثور على النموذج المدرّب. شغّل train.py أولًا.",
        "language": "اللغة",
    },
}


st.set_page_config(page_title="Student Learning Support Indicator", page_icon="📘", layout="wide")


@st.cache_resource
def load_model():
    model_file = ARTIFACT_DIR / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(TEXT["en"]["missing_model"])
    model = joblib.load(model_file)
    metadata_file = ARTIFACT_DIR / "model_metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    return model, metadata


@st.cache_data
def load_training_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return normalize_frame(pd.read_csv(DATA_PATH))


def get_category_options(data: pd.DataFrame, column: str) -> list[str]:
    if data.empty or column not in data.columns:
        return []
    return sorted(data[column].dropna().astype(str).unique().tolist())


def set_text_direction(language: str) -> None:
    direction = "rtl" if language == "ar" else "ltr"
    alignment = "right" if language == "ar" else "left"
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{ direction: {direction}; text-align: {alignment}; }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] > *,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] :is(h1,h2,h3,h4,p,ul,ol,li,blockquote),
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stTabs"],
    [data-testid="stAppViewContainer"] [role="tablist"],
    [data-testid="stAppViewContainer"] [role="tab"] {{
        direction: {direction} !important;
        text-align: {alignment} !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] ul,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] ol {{
        padding-inline-start: 1.5rem !important;
        padding-inline-end: 0 !important;
        list-style-position: outside !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li {{
        display: list-item !important;
        unicode-bidi: plaintext;
    }}
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea {{ direction: {direction}; text-align: {alignment}; }}
    [data-testid="stAppViewContainer"] button {{ direction: {direction}; }}
    .model-meta-row {{
        display: flex;
        align-items: flex-start;
        gap: .45rem;
        width: 100%;
        margin: .45rem 0;
    }}
    .model-meta-row[dir="rtl"] {{ direction: rtl; text-align: right; }}
    .model-meta-row[dir="ltr"] {{ direction: ltr; text-align: left; }}
    .model-meta-bullet {{ flex: 0 0 auto; }}
    .model-meta-copy {{ flex: 1 1 auto; min-width: 0; }}
    .model-meta-row[dir="rtl"] .model-meta-copy {{ direction: rtl; text-align: right; }}
    .model-meta-row[dir="ltr"] .model-meta-copy {{ direction: ltr; text-align: left; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def show_bilingual_header() -> None:
    st.markdown(
        """
        <div style="display:flex;direction:ltr;gap:2rem;align-items:flex-start;margin:.5rem 0 1rem">
          <div dir="ltr" style="flex:1;text-align:left">
            <h1>📘 Student Learning Support Indicator</h1>
            <p>Exploratory academic screening · Human support only</p>
          </div>
          <div dir="rtl" style="flex:1;text-align:right">
            <h1>📘 مؤشر دعم تعلم الطالب</h1>
            <p>تصنيف استكشافي للأداء الدراسي · للمساندة البشرية فقط</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_single_prediction(model, data: pd.DataFrame, language: str) -> None:
    text = TEXT[language]
    st.subheader(text["single_title"])
    st.caption(text["single_intro"])
    with st.form("single_prediction"):
        left, right = st.columns(2)
        values = {}
        for index, column in enumerate(CATEGORICAL_FEATURES):
            options = get_category_options(data, column) or [text["unknown"]]
            target = left if index % 2 == 0 else right
            values[column] = target.selectbox(
                FIELD_LABELS[language][column],
                options,
                format_func=lambda value, labels=OPTION_LABELS[language]: labels.get(str(value), str(value)),
                key=f"single_{language}_{column}",
            )
        st.markdown(f"**{text['engagement_title']}**")
        numeric_cols = st.columns(4)
        for column, target in zip(NUMERIC_FEATURES, numeric_cols):
            values[column] = target.number_input(
                FIELD_LABELS[language][column], min_value=0, max_value=100, value=30, step=1
            )
        submitted = st.form_submit_button(text["submit"], type="primary")

    if submitted:
        row = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
        prediction = str(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        score = float(probabilities[list(model.classes_).index(prediction)])
        labels = {"L": "Low" if language == "en" else "منخفض", "M": "Medium" if language == "en" else "متوسط", "H": "High" if language == "en" else "مرتفع"}
        first, second = st.columns([1, 2])
        first.metric(text["predicted"], labels.get(prediction, prediction))
        first.metric(text["score"], f"{score:.0%}")
        second.info(text["prediction_caution"])
        show_support_suggestions(values, data, language)


def show_support_suggestions(values: dict, data: pd.DataFrame, language: str) -> None:
    text = TEXT[language]
    suggestions = []
    if not data.empty:
        for feature in NUMERIC_FEATURES:
            median = pd.to_numeric(data[feature], errors="coerce").median()
            if pd.notna(median) and values[feature] < median:
                suggestions.append(
                    text["below_median"].format(feature=FIELD_LABELS[language][feature])
                )
    if values["StudentAbsenceDays"] == "Above-7":
        suggestions.append(text["absence"])
    if not suggestions:
        suggestions.append(text["no_suggestions"])
    st.markdown(f"**{text['suggestions_title']}**")
    for suggestion in suggestions[:4]:
        st.markdown(f"- {suggestion}")
    st.caption(text["suggestion_caveat"])


def make_batch_predictions(model, language: str) -> None:
    text = TEXT[language]
    st.subheader(text["batch_title"])
    st.write(text["batch_intro"])
    uploaded = st.file_uploader(text["upload"], type=["csv"], key="batch_csv")
    if uploaded is None:
        return
    try:
        frame = normalize_frame(pd.read_csv(uploaded))
    except Exception as error:
        st.error(text["read_error"].format(error=error))
        return
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        st.error(text["missing"].format(columns=", ".join(missing)))
        st.download_button(
            text["template"],
            data=pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(index=False).encode("utf-8-sig"),
            file_name="student_input_template.csv",
            mime="text/csv",
        )
        return
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_numeric = frame[NUMERIC_FEATURES].isna().any(axis=1)
    out_of_range = ((frame[NUMERIC_FEATURES] < 0) | (frame[NUMERIC_FEATURES] > 100)).any(axis=1)
    if invalid_numeric.any() or out_of_range.any():
        st.error(
            text["invalid"].format(
                invalid=int(invalid_numeric.sum()), range=int(out_of_range.sum())
            )
        )
        return
    predictions = model.predict(frame[FEATURE_COLUMNS])
    scores = model.predict_proba(frame[FEATURE_COLUMNS])
    labels_en = {"L": "Low", "M": "Medium", "H": "High"}
    labels_ar = {"L": "منخفض", "M": "متوسط", "H": "مرتفع"}
    result = frame.copy()
    result["predicted_class"] = predictions
    result["predicted_level_en"] = [labels_en.get(str(label), str(label)) for label in predictions]
    result["predicted_level_ar"] = [labels_ar.get(str(label), str(label)) for label in predictions]
    result["model_score_for_predicted_class"] = scores.max(axis=1).round(4)
    st.dataframe(result.head(100), width="stretch")
    st.download_button(
        text["results"],
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="student_performance_predictions.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(text["batch_caveat"])


def show_model_card(metadata: dict, language: str) -> None:
    text = TEXT[language]
    st.subheader(text["model_title"])
    st.write(text["model_intro"])
    if metadata:
        direction = "rtl" if language == "ar" else "ltr"
        selected_model = html.escape(str(metadata.get("selected_model", "N/A")))
        holdout_f1 = float(metadata.get("holdout_macro_f1", 0))
        rows = html.escape(str(metadata.get("rows", "N/A")))
        metric_rows = [
            (text["selected_model"], selected_model),
            (text["holdout_f1"], f"{holdout_f1:.3f}"),
            (text["rows"], rows),
        ]
        for label, value in metric_rows:
            st.markdown(
                f'<div class="model-meta-row" dir="{direction}">'
                f'<span class="model-meta-bullet" aria-hidden="true">•</span>'
                f'<span class="model-meta-copy"><strong>{html.escape(label)}:</strong> '
                f'{value}</span></div>',
                unsafe_allow_html=True,
            )
    st.warning(text["model_caution"])
    audit_path = ARTIFACT_DIR / "gender_audit.csv"
    if audit_path.exists():
        st.markdown(f"**{text['audit_title']}**")
        audit = pd.read_csv(audit_path)
        if language == "ar":
            audit = audit.rename(
                columns={
                    "gender_group": "المجموعة",
                    "n": "العدد",
                    "accuracy": "الدقة",
                    "macro_f1": "Macro F1",
                    "low_class_recall": "استدعاء الفئة المنخفضة",
                    "low_class_count": "عدد الفئة المنخفضة",
                }
            )
        st.dataframe(audit, width="stretch", hide_index=True)
        st.caption(text["audit_caveat"])


language_choice = st.selectbox("Language / اللغة", ["English", "العربية"], index=0)
language = "ar" if language_choice == "العربية" else "en"
set_text_direction(language)
show_bilingual_header()

try:
    model, metadata = load_model()
except Exception as error:
    st.error(f"{TEXT[language]['missing_model']} ({error})")
    st.stop()

training_data = load_training_data()
tab_single, tab_batch, tab_about = st.tabs(TEXT[language]["tabs"])
with tab_single:
    make_single_prediction(model, training_data, language)
with tab_batch:
    make_batch_predictions(model, language)
with tab_about:
    show_model_card(metadata, language)
