import os
import re
import sympy as sp
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

# =====================================================
# CONFIG
# =====================================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\10119145\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config("Matric Math Master", layout="wide", page_icon="🎓")

# =====================================================
# OCR FUNCTIONS
# =====================================================
def preprocess_image(pil_image):
    img = np.array(pil_image.convert("L"))
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return img_bin

def ocr_with_exponents(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    result, prev_bottom, prev_text = "", 0, ""
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        top, height = data["top"][i], data["height"][i]
        if prev_text and top + height < prev_bottom - 5:
            result += "^" + text
        else:
            if prev_text and re.match(r"[a-zA-Z]", prev_text) and re.match(r"\d", text):
                result += "*" + text
            else:
                result += text
        prev_bottom = top + height
        prev_text = text
    return result.replace(" ", "").replace("\n", "")

def clean_for_sympy(text):
    text = re.sub(r"([a-zA-Z])(\d+)", r"\1^\2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)
    return text

def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

# =====================================================
# SESSION STATE
# =====================================================
if "learner" not in st.session_state:
    st.session_state.learner = {"name": "", "solved": 0, "marks": 0}

if "copied_text" not in st.session_state:
    st.session_state.copied_text = ""

# =====================================================
# PRACTICE QUESTIONS (FULL – PAPER 1 & 2)
# =====================================================
practice_data = {
"Paper 1": {
"Algebra": [
{"question": r"\text{Solve for } x:\; x^2 - 5x + 6 = 0",
 "solution_steps":[
 r"(x-2)(x-3)=0 \quad (1)",
 r"x-2=0 \;\text{or}\; x-3=0 \quad (1)",
 r"x=2 \;\text{or}\; x=3 \quad (1)"
 ],
 "final_answer": r"x=2 \;\text{or}\; x=3",
 "marks":3},
{"question": r"\text{Solve for } x:\; 3x^2=12",
 "solution_steps":[
 r"x^2=4 \quad (1)",
 r"x=\pm2 \quad (2)"
 ],
 "final_answer": r"x=\pm2",
 "marks":3}
],
"Sequences": [
{"question": r"\text{Find the 10th term of } 3,7,11,\dots",
 "solution_steps":[
 r"a=3,\; d=4 \quad (1)",
 r"T_n=a+(n-1)d \quad (1)",
 r"T_{10}=39 \quad (1)"
 ],
 "final_answer": r"39",
 "marks":3}
],
"Financial Mathematics": [
{"question": r"\text{Find } A \text{ if } P=1000,\; i=10\%,\; n=2",
 "solution_steps":[
 r"A=P(1+i)^n \quad (1)",
 r"A=1000(1.1)^2=1210 \quad (2)"
 ],
 "final_answer": r"1210",
 "marks":3}
],
"Calculus": [
{"question": r"\text{Differentiate } f(x)=3x^2",
 "solution_steps":[
 r"\frac{d}{dx}(3x^2)=6x \quad (3)"
 ],
 "final_answer": r"6x",
 "marks":3}
]
},
"Paper 2": {
"Analytical Geometry": [
{"question": r"\text{Find the distance between } A(1,2), B(4,6)",
 "solution_steps":[
 r"d=\sqrt{(4-1)^2+(6-2)^2}=5 \quad (3)"
 ],
 "final_answer": r"5",
 "marks":3}
],
"Trigonometry": [
{"question": r"\text{Solve } \sin x=\frac12,\; 0^\circ\le x\le360^\circ",
 "solution_steps":[
 r"x=30^\circ,\;150^\circ \quad (3)"
 ],
 "final_answer": r"30^\circ,\;150^\circ",
 "marks":3}
],
"Statistics & Probability": [
{"question": r"\text{Find the mean of } 2,4,6,8",
 "solution_steps":[
 r"\bar{x}=\frac{20}{4}=5 \quad (3)"
 ],
 "final_answer": r"5",
 "marks":3}
]
}
}

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🎓 Matric Math Master")
mode = st.sidebar.radio(
    "Choose Mode",
    ["📚 Past Papers (PDF)",
     "📷 OCR Question",
     "🧮 AI Step-by-Step Solver",
     "📝 Practice Questions",
     "🎯 Learner Profile",
     "📏 Formula Sheet"]
)

# =====================================================
# PRACTICE QUESTIONS
# =====================================================
if mode=="📝 Practice Questions":
    st.title("📝 Practice Questions")
    paper = st.selectbox("Select Paper", list(practice_data.keys()))
    topic = st.selectbox("Select Topic", list(practice_data[paper].keys()))
    questions = practice_data[paper][topic]
    q_numbers = [f"Q{i+1}" for i in range(len(questions))]
    q_selected = st.selectbox("Select Question Number", q_numbers)
    q_data = questions[q_numbers.index(q_selected)]

    st.markdown(f"### {q_selected}")
    st.latex(q_data["question"])
    st.text_input("Attempt your answer first:")

    if st.button("Show Solution"):
        st.markdown("### ✏️ Step-by-Step Solution")
        for step in q_data["solution_steps"]:
            st.latex(step)
        st.success("Final Answer")
        st.latex(q_data["final_answer"])
        st.info(f"Total Marks: {q_data['marks']}")
        st.session_state.learner["solved"] += 1
        st.session_state.learner["marks"] += q_data["marks"]

# =====================================================
# AI SOLVER (FULL PAPER 1 & PAPER 2 LOGIC)
# =====================================================
elif mode=="🧮 AI Step-by-Step Solver":
    st.title("🧮 AI Step-by-Step Solver")

    paper = st.selectbox("Select Paper", ["Paper 1", "Paper 2"])

    if paper == "Paper 1":
        topic = st.selectbox(
            "Topic",
            ["Algebra", "Sequences", "Financial Mathematics", "Calculus", "Functions & Graphs"]
        )
    else:
        topic = st.selectbox(
            "Topic",
            ["Analytical Geometry", "Trigonometry", "Statistics", "Probability"]
        )

    question = st.text_input("Enter your expression:", st.session_state.copied_text)
    x = sp.symbols("x")

    if st.button("Solve") and question:
        try:
            # ------------------------------
            # CLEAN INPUT (RHS ONLY)
            # ------------------------------
            if "=" in question:
                question = question.split("=")[1]

            expr = sp.sympify(question)

            st.markdown("### 🔹 Interpreted Expression")
            st.latex(expr)

            # =====================================================
            # PAPER 1
            # =====================================================
            if topic == "Algebra":
                st.markdown("### ✏️ Algebra Solution")
                st.latex(sp.latex(expr) + "=0")

                solutions = sp.solve(expr, x)
                for sol in solutions:
                    st.latex(r"x=" + sp.latex(sol))

            elif topic == "Sequences":
                st.markdown("### 🔢 Arithmetic Sequence")

                a = st.number_input("First term (a)", 1.0)
                d = st.number_input("Common difference (d)", 1.0)
                n = st.number_input("Term number (n)", 5, step=1)

                Tn = a + (n - 1) * d

                st.latex(r"T_n = a + (n-1)d")
                st.latex(rf"T_{{{int(n)}}} = {Tn}")

            elif topic == "Financial Mathematics":
                st.markdown("### 💰 Compound Interest")

                P = st.number_input("Principal (P)", 1000.0)
                i = st.number_input("Interest rate (%)", 10.0) / 100
                n = st.number_input("Time (years)", 2.0)

                A = P * (1 + i) ** n

                st.latex(r"A = P(1+i)^n")
                st.latex(rf"A = {round(A,2)}")

            elif topic == "Calculus":
                st.markdown("### 📐 Differentiation")

                derivative = sp.diff(expr, x)

                st.latex(r"\frac{dy}{dx} = " + sp.latex(derivative))

            elif topic == "Functions & Graphs":
                st.markdown("### 📈 Function Graph")

                f = sp.lambdify(x, expr, "numpy")
                xs = np.linspace(-10, 10, 400)

                fig, ax = plt.subplots()
                ax.plot(xs, f(xs))
                ax.axhline(0)
                ax.axvline(0)
                ax.grid(True)

                st.pyplot(fig, use_container_width=True)
                st.latex(r"y=" + sp.latex(expr))

            # =====================================================
            # PAPER 2
            # =====================================================
            elif topic == "Analytical Geometry":
                st.markdown("### 📏 Distance Between Two Points")

                x1 = st.number_input("x₁", 1.0)
                y1 = st.number_input("y₁", 2.0)
                x2 = st.number_input("x₂", 4.0)
                y2 = st.number_input("y₂", 6.0)

                d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                st.latex(r"d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}")
                st.latex(rf"d={round(d,2)}")

            elif topic == "Trigonometry":
                st.markdown("### 📐 Trigonometric Ratios")

                angle = st.number_input("Angle (degrees)", 30.0)
                rad = np.deg2rad(angle)

                st.latex(rf"\sin({angle}^\circ) = {round(np.sin(rad),3)}")
                st.latex(rf"\cos({angle}^\circ) = {round(np.cos(rad),3)}")
                st.latex(rf"\tan({angle}^\circ) = {round(np.tan(rad),3)}")

            elif topic == "Statistics":
                st.markdown("### 📊 Mean Calculation")

                data = st.text_input("Enter data (comma-separated)", "2,4,6,8")
                values = list(map(float, data.split(",")))

                mean = np.mean(values)

                st.latex(r"\bar{x}=\frac{\sum x}{n}")
                st.latex(rf"\bar{{x}}={mean}")

            elif topic == "Probability":
                st.markdown("### 🎲 Probability")

                favourable = st.number_input("Favourable outcomes", 1)
                total = st.number_input("Total outcomes", 6)

                prob = favourable / total

                st.latex(r"P(E)=\frac{n(E)}{n(S)}")
                st.latex(rf"P(E)={round(prob,3)}")

        except Exception as e:
            st.error("Invalid expression or input")
            st.caption(str(e))

# =====================================================
# OCR
# =====================================================
elif mode=="📷 OCR Question":
    st.title("📷 OCR Question")
    img_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, use_container_width=True)
        raw = ocr_with_exponents(preprocess_image(img))
        cleaned = clean_for_sympy(raw)
        st.code(cleaned)
        if st.button("Transfer to Solver"):
            st.session_state.copied_text = cleaned

# =====================================================
# PDF
# =====================================================
elif mode=="📚 Past Papers (PDF)":
    st.title("📚 PDF Extractor")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf:
        text = extract_pdf_text(pdf)
        edited = st.text_area("Extracted Text", text, height=300)
        if st.button("Transfer to Solver"):
            st.session_state.copied_text = edited

# =====================================================
# PROFILE
# =====================================================
elif mode=="🎯 Learner Profile":
    st.title("🎯 Learner Profile")
    st.metric("Solved", st.session_state.learner["solved"])
    st.metric("Marks", st.session_state.learner["marks"])

# =====================================================
# FORMULA SHEET
# =====================================================
else:
    st.title("📏 Complete Matric Formula Sheet")
    st.info("Grouped according to NSC Papers")


    col1, col2 = st.columns(2)

    with col1:
        st.header("📑 Paper 1")
        with st.expander("Algebra & Sequences", expanded=True):
            st.latex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
            st.latex(r"T_n = a + (n-1)d")
            st.latex(r"S_n = \frac{n}{2}[2a + (n-1)d]")
            st.latex(r"T_n = ar^{n-1}")
            st.latex(r"S_\infty = \frac{a}{1-r}")

        with st.expander("Financial Mathematics"):
            st.latex(r"A = P(1 + i)^n")
            st.latex(r"A = P(1 + ni)")
            st.latex(r"F = \frac{x[(1+i)^n - 1]}{i}")
            st.latex(r"P = \frac{x[1-(1+i)^{-n}]}{i}")

        with st.expander("Calculus"):
            st.latex(r"f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}")
            st.latex(r"\frac{d}{dx}[x^n]=nx^{n-1}")

    with col2:
        st.header("📑 Paper 2")
        with st.expander("Analytical Geometry", expanded=True):
            st.latex(r"d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}")
            st.latex(r"(x-a)^2+(y-b)^2=r^2")

        with st.expander("Trigonometry"):
            st.latex(r"\frac{a}{\sin A}=\frac{b}{\sin B}")
            st.latex(r"a^2=b^2+c^2-2bc\cos A")

        with st.expander("Statistics & Probability"):
            st.latex(r"\bar{x}=\frac{\sum x}{n}")
            st.latex(r"\sigma^2=\frac{\sum(x-\bar{x})^2}{n}")
            st.latex(r"P(A)=\frac{n(A)}{n(S)}")
