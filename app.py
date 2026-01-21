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
from sympy.solvers.inequalities import solve_univariate_inequality

# =====================================================
# CONFIG
# =====================================================
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\10119145\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Only set path locally
if os.name == "nt":
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
    st.session_state.learner = {"name": "", "solved": 0, "Marks": 0}

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
 r"(x-2)(x-3)=0 \quad (1 Mark)",
 r"x-2=0 \;\text{or}\; x-3=0 \quad (1 Mark)",
 r"x=2 \;\text{or}\; x=3 \quad (1 Mark)"
 ],
 "final_answer": r"x=2 \;\text{or}\; x=3",
 "Marks":3},
{"question": r"\text{Solve for } x:\; 3x^2=12",
 "solution_steps":[
 r"x^2=4 \quad (1 Mark)",
 r"x=\pm2 \quad (2 Marks)"
 ],
 "final_answer": r"x=\pm2",
 "Marks":3}

],
"Sequences": [
{"question": r"\text{Find the 10th term of } 3,7,11,\dots",
 "solution_steps":[
 r"a=3,\; d=4 \quad (1 Mark)",
 r"T_n=a+(n-1)d \quad (1 Mark)",
 r"T_{10}=39 \quad (1 Mark)"
 ],
 "final_answer": r"39",
 "Marks":3}
],
"Financial Mathematics": [
{"question": r"\text{Find } A \text{ if } P=1000,\; i=10\%,\; n=2",
 "solution_steps":[
 r"A=P(1+i)^n \quad (1 Mark)",
 r"A=1000(1.1)^2=1210 \quad (2 Marks)"
 ],
 "final_answer": r"1210",
 "Marks":3}
],
"Calculus": [
{"question": r"\text{Differentiate } f(x)=3x^2",
 "solution_steps":[
 r"\frac{d}{dx}(3x^2)=6x \quad (3 Marks)"
 ],
 "final_answer": r"6x",
 "Marks":3}
]
},
"Paper 2": {
"Analytical Geometry": [
{"question": r"\text{Find the distance between } A(1,2), B(4,6)",
 "solution_steps":[
 r"d=\sqrt{(4-1)^2+(6-2)^2}=5 \quad (3 Marks)"
 ],
 "final_answer": r"5",
 "Marks":3}
],
"Trigonometry": [
{"question": r"\text{Solve } \sin x=\frac12,\; 0^\circ\le x\le360^\circ",
 "solution_steps":[
 r"x=30^\circ,\;150^\circ \quad (3 Marks)"
 ],
 "final_answer": r"30^\circ,\;150^\circ",
 "Marks":3}
],
"Statistics & Probability": [
{"question": r"\text{Find the mean of } 2,4,6,8",
 "solution_steps":[
 r"\bar{x}=\frac{20}{4}=5 \quad (3 Marks)"
 ],
 "final_answer": r"5",
 "Marks":3}
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
     "📷 OCR Question (Coming to live soon!)",
     "🧮 AI Tutor",
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
        st.info(f"Total Marks: {q_data['Marks']}")
        st.session_state.learner["solved"] += 1
        st.session_state.learner["Marks"] += q_data["Marks"]

# =====================================================
# AI SOLVER (FULL PAPER 1 & PAPER 2 LOGIC)
# =====================================================
elif mode=="🧮 AI Tutor":
    st.title("🧮 AI Tutor")

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
            # CLEAN & PARSE INPUT
            # ------------------------------
            # Replace ^ with ** for SymPy and remove spaces
            q_clean = question.replace("^", "**").replace(" ", "")
            
            # Split by comma to handle simultaneous equations
            raw_eqs = q_clean.split(",")
            
            # Identify all variable symbols (e.g., x, y)
            symbols_in_expr = sorted(list(set(re.findall(r"[a-zA-Z]", q_clean))))
            symbols_dict = {s: sp.symbols(s) for s in symbols_in_expr}
            var_list = list(symbols_dict.values())

            # =====================================================
            # PAPER 1
            # =====================================================
            if topic == "Algebra":
                st.markdown("### Algebra Solution")

                # ------------------------------
                # CLEAN & PARSE INPUT
                # ------------------------------
                question_clean = question.replace("^", "**").replace(" ", "")
                question_clean = re.sub(r'(\))(\()', r'\1*\2', question_clean)
                question_clean = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', question_clean)
                question_clean = question_clean.replace("≤", "<=").replace("≥", ">=")

                raw_eqs = question_clean.split(",")
                symbols_in_expr = sorted(list(set(re.findall(r"[a-zA-Z]", question_clean))))
                symbols_dict = {s: sp.symbols(s) for s in symbols_in_expr}
                var_list = list(symbols_dict.values())

                try:
                    parsed_eqs = []
                    is_inequality = False

                    for eq_str in raw_eqs:
                        if any(op in eq_str for op in ["<=", ">=", "<", ">"]):
                            is_inequality = True
                            parsed_eqs.append(sp.sympify(eq_str, locals=symbols_dict))
                        elif "=" in eq_str:
                            lhs_str, rhs_str = eq_str.split("=")
                            lhs = sp.sympify(lhs_str, locals=symbols_dict)
                            rhs = sp.sympify(rhs_str, locals=symbols_dict)
                            parsed_eqs.append(lhs - rhs)
                        else:
                            parsed_eqs.append(sp.sympify(eq_str, locals=symbols_dict))

#---------------------------------------------START INEQUALITY SOLVER----------------------------------------------------------------------------------
                    # ---------------------------------------------
                    # START INEQUALITY SOLVER (CLEAN VERSION)
                    # ---------------------------------------------
                    if is_inequality:

                        var = var_list[0]
                        inequality = parsed_eqs[0]

                        st.write("##### 💡 Step 1: Analyse the inequality")
                        st.latex(sp.latex(inequality))

                        # ---------------------------------------------
                        # STEP 2.1: Write in standard form
                        # ---------------------------------------------
                        st.write("##### 📝 Step 2: Calculation")

                        lhs, rhs = inequality.lhs, inequality.rhs
                        expr = sp.simplify(lhs - rhs)

                        st.markdown("**Step 2.1: Write the inequality with zero on one side**")
                        st.latex(sp.latex(inequality.func(expr, 0)))

                        # ---------------------------------------------
                        # STEP 2.2: Determine degree
                        # ---------------------------------------------
                        degree = sp.degree(expr, var)
                        st.markdown("**Step 2.2: Determine the degree of the expression**")
                        st.write(f"The degree of the expression is **{degree}**.")

                        roots = []
                        can_proceed = False

                        # ---------------------------------------------
                        # STEP 2.3: Find the roots
                        # ---------------------------------------------
                        st.markdown("**Step 2.3: Find the roots of the expression**")

                        factored = sp.factor(expr)
                        expanded = sp.expand(expr)

                        # CASE 1: Already factorised
                        if expr.is_Mul:
                            st.write("The expression is already factorised.")
                            st.latex(sp.latex(expr))
                            roots = sp.solve(expr, var)
                            can_proceed = True

                        # CASE 2: Factorisable after factoring
                        elif factored != expanded:
                            st.write("Factorising the expression:")
                            st.latex(sp.latex(factored))
                            roots = sp.solve(factored, var)
                            can_proceed = True

                        # CASE 3: Quadratic but not factorisable
                        elif degree == 2:
                            st.write("The expression cannot be factorised easily, We use the quadratic formula.")
                            #st.write("We use the quadratic formula.")

                            a = expanded.coeff(var, 2)
                            b = expanded.coeff(var, 1)
                            c = expanded.coeff(var, 0)

                            st.latex(rf"a = {a}, \quad b = {b}, \quad c = {c}")
                            st.latex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
                            st.latex(rf"x = \frac{{-({b}) \pm \sqrt{{({b})^2 - 4({a})({c})}}}}{{2({a})}}")

                            discriminant = b**2 - 4*a*c
                            st.latex(rf"\Delta = ({b})^2 - 4({a})({c})")
                            st.latex(rf"\Delta = {sp.latex(discriminant)}")

                            if discriminant.is_negative:
                                st.error(
                                    "Since the discriminant is negative, there are **no real roots**."
                                )
                                st.info("Grade 12 learners do not work with complex numbers.")
                                can_proceed = False
                            else:
                                st.info(
                                    "Since the discriminant is non-negative, real roots exist."
                                )
                                roots = sp.solve(expanded, var)
                                can_proceed = True

                        # CASE 4: Higher degree (Grade 12 limit)
                        else:
                            st.warning(
                                "This inequality cannot be solved using Grade 12 methods."
                            )
                            can_proceed = False

                        # ---------------------------------------------
                        # STEP 2.4: Display roots
                        # ---------------------------------------------
                        if can_proceed and roots:
                            st.markdown("**Step 2.4: Critical values**")
                            for r in roots:
                                st.latex(f"{sp.latex(var)} = {sp.latex(r)}")

                            # ---------------------------------------------
                            # STEP 2.5: Solve inequality
                            # ---------------------------------------------
                            st.markdown("**Step 2.5: Solve the inequality**")
                            solution = solve_univariate_inequality(
                                inequality, var, relational=False
                            )

                            # ---------------------------------------------
                            # FINAL ANSWER
                            # ---------------------------------------------
                            st.markdown("### 🏁 Final Answer")

                            if isinstance(solution, sp.Interval):
                                left, right = solution.start, solution.end
                                left_op = "<" if solution.left_open else r"\leq"
                                right_op = "<" if solution.right_open else r"\leq"

                                st.latex(
                                    rf"{sp.latex(left)} {left_op} {sp.latex(var)} {right_op} {sp.latex(right)}"
                                )
                            else:
                                st.latex(sp.latex(solution))


#--------------------------------------------------END INEQUALITY SOLVER----------------------------------------------------------------------------------

                    # --------------------------------------------------
                    # ALGEBRAIC EQUATION SOLVER (NO INEQUALITIES)
                    # GRADE 12 SAFE – REAL ROOTS ONLY
                    # --------------------------------------------------
                    else:
                        breakpoint = False
                    #st.markdown("✏️ **Algebraic Equation Solution**")

                    # Assumptions:
                    # - parsed_eqs: list of sympy expressions already equal to 0
                    # - raw_eqs: original user input strings
                    # - var_list: detected variables
                    # - symbols_dict: sympy symbol dictionary

                        if len(parsed_eqs) == 1 and len(var_list) == 1:
                            var = var_list[0]
                            expr = parsed_eqs[0]

                            # ----------------------------------
                            # STEP 1: Write equation
                            # ----------------------------------
                            st.markdown("###### Step 1: Write the equation")

                            if "=" in raw_eqs[0]:
                                lhs_str, rhs_str = raw_eqs[0].split("=")
                                lhs = sp.sympify(lhs_str, locals=symbols_dict)
                                rhs = sp.sympify(rhs_str, locals=symbols_dict)
                                equation = lhs - rhs
                                st.latex(sp.latex(sp.Eq(lhs, rhs)))
                            else:
                                equation = expr
                                st.latex(sp.latex(expr) + " = 0")

                            # ----------------------------------
                            # STEP 2: Standard form
                            # ----------------------------------

                            st.markdown("###### Step 2: Write in standard form")

                            # expr_raw = equation BEFORE expansion
                            expr_raw = expr

                            # RHS is zero because we moved everything to LHS
                            rhs_is_zero = True

                            # ----------------------------------
                            # CASE 1: Already factorised (product form)
                            # ----------------------------------
                            if rhs_is_zero and expr_raw.is_Mul:
                                st.write("The equation is already factorised.")
                                st.latex(sp.latex(expr_raw) + " = 0")

                                factored = expr_raw

                            # ----------------------------------
                            # CASE 2: Not factorised → try factorising
                            # ----------------------------------
                            elif rhs_is_zero:
                                expr_std = sp.expand(expr_raw)
                                st.latex(sp.latex(expr_std) + " = 0")

                                degree = sp.degree(expr_std, var)
                                st.info(f"The degree of the equation is **{degree}**.")

                                factored = sp.factor(expr_std)

                                if factored != expr_std:
                                    st.write("Factorising the expression:")
                                    st.latex(sp.latex(factored) + " = 0")
                                else:
                                    pass
                                    #st.warning("Expression cannot be factorised further using Grade 12 methods.")
                                    #factored = expr_std

                            # ----------------------------------
                            # CASE 3: RHS ≠ 0 → must expand
                            # ----------------------------------
                            else:
                                st.write("Right-hand side is not zero. Rewrite in standard form.")
                                expr_std = sp.expand(expr_raw)
                                st.latex(sp.latex(expr_std) + " = 0")
                                factored = sp.factor(expr_std)


                            # ----------------------------------
                            # STEP 4: Solve factor-by-factor
                            # ----------------------------------
                            #st.info("Solve each factor:")

                            factors = factored.as_ordered_factors()
                            all_roots = []

                            for f in factors:
                                
                                # Remove powers: (x-2)^2 → x-2
                                base, power = f.as_base_exp()
                                base = sp.factor(base)

                                if not base.has(var):
                                    continue

                                deg = sp.degree(base, var)

                                # ------------------------------
                                # LINEAR FACTOR
                                # ------------------------------
                                
                                if deg == 1:
                                    st.markdown(f"**Solve:** ${sp.latex(base)} = 0$")
                                    root = sp.solve(base, var)[0]
                                    st.latex(rf"{sp.latex(var)} = {sp.latex(root)}")
                                    all_roots.append(root)

                                # ------------------------------
                                # QUADRATIC FACTOR
                                # ------------------------------
                                elif deg == 2:
                                    st.info(f"**Solve quadratic factor:** ${sp.latex(base)} = 0$")

                                    a = base.coeff(var, 2)
                                    b = base.coeff(var, 1)
                                    c = base.coeff(var, 0)

                                    st.markdown("Quadratic Formula:")
                                    st.latex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
                                    st.latex(rf"a = {a}, \quad b = {b}, \quad c = {c}")
                                    st.latex(rf"{sp.latex(var)} = \frac{{-({b}) \pm \sqrt{{({b})^2 - 4({a})({c})}}}}{{2({a})}}")

                                    discriminant = b**2 - 4*a*c
                                    st.latex(r"\Delta = b^2 - 4ac")
                                    st.latex(rf"\Delta = ({b})^2 - 4({a})({c})")
                                    st.latex(rf"\Delta = {sp.latex(discriminant)}")

                                    if discriminant < 0:
                                        st.error("No real roots (ignored at Grade 12 level).")
                                        continue

                                    roots = sp.solve(base, var)
                                    for r in roots:
                                        st.latex(rf"{sp.latex(var)} = {sp.latex(r)}")
                                        all_roots.append(r)

                                # ------------------------------
                                # HIGHER DEGREE (IGNORED)
                                # ------------------------------
                                else:
                                    st.warning(
                                        f"Factor ${sp.latex(base)}$ is degree {deg} and "
                                        "cannot be solved using Grade 12 methods."
                                    )

                            # ----------------------------------
                            # FINAL ANSWER
                            # ----------------------------------
                            st.markdown("###### 🏁 Final Answer")

                            # Remove duplicates (handles repeated roots)
                            final_roots = list(dict.fromkeys(all_roots))

                            if final_roots:
                                answer = " \\text{ or } ".join(
                                    [rf"{sp.latex(var)} = {sp.latex(r)}" for r in final_roots]
                                )
                                st.latex(answer)
                            else:
                                st.error("No real solutions found.")

                        else:
                            st.warning("This solver currently supports ONE equation with ONE variable only.")


                except Exception as e:
                    st.error("Error parsing expression.")
                    st.caption(str(e))

       #-----------------------------------------------------------------------------------------------         
                           #SEQUENCES MODULE
        #----------------------------------------------------------------------------------------------
            elif topic == "Sequences":
                st.markdown("### 🔢 Sequence Analyzer")

                try:
                    raw = question.strip()

                    # ------------------------------
                    # STEP 1: Split at ...
                    # ------------------------------
                    if "..." in raw or ".." in raw:
                        parts = re.split(r"\.\.\.|\.{2}", raw, maxsplit=1)
                        left_part = parts[0]
                        right_part = parts[1] if len(parts) > 1 else ""
                    else:
                        left_part = raw
                        right_part = ""

                    # Normalize separators
                    left_part = left_part.replace("+", ",")
                    right_part = right_part.replace("+", ",")

                    # Extract numbers
                    left_numbers = re.findall(r"-?\d+\.?\d*", left_part)
                    seq = [int(n) for n in left_numbers]

                    right_numbers = re.findall(r"-?\d+\.?\d*", right_part)
                    last_term = int(right_numbers[-1]) if right_numbers else None

                    # ------------------------------
                    # STEP 2: Display given sequence
                    # ------------------------------
                    st.markdown("**Step 1: Write the sequence**")
                    st.latex(",\\;".join(map(str, seq)) + (",\\;\\ldots" if "..." in raw else ""))

                    if len(seq) < 3:
                        st.error("At least 3 terms are required to identify a sequence.")
                        st.stop()

                    # ------------------------------
                    # STEP 3: Detect sequence type
                    # ------------------------------
                    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
                    ratios = []

                    if all(seq[i] != 0 for i in range(len(seq)-1)):
                        ratios = [seq[i+1] / seq[i] for i in range(len(seq)-1)]

                    TOL = 1e-6
                    is_arithmetic = all(abs(d - diffs[0]) < TOL for d in diffs)
                    is_geometric = ratios and all(abs(r - ratios[0]) < TOL for r in ratios)

                    # ------------------------------
                    # ARITHMETIC SEQUENCE
                    # ------------------------------
                    if is_arithmetic:
                        a = seq[0]
                        d = diffs[0]

                        st.success("This is an **Arithmetic Sequence**")

                        st.markdown("**Step 2: Identify parameters**")

                        #st.markdown("### 🔍 Step 2: Find a and d")

                        # Ensure at least 2 terms exist
                        if len(seq) < 2:
                            st.error("At least two terms are required to find a and d.")
                        else:
                            # First term
                            a = seq[0]
                            # Common difference
                            d = seq[1] - seq[0]

                            # Display steps
                            st.markdown("**First term (a):**")
                            st.latex(r"a = T_1")
                            st.latex(rf"a = {a}")

                            st.markdown("**Common difference (d):**")
                            st.latex(r"d = T_2 - T_1")
                            #st.latex(rf"d = {seq[1]} - {seq[0]} = {d}")
                            st.latex(rf"d = {seq[1]} - {seq[0]}")
                            #st.latex(rf"a = {a}, \quad d = {d}")
                            st.latex(rf"\quad d = {d}")


                        st.markdown("**Step 3: General term**")
                        st.latex(r"T_n = a + (n-1)d")
                        st.latex(rf"T_n = {a} + (n-1)({d})")
                        st.latex(rf"T_n = {a} + {d}n-{d}")
                        expanded = sp.expand(a + (sp.Symbol('n') - 1)*d)
                        #st.markdown("**Expand**")
                        st.latex(rf"T_n = {sp.latex(expanded)}")
                        #simplified = sp.simplify(expanded)
                        #st.markdown("**Simplified general term**")
                        #st.latex(rf"T_n = {sp.latex(simplified)}")



                        if last_term is not None:
                            st.markdown("**Step 4: Find number of terms**")
                            st.latex(rf"{last_term} = {a} + (n-1){d}")
                            n = (last_term - a) / d + 1
                            n = int(n) if n.is_integer() else n
                            st.latex(rf"n = {n}")

                            st.markdown("**Step 5: Sum of terms**")
                            st.latex(r"S_n = \frac{n}{2}(a + l)")
                            st.latex(rf"S_{n} = \frac{{{n}}}{2}({a}+{last_term})")

                    # ------------------------------
                    # GEOMETRIC SEQUENCE
                    # ------------------------------
                    elif is_geometric:
                        a = seq[0]
                        r = ratios[0]

                        st.success("This is a **Geometric Sequence**")

                        st.markdown("**Step 2: Identify parameters**")

                        
                        a = seq[0]
                        #st.markdown("**Step 2: Identify the first term (a)**")
                        st.latex(r"a = T_1")
                        st.latex(rf"a = {a}")

                        # Step 3: Identify common ratio (r)
                        if len(seq) >= 2:
                            r = seq[1] / seq[0]
                            #st.markdown("**Step 3: Identify the common ratio (r)**")
                            st.latex(r"r = \frac{T_2}{T_1}=\frac{T_3}{T_2}")
                            st.latex(rf"r = \frac{{{seq[1]}}}{{{seq[0]}}}")
                            #r_frac = sp.Rational(r).limit_denominator()
                            #st.latex(rf"\quad r = \frac{{{r_frac.numerator}}}{{{r_frac.denominator}}}")
                            st.latex(rf"\quad r = {r}")

                        st.markdown("**Step 3: General term**")
                        st.latex(r"T_n = ar^{n-1}")
                        st.latex(rf"T_n = {a}({r})^{{n-1}}")

                        if last_term is not None:
                            st.markdown("**Step 4: Find number of terms**")
                            st.latex(rf"{last_term} = {a}({r})^{{n-1}}")
                            n = sp.solve(sp.Eq(last_term, a * r**(sp.symbols("n")-1)), sp.symbols("n"))
                            st.latex(rf"n = {sp.latex(n)}")

                    # ------------------------------
                    # NEITHER
                    # ------------------------------
                    else:
                        st.error("This sequence is neither arithmetic nor geometric.")

                except Exception as e:
                    st.error("Could not analyse the sequence.")
                    st.caption(str(e))






            elif topic == "Financial Mathematics":
                st.markdown("### 💰 Compound Interest")

                try:
                    # Extract P, i, n from question automatically if possible
                    P_match = re.search(r"P\s*=\s*([-+]?\d*\.?\d+)", question)
                    i_match = re.search(r"i\s*=\s*([-+]?\d*\.?\d+)", question)
                    n_match = re.search(r"n\s*=\s*([-+]?\d*\.?\d+)", question)

                    P = float(P_match.group(1)) if P_match else st.number_input("Principal (P)", 1000.0)
                    i = float(i_match.group(1))/100 if i_match else st.number_input("Interest rate (%)", 10.0)/100
                    n = float(n_match.group(1)) if n_match else st.number_input("Time (years)", 2.0)

                    A = P * (1 + i) ** n

                    st.latex(r"A = P(1+i)^n")
                    st.latex(rf"A = {round(A, 2)}")

                except Exception as e:
                    st.error("Could not parse financial parameters from the question.")
                    st.caption(str(e))


            elif topic == "Calculus":
                st.markdown("### 📐 Differentiation: Comparison of Methods")
                
                try:
                    # --- 1. CLEAN & PARSE INPUT ---
                    expr_str = question.lower()
                    # Remove common prefixes
                    expr_str = re.sub(r"(find derivative of|differentiate|dy/dx|y\s*=|f\(x\)\s*=)", "", expr_str)
                    expr_str = expr_str.strip()

                    # Handle implicit multiplication and powers
                    expr_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr_str)
                    expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)
                    expr_str = expr_str.replace("^", "**")

                    # Define symbols and parse
                    x = sp.symbols("x")
                    expr = sp.sympify(expr_str)
                    h = sp.symbols("h")

                    # --- 2. CREATE SIDE-BY-SIDE COLUMNS ---
                    col1, col2 = st.columns(2)

                    # --- LEFT COLUMN: POWER RULE ---
                    with col1:
                        st.subheader("🚀 Power Rule")
                        st.info("The standard shortcut method.")
                        
                        derivative_pr = sp.diff(expr, x)
                        
                        st.markdown("**Step 1: Apply rules to terms**")
                        terms = expr.as_ordered_terms()
                        for term in terms:
                            coeff, power = term.as_coeff_exponent(x)
                            if power != 0:
                                st.latex(rf"\frac{{d}}{{dx}}({sp.latex(term)}) = {sp.latex(coeff * power)}x^{{{sp.latex(power-1)}}}")
                            else:
                                st.latex(rf"\frac{{d}}{{dx}}({sp.latex(term)}) = 0")
                        
                        st.markdown("**Final Result (Power Rule):**")
                        st.latex(rf"f'(x) = {sp.latex(derivative_pr)}")

                    # --- RIGHT COLUMN: FIRST PRINCIPLE ---
                    with col2:
                        st.subheader("📝 First Principle")
                        st.info("Definition using limits.")
                        
                        # Step 1: Formula and Substitution
                        st.markdown("**Step 1: Substitution**")
                        f_x = expr
                        f_xh = expr.subs(x, x + h)
                        
                        st.latex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}")
                        st.latex(rf"f'(x) = \lim_{{h \to 0}} \frac{{{sp.latex(f_xh)} - ({sp.latex(f_x)})}}{{h}}")
                        
                        # Step 2: Simplify numerator
                        st.markdown("**Step 2: Expand Numerator**")
                        numerator_expanded = sp.expand(f_xh - f_x)
                        st.latex(rf"f'(x) = \lim_{{h \to 0}} \frac{{{sp.latex(numerator_expanded)}}}{{h}}")
                        
                        # Step 3: Factor and Cancel h
                        st.markdown("**Step 3: Cancel $h$**")
                        # We divide by h manually to show the cancellation clearly
                        terms_after_h = sp.expand(numerator_expanded / h)
                        st.latex(rf"f'(x) = \lim_{{h \to 0}} ({sp.latex(terms_after_h)})")
                        
                        # Step 4: Final Limit
                        st.markdown("**Step 4: Final Result**")
                        derivative_fp = sp.limit(numerator_expanded / h, h, 0)
                        st.latex(rf"f'(x) = {sp.latex(derivative_fp)}")

                except Exception as e:
                    st.error("Could not parse the expression for differentiation.")
                    st.caption(f"Error details: {str(e)}")


            elif topic == "Functions & Graphs":
                st.markdown("### 📈 Functions & Graphs")

                try:
                    # ---------------------------------------------------
                    # 1. CLEAN INPUT
                    # ---------------------------------------------------
                    expr_str = question.lower()
                    expr_str = re.sub(r"(graph|sketch|draw|y\s*=|f\(x\)\s*=)", "", expr_str)
                    expr_str = expr_str.strip()

                    # Handle implicit multiplication
                    expr_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr_str)
                    expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)

                    x = sp.symbols("x")
                    expr = sp.sympify(expr_str)

                    st.markdown("##### 🔹 Given Function")
                    st.latex(r"y = " + sp.latex(expr))

                    # ---------------------------------------------------
                    # 2. DOMAIN
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 Domain")

                    domain = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
                    st.latex(r"\text{Domain: } " + sp.latex(domain))

                    # ---------------------------------------------------
                    # 3. Y-INTERCEPT
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 y-intercept")
                    y_int = expr.subs(x, 0)
                    st.latex(r"x = 0")
                    st.latex(r"y = " + sp.latex(y_int))
                    st.latex(rf"\text{{y-intercept: }} (0, {sp.latex(y_int)})")

                    # ---------------------------------------------------
                    # 4. X-INTERCEPTS
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 x-intercepts")
                    roots = sp.solve(expr, x)

                    if roots:
                        for r in roots:
                            st.latex(rf"x = {sp.latex(r)}")
                            st.latex(rf"\text{{Intercept: }} ({sp.latex(r)}, 0)")
                    else:
                        st.latex(r"\text{No real x-intercepts}")

                    # ---------------------------------------------------
                    # 5. FIRST DERIVATIVE
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 First Derivative")
                    derivative = sp.diff(expr, x)
                    st.latex(r"\frac{dy}{dx} = " + sp.latex(derivative))

                    # ---------------------------------------------------
                    # 6. TURNING POINTS
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 Turning Points")
                    turning_x = sp.solve(derivative, x)

                    turning_points = []

                    if turning_x:
                        second_derivative = sp.diff(derivative, x)

                        for tx in turning_x:
                            ty = expr.subs(x, tx)
                            turning_points.append(ty)

                            st.latex(rf"x = {sp.latex(tx)}")
                            st.latex(rf"y = {sp.latex(ty)}")

                            nature = second_derivative.subs(x, tx)
                            if nature > 0:
                                st.latex(r"\text{Minimum turning point}")
                            elif nature < 0:
                                st.latex(r"\text{Maximum turning point}")
                            else:
                                st.latex(r"\text{Point of inflection}")
                    else:
                        st.latex(r"\text{No turning points}")

                    # ---------------------------------------------------
                    # 7. AXIS OF SYMMETRY (QUADRATIC ONLY)
                    # ---------------------------------------------------
                    if sp.degree(expr, x) == 2:
                        st.markdown("##### 🔹 Axis of Symmetry")

                        a = expr.coeff(x, 2)
                        b = expr.coeff(x, 1)

                        axis = -b / (2 * a)
                        st.latex(r"x = -\frac{b}{2a}")
                        st.latex(rf"x = {sp.latex(axis)}")

                    # ---------------------------------------------------
                    # 8. RANGE
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 Range")

                    if turning_points:
                        # Use leading coefficient to determine concavity
                        deg = sp.degree(expr, x)
                        lead_coeff = expr.coeff(x, deg)

                        # Find min and max of turning points
                        y_values = [expr.subs(x, tx) for tx in turning_points]
                        y_min = min(y_values)
                        y_max = max(y_values)

                        if deg % 2 == 0:  # Even degree (quadratic)
                            if lead_coeff > 0:
                                st.latex(rf"y \geq {sp.latex(y_min)}")
                            else:
                                st.latex(rf"y \leq {sp.latex(y_max)}")
                        else:  # Odd degree (cubic)
                            st.latex(r"\text{Range: } (-\infty, \infty)")
                    else:
                        st.latex(r"\text{Range depends on domain and end behaviour}")


                    # ---------------------------------------------------
                    # 9. ASYMPTOTES (RATIONAL FUNCTIONS)
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 Asymptotes")

                    num, den = sp.fraction(expr)

                    # Vertical asymptotes
                    vert_asym = sp.solve(den, x)
                    if vert_asym:
                        for va in vert_asym:
                            st.latex(rf"x = {sp.latex(va)}")
                    else:
                        st.latex(r"\text{No vertical asymptotes}")

                    # Horizontal asymptote
                    horiz_asym = sp.limit(expr, x, sp.oo)
                    if horiz_asym.is_finite:
                        st.latex(rf"y = {sp.latex(horiz_asym)}")

                    # ---------------------------------------------------
                    # 10. END BEHAVIOUR
                    # ---------------------------------------------------
                    st.markdown("##### 🔹 End Behaviour")

                    left = sp.limit(expr, x, -sp.oo)
                    right = sp.limit(expr, x, sp.oo)

                    st.latex(r"\lim_{x \to -\infty} f(x) = " + sp.latex(left))
                    st.latex(r"\lim_{x \to \infty} f(x) = " + sp.latex(right))

                    # ---------------------------------------------------
                    # 11. GRAPH
                    # ---------------------------------------------------
                    st.markdown("##### 📉 Sketch of the Graph")

                #try:
                    # 1. Use 'x' directly as it was defined in your symbols section
                    # Convert roots to floats for the x-axis range
                    real_roots = [float(r.evalf()) for r in roots if sp.simplify(r).is_real]
                    
                    if real_roots:
                        x_min = min(real_roots) - 2
                        x_max = max(real_roots) + 2
                    else:
                        x_min, x_max = -5, 5

                    # 2. Convert SymPy expression to a NumPy-friendly function
                    # We use 'x' here because it is your defined symbol
                    f_np = sp.lambdify(x, expr, modules=['numpy'])
                    
                    # 3. Generate points for plotting
                    xs = np.linspace(float(x_min), float(x_max), 400)
                    ys = f_np(xs)

                    # 4. Create the Plot
                    fig, ax = plt.subplots()
                    ax.plot(xs, ys, label=f"f(x) = {sp.latex(expr)}")
                    ax.axhline(0, color='black', linewidth=1.2) # x-axis
                    ax.axvline(0, color='black', linewidth=1.2) # y-axis
                    
                    # Mark the intercepts (roots) with red dots
                    for root in real_roots:
                        ax.plot(root, 0, 'ro') 

                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.set_title(f"Visualizing the Function")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")

                    # 5. Set y-limits safely
                    y_vals = ys[np.isfinite(ys)]
                    if len(y_vals) > 0:
                        ax.set_ylim(float(np.min(y_vals)) - 1, float(np.max(y_vals)) + 1)

                    st.pyplot(fig, use_container_width=True)


                except Exception as e:
                    st.error("Could not parse the function for graphing.")
                    st.caption(str(e))



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
    st.metric("Marks", st.session_state.learner["Marks"])

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
