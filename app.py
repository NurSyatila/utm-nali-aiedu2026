import io
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# AiEDU v3 - Pre-Post Learning Gain Analyzer
# How to run:  pip install -r requirements.txt   then   streamlit run app.py
#
# New in v3:
#   - Three ways to load data: two files, one file with a Round column,
#     or a live Google Sheet link (e.g. the Fried Rice Mystery game's form).
#   - Separate correct answers for the Before and After versions of a question.
#   - Confidence analysis ("sure but wrong" answers) when a confidence column exists.
#   - Students with full marks before the lesson are reported as "At ceiling"
#     instead of "Low gain".
#   - An activity preset that fills in everything for the Fried Rice Mystery game.
st.set_page_config(
    page_title="AiEDU – Pre–Post Learning Gain Analyzer",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Activity presets
# -----------------------------
# Each preset item finds its question column by prefix (e.g. "Q1") and supplies
# a topic label plus the correct answer for the Before and After versions.
FRIED_RICE_PRESET = {
    "label": "Fried Rice Mystery (game)",
    "id_col": "Code",
    "name_col": "Name",
    "round_col": "Round",
    "before_value": "Before",
    "after_value": "After",
    "confidence_col": "Sure",
    "items": [
        {
            "prefix": "Q1",
            "topic": "What made him sick",
            "before": "A poison made by germs that grew in the rice",
            "after": "Germs grew in the warm rice and made a poison",
        },
        {
            "prefix": "Q2",
            "topic": "Germs can survive cooking",
            "before": "Yes. Some germs hide inside a tough shell",
            "after": "No. Some germs survive cooking inside a tough shell",
        },
        {
            "prefix": "Q3",
            "topic": "Heating does not remove the poison",
            "before": "No. Heat can kill germs, but not the poison they already made",
            "after": "No. Heating does not remove the poison that is already there",
        },
        {
            "prefix": "Q4",
            "topic": "Keeping leftover rice safe",
            "before": "Put it in the fridge soon after the meal",
            "after": "Put it in the fridge within an hour",
        },
    ],
}
PRESETS = {FRIED_RICE_PRESET["label"]: FRIED_RICE_PRESET}
NO_PRESET = "None (set up manually)"
CEILING = "At ceiling"
SURE_WORDS = {"sure", "certain", "confident", "yes", "y", "high", "1", "true"}


# -----------------------------
# Utility functions
# -----------------------------
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel uploads into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload CSV, XLSX, or XLS.")


class SheetLoadError(Exception):
    """Raised with a message that can be shown to the user as is."""


def csv_url_candidates(url: str) -> List[str]:
    """Turn a Google Sheets link into CSV download links to try, best first.

    - A "Publish to web" CSV link is used as is.
    - A normal sheet link (.../spreadsheets/d/<id>/edit...) is converted to its CSV
      export. If the link names a tab (gid=...), that tab is used; otherwise the
      first tab is used. Form-response sheets often have a tab id other than 0,
      so no tab id is guessed.
    """
    url = url.strip()
    if "output=csv" in url or "format=csv" in url or "tqx=out:csv" in url:
        return [url]
    if "/pubhtml" in url:
        return [url.replace("/pubhtml", "/pub").split("#")[0] + ("&" if "?" in url else "?") + "output=csv"]
    match = re.search(r"docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)", url)
    if match and "/d/e/" not in url:
        base = f"https://docs.google.com/spreadsheets/d/{match.group(1)}"
        gid_match = re.search(r"[#&?]gid=(\d+)", url)
        if gid_match:
            gid = gid_match.group(1)
            return [
                f"{base}/export?format=csv&gid={gid}",
                f"{base}/gviz/tq?tqx=out:csv&gid={gid}",
            ]
        return [f"{base}/export?format=csv", f"{base}/gviz/tq?tqx=out:csv"]
    return [url]


@st.cache_data(ttl=60, show_spinner="Loading the latest responses…")
def read_sheet(url: str) -> pd.DataFrame:
    """Download a Google Sheet as CSV, trying each candidate link in turn."""
    import urllib.error
    import urllib.request

    problems = []
    for candidate in csv_url_candidates(url):
        request = urllib.request.Request(candidate, headers={"User-Agent": "Mozilla/5.0 AiEDU"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
        except urllib.error.HTTPError as exc:
            problems.append(f"HTTP {exc.code}")
            continue
        except urllib.error.URLError as exc:
            raise SheetLoadError(f"Google Sheets could not be reached ({exc.reason}). Check the internet connection.")
        if "text/html" in content_type or body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
            # Google answers with a sign-in page when the sheet is private.
            problems.append("sign-in page")
            continue
        return pd.read_csv(io.BytesIO(body))

    if "sign-in page" in problems:
        raise SheetLoadError(
            "The sheet is private. In Google Sheets choose Share → General access → "
            "'Anyone with the link' → Viewer, then press Refresh data."
        )
    raise SheetLoadError(
        "Google Sheets refused the link (" + ", ".join(problems) + "). Open the sheet, click the "
        "tab with the form responses, copy the address from the browser bar (it ends in gid=…) "
        "and paste that here instead."
    )


def split_by_round(
    df: pd.DataFrame, round_col: str, before_value: str, after_value: str
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split one combined table into Before and After rows.

    Rows whose round is neither value (for example "Test") are ignored and
    counted so the interface can report them.
    """
    rounds = df[round_col].map(clean_answer)
    is_before = rounds == clean_answer(before_value)
    is_after = rounds == clean_answer(after_value)
    ignored = int((~is_before & ~is_after).sum())
    return (
        df[is_before].reset_index(drop=True),
        df[is_after].reset_index(drop=True),
        ignored,
    )


def find_column(columns: List[str], prefix: str) -> Optional[str]:
    """Return the first column named like the prefix, e.g. "Q1" or "Q1 - Topic"."""
    pattern = re.compile(rf"^\s*{re.escape(prefix)}(?!\d)", re.IGNORECASE)
    for column in columns:
        if pattern.match(str(column)):
            return column
    return None


def parse_confidence(value, count: int) -> List[bool]:
    """Read a comma-separated confidence string such as "Sure, Guess, Sure, Sure".

    Returns one True (sure) or False (not sure) per question, padded to `count`.
    """
    if pd.isna(value):
        return [False] * count
    tokens = [token.strip().casefold() for token in str(value).split(",")]
    flags = [token in SURE_WORDS for token in tokens][:count]
    return flags + [False] * (count - len(flags))


def normalise_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def clean_answer(value) -> str:
    """Normalise answer text for case-insensitive exact comparison."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def unique_nonempty_values(series: pd.Series) -> List[str]:
    values = []
    seen = set()
    for value in series.dropna().astype(str):
        value = re.sub(r"\s+", " ", value.strip())
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            values.append(value)
    return values


def score_answer_series(series: pd.Series, correct_answer: str) -> pd.Series:
    expected = clean_answer(correct_answer)
    return series.map(lambda x: int(clean_answer(x) == expected))


def safe_gain(pre_score: pd.Series, post_score: pd.Series, total: int) -> pd.Series:
    """Calculate Hake normalised gain.

    A learner with full marks before the lesson has no gap to close, so their
    gain is left empty (NaN) and reported as "At ceiling" rather than as 0,
    which would wrongly flag the strongest students as "Low gain".
    """
    denominator = (total - pre_score).astype(float)
    gain = (post_score - pre_score) / denominator.replace(0, np.nan)
    return gain.replace([np.inf, -np.inf], np.nan)


def gain_category(gain: float) -> str:
    if pd.isna(gain):
        return CEILING
    if gain >= 0.7:
        return "High gain"
    if gain >= 0.3:
        return "Medium gain"
    return "Low gain"


def item_status(post_pct: float, improvement: float) -> str:
    if post_pct >= 80 and improvement >= 0:
        return "Strength"
    if improvement >= 20:
        return "Strong improvement"
    if post_pct < 60:
        return "Needs attention"
    if improvement <= 0:
        return "No improvement"
    return "Developing"


def build_feedback(item_results: pd.DataFrame, mean_gain: float) -> List[str]:
    messages = []

    strengths = item_results[item_results["Post correct (%)"] >= 80]
    weaknesses = item_results[
        (item_results["Post correct (%)"] < 60)
        | (item_results["Improvement (pp)"] <= 0)
    ]
    biggest = item_results.sort_values("Improvement (pp)", ascending=False).head(1)

    if not strengths.empty:
        topics = ", ".join(strengths["Topic"].astype(str).tolist())
        messages.append(
            f"Strong post-lesson understanding was demonstrated for: {topics}."
        )

    if not biggest.empty:
        row = biggest.iloc[0]
        messages.append(
            f"The largest improvement was observed for {row['Topic']} "
            f"({row['Improvement (pp)']:.1f} percentage points)."
        )

    if not weaknesses.empty:
        topics = ", ".join(weaknesses["Topic"].astype(str).tolist())
        messages.append(
            f"Consider revisiting or using an alternative explanation for: {topics}."
        )
    else:
        messages.append(
            "No item met the current warning criteria; reinforcement can focus on maintaining retention."
        )

    if "Confident errors before (%)" in item_results.columns:
        before_ce = item_results["Confident errors before (%)"].mean()
        after_ce = item_results["Confident errors after (%)"].mean()
        if before_ce > 0 or after_ce > 0:
            messages.append(
                f"Confident misconceptions (students sure of a wrong answer) went from "
                f"{before_ce:.1f}% to {after_ce:.1f}% of answers."
            )
        stubborn = item_results.sort_values("Confident errors after (%)", ascending=False).head(1)
        if not stubborn.empty and stubborn.iloc[0]["Confident errors after (%)"] > 0:
            row = stubborn.iloc[0]
            messages.append(
                f"The most persistent confident misconception after the lesson was in "
                f"{row['Topic']} ({row['Confident errors after (%)']:.1f}% of students). "
                "Address it directly, for example with a counter-example or discussion."
            )

    if pd.isna(mean_gain):
        messages.append(
            "Every matched student had full marks before the lesson, so normalised gain "
            "cannot be calculated. Consider harder or more open questions next time."
        )
    elif mean_gain >= 0.7:
        messages.append("Overall learning gain was high.")
    elif mean_gain >= 0.3:
        messages.append("Overall learning gain was moderate.")
    else:
        messages.append(
            "Overall learning gain was low; review item difficulty, lesson alignment, and opportunities for guided practice."
        )

    return messages


def dataframe_to_excel_bytes(
    student_results: pd.DataFrame, item_results: pd.DataFrame
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        student_results.to_excel(writer, sheet_name="Student results", index=False)
        item_results.to_excel(writer, sheet_name="Item analysis", index=False)
    output.seek(0)
    return output.getvalue()



# -----------------------------
# Interface
# -----------------------------
st.title("AiEDU – Pre–Post Learning Gain Analyzer")
st.write(
    "Load responses collected before and after a lesson. "
    "The app matches students, scores each item, calculates learning gain, and generates feedback."
)

preset_label = st.selectbox(
    "Activity preset",
    [NO_PRESET] + list(PRESETS),
    help="A preset fills in the student ID, question pairs, topics and correct answers for a known activity.",
)
preset = PRESETS.get(preset_label)

SOURCE_TWO = "Two files (Before and After)"
SOURCE_ONE = "One file with a Round column"
SOURCE_LINK = "Google Sheet link (live)"
source = st.radio(
    "How is your data organised?",
    [SOURCE_TWO, SOURCE_ONE, SOURCE_LINK],
    index=2 if preset else 0,
    horizontal=True,
)

with st.expander("Required file structure", expanded=False):
    st.markdown(
        """
        - Every response needs a student identifier that is the same before and after, such as **Student ID** or the game's **Code**.
        - **Two files:** one file of Before responses and one of After responses.
        - **One file / Google Sheet:** a single table with a **Round** column saying *Before* or *After*. Other rows, such as *Test*, are ignored.
        - Question wording and correct answers may differ between Before and After.
        - An optional confidence column (e.g. *Sure, Guess, Sure, Sure*, in question order) enables the confidence analysis.
        - CSV, XLSX, and XLS files are supported.
        """
    )

ignored_rows = 0
if source == SOURCE_TWO:
    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        pre_file = st.file_uploader(
            "Upload BEFORE file", type=["csv", "xlsx", "xls"], key="pre_file"
        )
    with upload_col2:
        post_file = st.file_uploader(
            "Upload AFTER file", type=["csv", "xlsx", "xls"], key="post_file"
        )

    if not pre_file or not post_file:
        st.info("Upload both files to begin the analysis.")
        st.stop()

    try:
        pre_df = read_uploaded_file(pre_file)
        post_df = read_uploaded_file(post_file)
    except Exception as exc:
        st.error(f"The files could not be read: {exc}")
        st.stop()
else:
    if source == SOURCE_ONE:
        combined_file = st.file_uploader(
            "Upload the combined file (e.g. the Google Sheet downloaded as CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="combined_file",
        )
        if not combined_file:
            st.info("Upload the file to begin the analysis.")
            st.stop()
        try:
            combined_df = read_uploaded_file(combined_file)
        except Exception as exc:
            st.error(f"The file could not be read: {exc}")
            st.stop()
    else:
        sheet_url = st.text_input(
            "Google Sheet link",
            key="sheet_url",
            placeholder="https://docs.google.com/spreadsheets/d/…",
            help=(
                "In the Google Sheet: Share → General access → Anyone with the link → Viewer. "
                "Or use File → Share → Publish to web → CSV."
            ),
        )
        link_col1, link_col2 = st.columns([1, 3])
        with link_col1:
            if st.button("Refresh data"):
                read_sheet.clear()
        with link_col2:
            st.caption("Data refreshes automatically every minute, or when you press Refresh.")
        if not sheet_url.strip():
            st.info(
                "Paste the link to the Google Sheet that collects your form responses. "
                "Share it as *Anyone with the link can view* first."
            )
            st.stop()
        try:
            combined_df = read_sheet(sheet_url)
        except SheetLoadError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"The sheet was downloaded but could not be read as a table ({exc}).")
            st.stop()

    if combined_df.empty:
        st.error("The table has no rows yet.")
        st.stop()

    columns = list(combined_df.columns)
    round_default = (preset or {}).get("round_col", "Round")
    round_index = columns.index(round_default) if round_default in columns else 0
    round_col = st.selectbox("Round column", columns, index=round_index)
    round_values = unique_nonempty_values(combined_df[round_col])
    if len(round_values) < 2:
        st.error(
            f"The column '{round_col}' needs at least two different values, such as Before and After."
        )
        st.stop()

    def value_index(target: str, fallback: int) -> int:
        for position, value in enumerate(round_values):
            if value.casefold() == target.casefold():
                return position
        return min(fallback, len(round_values) - 1)

    round_col1, round_col2 = st.columns(2)
    with round_col1:
        before_value = st.selectbox(
            "Value meaning BEFORE",
            round_values,
            index=value_index((preset or {}).get("before_value", "Before"), 0),
        )
    with round_col2:
        after_value = st.selectbox(
            "Value meaning AFTER",
            round_values,
            index=value_index((preset or {}).get("after_value", "After"), 1),
        )
    if clean_answer(before_value) == clean_answer(after_value):
        st.error("Choose different values for Before and After.")
        st.stop()

    pre_df, post_df, ignored_rows = split_by_round(
        combined_df, round_col, before_value, after_value
    )
    pre_df = pre_df.drop(columns=[round_col])
    post_df = post_df.drop(columns=[round_col])
    if ignored_rows:
        st.caption(
            f"{ignored_rows} row(s) with another round value (for example test rows) were ignored."
        )

if pre_df.empty or post_df.empty:
    st.error("There are no Before responses or no After responses yet.")
    st.stop()

with st.expander("Preview uploaded data", expanded=False):
    preview1, preview2 = st.columns(2)
    with preview1:
        st.subheader("Before")
        st.dataframe(pre_df.head(), width="stretch")
    with preview2:
        st.subheader("After")
        st.dataframe(post_df.head(), width="stretch")

st.header("1. Match students")
common_columns = [column for column in pre_df.columns if column in post_df.columns]
if not common_columns:
    st.error("The two files do not have any column names in common.")
    st.stop()

def first_present(options: List[str], candidates: List[Optional[str]], fallback: int = 0) -> int:
    for candidate in candidates:
        if candidate and candidate in options:
            return options.index(candidate)
    return fallback


id_default = first_present(
    common_columns, [(preset or {}).get("id_col"), "Student ID", "Code", "ID", "Matric No"]
)
id_col = st.selectbox("Student identifier column", common_columns, index=id_default)

name_options = ["None"] + common_columns
name_default = first_present(name_options, [(preset or {}).get("name_col"), "Student Name", "Name"])
name_col = st.selectbox(
    "Student name column (optional)", name_options, index=name_default
)

confidence_options = ["None"] + [c for c in common_columns if c not in (id_col, name_col)]
confidence_default = first_present(
    confidence_options, [(preset or {}).get("confidence_col"), "Sure", "Confidence"]
)
confidence_col = st.selectbox(
    "Confidence column (optional)",
    confidence_options,
    index=confidence_default,
    help=(
        "A column listing how sure the student was for each question, in question order, "
        "e.g. 'Sure, Guess, Sure, Sure'. Used to find confident misconceptions."
    ),
)

pre_df = pre_df.copy()
post_df = post_df.copy()
pre_df[id_col] = normalise_id(pre_df[id_col])
post_df[id_col] = normalise_id(post_df[id_col])

pre_duplicate_count = int(pre_df[id_col].duplicated(keep=False).sum())
post_duplicate_count = int(post_df[id_col].duplicated(keep=False).sum())

if pre_duplicate_count or post_duplicate_count:
    st.warning(
        f"Duplicate identifier rows detected — before: {pre_duplicate_count}, "
        f"after: {post_duplicate_count}. Only the last response for each student will be used."
    )
    pre_df = pre_df.drop_duplicates(subset=id_col, keep="last")
    post_df = post_df.drop_duplicates(subset=id_col, keep="last")

pre_only = sorted(set(pre_df[id_col]) - set(post_df[id_col]))
post_only = sorted(set(post_df[id_col]) - set(pre_df[id_col]))

match1, match2, match3 = st.columns(3)
matched_count = len(set(pre_df[id_col]) & set(post_df[id_col]))
match1.metric("Before responses", len(pre_df))
match2.metric("After responses", len(post_df))
match3.metric("Matched students", matched_count)

if pre_only or post_only:
    with st.expander("Unmatched identifiers", expanded=False):
        if pre_only:
            st.write("Only in before file:", ", ".join(pre_only))
        if post_only:
            st.write("Only in after file:", ", ".join(post_only))

if matched_count == 0:
    st.error("No matching student identifiers were found.")
    st.stop()

st.header("2. Pair questions and define correct answers")

excluded = {id_col}
if name_col != "None":
    excluded.add(name_col)
if confidence_col != "None":
    excluded.add(confidence_col)

# Exclude common metadata fields from the default question list.
metadata_keywords = (
    "timestamp", "date", "time", "email", "name", "class", "group",
    "round", "score", "sure", "confidence",
)
for column in set(pre_df.columns).union(post_df.columns):
    if any(keyword in str(column).casefold() for keyword in metadata_keywords):
        excluded.add(column)

pre_candidates = [column for column in pre_df.columns if column not in excluded]
post_candidates = [column for column in post_df.columns if column not in excluded]

pre_default = pre_candidates
post_default = post_candidates
preset_items: List[dict] = []
if preset:
    pre_found = [find_column(pre_candidates, item["prefix"]) for item in preset["items"]]
    post_found = [find_column(post_candidates, item["prefix"]) for item in preset["items"]]
    if all(pre_found) and all(post_found):
        pre_default, post_default = pre_found, post_found
        preset_items = preset["items"]
        st.success(f"Preset applied: {preset['label']}. Check the settings below, then press Analyze.")
    else:
        st.warning(
            "The preset's question columns were not all found, so set up the questions manually."
        )

pre_selected = st.multiselect(
    "Before-test question columns, in question order",
    pre_candidates,
    default=pre_default,
)
post_selected = st.multiselect(
    "After-test question columns, in corresponding order",
    post_candidates,
    default=post_default,
)

if not pre_selected or not post_selected:
    st.warning("Select at least one question column from each file.")
    st.stop()

if len(pre_selected) != len(post_selected):
    st.error(
        "The number of selected before and after question columns must be the same."
    )
    st.stop()

TYPE_ANOTHER = "Type another answer..."


def answer_picker(label: str, values: pd.Series, default: Optional[str], key: str) -> str:
    """Select box of the answers students gave, with the preset answer pre-selected."""
    options = unique_nonempty_values(values)
    index = 0
    if default:
        matches = [i for i, option in enumerate(options) if option.casefold() == default.casefold()]
        if matches:
            index = matches[0]
        else:
            # No student chose the correct answer yet; still offer it.
            options = [default] + options
    options = options + [TYPE_ANOTHER]
    chosen = st.selectbox(label, options, index=index, key=f"{key}_select")
    if chosen == TYPE_ANOTHER:
        chosen = st.text_input(
            "Enter the exact correct answer or option label", key=f"{key}_text"
        )
    return chosen


question_config = []
st.caption(
    "For each paired item, enter a short topic label and choose or type its correct answer. "
    "Untick 'Same correct answer' when the After question is worded differently."
)

for index, (pre_question, post_question) in enumerate(
    zip(pre_selected, post_selected), start=1
):
    item = preset_items[index - 1] if index <= len(preset_items) else None
    with st.expander(f"Question {index}", expanded=(index == 1)):
        st.write("**Before column:**", pre_question)
        st.write("**After column:**", post_question)

        topic = st.text_input(
            "Topic or item label",
            value=item["topic"] if item else f"Question {index}",
            key=f"topic_{index}",
        )

        same_answer = st.checkbox(
            "Same correct answer in Before and After",
            value=not (item and clean_answer(item["before"]) != clean_answer(item["after"])),
            key=f"same_{index}",
        )
        if same_answer:
            before_answer = answer_picker(
                "Correct answer",
                pd.concat([pre_df[pre_question], post_df[post_question]], ignore_index=True),
                item["before"] if item else None,
                f"answer_{index}",
            )
            after_answer = before_answer
        else:
            answer_col1, answer_col2 = st.columns(2)
            with answer_col1:
                before_answer = answer_picker(
                    "Correct answer – BEFORE",
                    pre_df[pre_question],
                    item["before"] if item else None,
                    f"answer_before_{index}",
                )
            with answer_col2:
                after_answer = answer_picker(
                    "Correct answer – AFTER",
                    post_df[post_question],
                    item["after"] if item else None,
                    f"answer_after_{index}",
                )

        question_config.append(
            {
                "number": index,
                "topic": topic.strip() or f"Question {index}",
                "pre_column": pre_question,
                "post_column": post_question,
                "before_answer": before_answer,
                "after_answer": after_answer,
            }
        )

missing_answer = any(
    not config["before_answer"] or not config["after_answer"] for config in question_config
)
if missing_answer:
    st.warning("Define a correct answer for every selected question.")
    st.stop()

analyze = st.button("Analyze learning improvement", type="primary")
if not analyze:
    st.stop()

# -----------------------------
# Analysis
# -----------------------------
use_confidence = confidence_col != "None"
merge_columns_pre = (
    [id_col]
    + ([name_col] if name_col != "None" else [])
    + ([confidence_col] if use_confidence else [])
    + pre_selected
)
merge_columns_post = [id_col] + ([confidence_col] if use_confidence else []) + post_selected

merged = pd.merge(
    pre_df[merge_columns_pre],
    post_df[merge_columns_post],
    on=id_col,
    how="inner",
    suffixes=("_before", "_after"),
)

# When question column names are identical, pandas adds suffixes. When names differ, it does not.
def merged_column_name(column: str, side: str, other_columns: List[str]) -> str:
    if column in other_columns:
        return f"{column}_{side}"
    return column

pre_score_columns = []
post_score_columns = []
pre_ce_columns = []
post_ce_columns = []
item_rows = []
total_questions = len(question_config)

if use_confidence:
    # The confidence column exists in both rounds, so pandas adds suffixes.
    pre_sure = merged[f"{confidence_col}_before"].map(
        lambda value: parse_confidence(value, total_questions)
    )
    post_sure = merged[f"{confidence_col}_after"].map(
        lambda value: parse_confidence(value, total_questions)
    )

for position, config in enumerate(question_config):
    number = config["number"]
    pre_question = config["pre_column"]
    post_question = config["post_column"]

    pre_merged_col = merged_column_name(pre_question, "before", post_selected)
    post_merged_col = merged_column_name(post_question, "after", pre_selected)

    pre_binary_col = f"Q{number}_before_correct"
    post_binary_col = f"Q{number}_after_correct"
    merged[pre_binary_col] = score_answer_series(merged[pre_merged_col], config["before_answer"])
    merged[post_binary_col] = score_answer_series(merged[post_merged_col], config["after_answer"])
    pre_score_columns.append(pre_binary_col)
    post_score_columns.append(post_binary_col)

    before_pct = merged[pre_binary_col].mean() * 100
    after_pct = merged[post_binary_col].mean() * 100
    improvement = after_pct - before_pct

    row = {
        "Question": f"Q{number}",
        "Topic": config["topic"],
        "Before correct (%)": before_pct,
        "Post correct (%)": after_pct,
        "Improvement (pp)": improvement,
        "Status": item_status(after_pct, improvement),
    }

    if use_confidence:
        # A confident error = the student said they were sure but answered wrongly.
        pre_ce_col = f"Q{number}_before_confident_error"
        post_ce_col = f"Q{number}_after_confident_error"
        merged[pre_ce_col] = [
            int(flags[position] and not correct)
            for flags, correct in zip(pre_sure, merged[pre_binary_col])
        ]
        merged[post_ce_col] = [
            int(flags[position] and not correct)
            for flags, correct in zip(post_sure, merged[post_binary_col])
        ]
        pre_ce_columns.append(pre_ce_col)
        post_ce_columns.append(post_ce_col)
        row["Confident errors before (%)"] = merged[pre_ce_col].mean() * 100
        row["Confident errors after (%)"] = merged[post_ce_col].mean() * 100

    item_rows.append(row)

item_results = pd.DataFrame(item_rows)
merged["Before score"] = merged[pre_score_columns].sum(axis=1)
merged["After score"] = merged[post_score_columns].sum(axis=1)
if use_confidence:
    merged["Confident errors before"] = merged[pre_ce_columns].sum(axis=1)
    merged["Confident errors after"] = merged[post_ce_columns].sum(axis=1)
merged["Score change"] = merged["After score"] - merged["Before score"]
merged["Normalised gain (g)"] = safe_gain(
    merged["Before score"], merged["After score"], total_questions
)
merged["Gain category"] = merged["Normalised gain (g)"].map(gain_category)

result_columns = [id_col]
if name_col != "None" and name_col in merged.columns:
    result_columns.append(name_col)
result_columns += [
    "Before score",
    "After score",
    "Score change",
    "Normalised gain (g)",
    "Gain category",
]
if use_confidence:
    result_columns += ["Confident errors before", "Confident errors after"]
student_results = merged[result_columns].copy()
student_results["Before score (%)"] = (
    student_results["Before score"] / total_questions * 100
)
student_results["After score (%)"] = (
    student_results["After score"] / total_questions * 100
)

mean_before = student_results["Before score (%)"].mean()
mean_after = student_results["After score (%)"].mean()
mean_gain = student_results["Normalised gain (g)"].mean()  # skips students at ceiling
ceiling_count = int(student_results["Normalised gain (g)"].isna().sum())
positive_change = (student_results["Score change"] > 0).mean() * 100

st.header("3. Results")
st.caption(f"{len(student_results)} matched students.")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Mean before score", f"{mean_before:.1f}%")
metric2.metric(
    "Mean after score", f"{mean_after:.1f}%", delta=f"{mean_after - mean_before:.1f} pp"
)
metric3.metric(
    "Mean normalised gain",
    "n/a" if pd.isna(mean_gain) else f"{mean_gain:.2f}",
    help=(
        f"{ceiling_count} student(s) at ceiling (full marks before) are excluded."
        if ceiling_count
        else None
    ),
)
metric4.metric("Students who improved", f"{positive_change:.1f}%")

feedback = build_feedback(item_results, mean_gain)
st.subheader("Automated teaching feedback")
for message in feedback:
    st.write(f"- {message}")

# ---------------------------------------------------------
# Visualization 1: Overall class performance
# ---------------------------------------------------------
st.header("Overall class performance")

# Figure 1: Mean score before and after
class_performance_df = pd.DataFrame(
    {
        "Assessment": ["Before", "After"],
        "Mean score (%)": [mean_before, mean_after],
    }
)

fig_class_performance = px.bar(
    class_performance_df,
    x="Assessment",
    y="Mean score (%)",
    text="Mean score (%)",
    title="Overall class performance",
)

fig_class_performance.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="inside",
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Mean score: %{y:.1f}%"
        "<extra></extra>"
    ),
)

fig_class_performance.update_layout(
    yaxis=dict(
        title="Mean score (%)",
        range=[0, 100],
        dtick=20,
    ),
    xaxis_title=None,
    showlegend=False,
    hovermode="x unified",
)

st.plotly_chart(
    fig_class_performance,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    },
)

mean_difference = mean_after - mean_before

if mean_difference > 0:
    st.success(
        f"The overall mean score increased from {mean_before:.1f}% "
        f"to {mean_after:.1f}%, corresponding to an improvement of "
        f"{mean_difference:.1f} percentage points."
    )
elif mean_difference < 0:
    st.warning(
        f"The overall mean score decreased from {mean_before:.1f}% "
        f"to {mean_after:.1f}%, corresponding to a decline of "
        f"{abs(mean_difference):.1f} percentage points."
    )
else:
    st.info(
        f"The overall mean score remained unchanged at "
        f"{mean_before:.1f}%."
    )


# ---------------------------------------------------------
# Visualization 2: Distribution of learning-gain categories
# ---------------------------------------------------------
st.subheader("Distribution of learning-gain categories")

gain_order = ["Low gain", "Medium gain", "High gain", CEILING]

gain_counts = (
    student_results["Gain category"]
    .value_counts()
    .reindex(gain_order, fill_value=0)
    .rename_axis("Gain category")
    .reset_index(name="Students")
)

gain_counts["Percentage"] = (
    gain_counts["Students"]
    / gain_counts["Students"].sum()
    * 100
)

# Plot only categories containing at least one student.
# This prevents a 0% category label from appearing outside the chart.
gain_counts_plot = gain_counts[
    gain_counts["Students"] > 0
].copy()

fig_gain_distribution = px.pie(
    gain_counts_plot,
    names="Gain category",
    values="Students",
    hole=0.48,
    title="Distribution of learning-gain categories",
    category_orders={
        "Gain category": gain_order,
    },
)

fig_gain_distribution.update_traces(
    textinfo="label+percent",
    textposition="inside",
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Students: %{value}<br>"
        "Percentage: %{percent}"
        "<extra></extra>"
    ),
)

fig_gain_distribution.update_layout(
    legend_title_text="Gain category",
)

st.plotly_chart(
    fig_gain_distribution,
    width="stretch",
    config={
        "displaylogo": False,
        "responsive": True,
    },
)

gain_columns = st.columns(len(gain_order))

for column, category in zip(
    gain_columns,
    gain_order,
):
    category_row = gain_counts[
        gain_counts["Gain category"] == category
    ].iloc[0]

    column.metric(
        category,
        int(category_row["Students"]),
        help=f"{category_row['Percentage']:.1f}% of matched students",
    )

st.caption(
    "Normalised-gain categories: low gain, g < 0.30; "
    "medium gain, 0.30 ≤ g < 0.70; and high gain, g ≥ 0.70. "
    "At ceiling: full marks before the lesson, so there was no gap to close."
)

st.divider()
# ---------------------------------------------------------
# Visualization 3: Percentage improvement by question/topic
# ---------------------------------------------------------
st.subheader("Percentage improvement by question/topic")

improvement_plot = item_results.sort_values(
    "Improvement (pp)",
    ascending=True,
).reset_index(drop=True)

improvement_plot["Direction"] = np.select(
    [
        improvement_plot["Improvement (pp)"] > 0,
        improvement_plot["Improvement (pp)"] < 0,
    ],
    [
        "Improvement",
        "Decline",
    ],
    default="No change",
)

max_abs_value = max(
    abs(improvement_plot["Improvement (pp)"].min()),
    abs(improvement_plot["Improvement (pp)"].max()),
    5,
)

padding = max_abs_value * 0.12

fig_improvement = px.bar(
    improvement_plot,
    x="Improvement (pp)",
    y="Topic",
    orientation="h",
    color="Direction",
    category_orders={
        "Direction": ["Improvement", "No change", "Decline"],
    },
    color_discrete_map={
        "Improvement": "green",
        "Decline": "red",
        "No change": "grey",
    },
    title="Knowledge growth ranked by improvement",
    custom_data=[
        "Question",
        "Before correct (%)",
        "Post correct (%)",
        "Status",
    ],
)

fig_improvement.update_traces(
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Question: %{customdata[0]}<br>"
        "Before correct: %{customdata[1]:.1f}%<br>"
        "After correct: %{customdata[2]:.1f}%<br>"
        "Change: %{x:+.1f} percentage points<br>"
        "Status: %{customdata[3]}"
        "<extra></extra>"
    ),
    text=None,
)

fig_improvement.add_vline(
    x=0,
    line_width=1,
    line_color="black",
)

fig_improvement.update_layout(
    xaxis=dict(
        title="Change in correct responses (percentage points)",
        range=[
            -(max_abs_value + padding),
            max_abs_value + padding,
        ],
        zeroline=False,
    ),
    yaxis=dict(
        title=None,
        categoryorder="array",
        categoryarray=improvement_plot["Topic"].tolist(),
    ),
    legend_title_text="Direction",
    hovermode="closest",
    height=max(420, 70 * len(improvement_plot)),
)

st.plotly_chart(
    fig_improvement,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    },
)
# ---------------------------------------------------------
# Visualization 4: Understanding by question/topic
# ---------------------------------------------------------
st.subheader("Understanding by question/topic")

understanding_plot = item_results.melt(
    id_vars=[
        "Question",
        "Topic",
        "Improvement (pp)",
        "Status",
    ],
    value_vars=[
        "Before correct (%)",
        "Post correct (%)",
    ],
    var_name="Assessment",
    value_name="Correct responses (%)",
)

understanding_plot["Assessment"] = understanding_plot[
    "Assessment"
].replace(
    {
        "Before correct (%)": "Before",
        "Post correct (%)": "After",
    }
)

fig_item = px.bar(
    understanding_plot,
    x="Correct responses (%)",
    y="Topic",
    color="Assessment",
    orientation="h",
    barmode="group",
    category_orders={
        "Assessment": ["Before", "After"],
        "Topic": item_results["Topic"].tolist(),
    },
    title="Student understanding before and after the lesson",
    custom_data=[
        "Question",
        "Improvement (pp)",
        "Status",
    ],
)

fig_item.update_traces(
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Question: %{customdata[0]}<br>"
        "Assessment: %{fullData.name}<br>"
        "Students answering correctly: %{x:.1f}%<br>"
        "Overall change: %{customdata[1]:+.1f} percentage points<br>"
        "Status: %{customdata[2]}"
        "<extra></extra>"
    ),
)

fig_item.update_layout(
    xaxis=dict(
        title="Students answering correctly (%)",
        range=[0, 100],
        dtick=20,
    ),
    yaxis_title=None,
    legend_title_text="Assessment",
    hovermode="closest",
    height=max(420, 80 * len(item_results)),
)

st.plotly_chart(
    fig_item,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    },
)
# ---------------------------------------------------------
# Visualization 4b: Confident misconceptions (only with a confidence column)
# ---------------------------------------------------------
if use_confidence:
    st.subheader("Confident misconceptions by question/topic")
    st.caption(
        "A confident misconception is an answer the student was sure about but got wrong. "
        "These are the hardest ideas to change, so a drop after the lesson is strong evidence of learning."
    )

    confidence_plot = item_results.melt(
        id_vars=["Question", "Topic"],
        value_vars=["Confident errors before (%)", "Confident errors after (%)"],
        var_name="Assessment",
        value_name="Students sure but wrong (%)",
    )
    confidence_plot["Assessment"] = confidence_plot["Assessment"].replace(
        {
            "Confident errors before (%)": "Before",
            "Confident errors after (%)": "After",
        }
    )

    fig_confidence = px.bar(
        confidence_plot,
        x="Students sure but wrong (%)",
        y="Topic",
        color="Assessment",
        orientation="h",
        barmode="group",
        category_orders={
            "Assessment": ["Before", "After"],
            "Topic": item_results["Topic"].tolist(),
        },
        title="Students who were sure of a wrong answer",
        custom_data=["Question"],
    )
    fig_confidence.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Question: %{customdata[0]}<br>"
            "Assessment: %{fullData.name}<br>"
            "Sure but wrong: %{x:.1f}% of students"
            "<extra></extra>"
        ),
    )
    confidence_max = max(confidence_plot["Students sure but wrong (%)"].max(), 10)
    fig_confidence.update_layout(
        xaxis=dict(
            title="Students sure of a wrong answer (%)",
            range=[0, min(100, confidence_max * 1.15)],
        ),
        yaxis_title=None,
        legend_title_text="Assessment",
        hovermode="closest",
        height=max(380, 80 * len(item_results)),
    )
    st.plotly_chart(
        fig_confidence,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

    total_before_ce = int(student_results["Confident errors before"].sum())
    total_after_ce = int(student_results["Confident errors after"].sum())
    st.write(
        f"Across the class there were **{total_before_ce}** confident misconceptions before "
        f"the lesson and **{total_after_ce}** after."
    )

# ---------------------------------------------------------
# Visualization 5: Normalised learning gain by student
# ---------------------------------------------------------
st.subheader("Normalised learning gain by student")

plot_results = (
    student_results.dropna(subset=["Normalised gain (g)"])
    .sort_values("Normalised gain (g)", ascending=False)
    .copy()
)
if ceiling_count:
    st.caption(
        f"{ceiling_count} student(s) at ceiling (full marks before the lesson) are not shown."
    )

fig_gain = px.bar(
    plot_results,
    x=id_col,
    y="Normalised gain (g)",
    color="Gain category",
    category_orders={
        "Gain category": [
            "Low gain",
            "Medium gain",
            "High gain",
        ],
    },
    title="Normalised learning gain by student",
    custom_data=[
        "Before score",
        "After score",
        "Before score (%)",
        "After score (%)",
        "Score change",
        "Gain category",
    ],
)

fig_gain.update_traces(
    hovertemplate=(
        "<b>Student: %{x}</b><br>"
        "Before score: %{customdata[0]} "
        "(%{customdata[2]:.1f}%)<br>"
        "After score: %{customdata[1]} "
        "(%{customdata[3]:.1f}%)<br>"
        "Score change: %{customdata[4]:+d}<br>"
        "Normalised gain: %{y:.2f}<br>"
        "Category: %{customdata[5]}"
        "<extra></extra>"
    ),
)

fig_gain.add_hline(
    y=0.7,
    line_dash="dash",
    annotation_text="High-gain threshold (0.7)",
    annotation_position="top left",
)

fig_gain.add_hline(
    y=0.3,
    line_dash="dash",
    annotation_text="Medium-gain threshold (0.3)",
    annotation_position="top left",
)

fig_gain.add_hline(
    y=0,
    line_width=1,
    line_color="black",
)

if plot_results.empty:
    gain_min, gain_max = 0.0, 0.0
else:
    gain_min = plot_results["Normalised gain (g)"].min()
    gain_max = plot_results["Normalised gain (g)"].max()

gain_padding = max(
    (gain_max - gain_min) * 0.12,
    0.1,
)

fig_gain.update_layout(
    xaxis=dict(
        title="Student identifier",
        tickangle=-60,
    ),
    yaxis=dict(
        title="Normalised gain (g)",
        range=[
            min(gain_min - gain_padding, -0.1),
            max(gain_max + gain_padding, 0.8),
        ],
    ),
    legend_title_text="Gain category",
    hovermode="closest",
    height=550,
)

st.plotly_chart(
    fig_gain,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    },
)
# ---------------------------------------------------------
# Item analysis section
# ---------------------------------------------------------

st.subheader("Item analysis")
item_formats = {
    column: "{:.1f}"
    for column in item_results.columns
    if column.endswith("(%)") or column.endswith("(pp)")
}
st.dataframe(
    item_results.style.format(item_formats),
    width="stretch",
)

# ---------------------------------------------------------
# Student results section
# ---------------------------------------------------------
st.subheader("Student results")
st.dataframe(
    student_results.style.format(
        {
            "Normalised gain (g)": "{:.2f}",
            "Before score (%)": "{:.1f}",
            "After score (%)": "{:.1f}",
        },
        na_rep="–",
    ),
    width="stretch",
)

# ---------------------------------------------------------
# Download section
# ---------------------------------------------------------
st.header("4. Download outputs")

excel_bytes = dataframe_to_excel_bytes(
    student_results,
    item_results,
)

button1, button2, button3, button4 = st.columns(4)

with button1:
    st.download_button(
        "Download workbook",
        data=excel_bytes,
        file_name="pre_post_learning_analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

with button2:
    st.download_button(
        "Download improvement chart",
        data=fig_improvement.to_html(
            include_plotlyjs=True,
            full_html=True,
        ).encode("utf-8"),
        file_name="percentage_improvement.html",
        mime="text/html",
    )

with button3:
    st.download_button(
        "Download understanding chart",
        data=fig_item.to_html(
            include_plotlyjs=True,
            full_html=True,
        ).encode("utf-8"),
        file_name="item_understanding.html",
        mime="text/html",
    )

with button4:
    st.download_button(
        "Download student gain chart",
        data=fig_gain.to_html(
            include_plotlyjs=True,
            full_html=True,
        ).encode("utf-8"),
        file_name="student_learning_gain.html",
        mime="text/html",
    )

st.caption(
    "Normalised gain: g = (post-score − pre-score) / (maximum score − pre-score). "
    "Students with a maximum pre-score are reported as 'At ceiling' and left out of the mean gain, "
    "because no additional gain is available."
)
