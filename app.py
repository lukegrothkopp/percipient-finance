# app.py NEW

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re
import numpy as np
import os

# Page configuration
from pathlib import Path
import os

# Resolve repo-relative path to the logo
REPO_DIR = Path(__file__).parent
LOGO_PATH = REPO_DIR / "assets" / "header_logo.png"   # <-- your file in the repo

# Helper: return a usable icon value for Streamlit (path if exists, else emoji)
def page_icon_value():
    return str(LOGO_PATH) if LOGO_PATH.exists() else "📈"

st.set_page_config(
    page_title="Grothko Consulting's Business Intelligence Generator, AKA BIG)",
    page_icon=page_icon_value(),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- API key loader: prefer Streamlit Secrets, then env var, then sidebar fallback ---
def get_openai_api_key():
    """
    Load the OpenAI API key with the following precedence:
    1) st.secrets["OPENAI_API_KEY"]
    2) st.secrets["openai"]["api_key"]
    3) os.environ["OPENAI_API_KEY"] (if already set by your host)
    Returns None if not found.
    """
    key = None
    try:
        # Common flat key
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        try:
            # Optional nested structure: [openai]["api_key"]
            key = st.secrets["openai"]["api_key"]
        except Exception:
            key = os.environ.get("OPENAI_API_KEY")
    return key

# -------------------------
# Session State
# -------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"assistant", "content": str}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = ""
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
page = st.session_state.get("page", "Dashboard")

# -------------------------
# Helpers
# -------------------------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _find_col(df: pd.DataFrame, name: str):
    """
    Fuzzy / case-insensitive column matcher with semantic aliases.
    Handles percent/percentage/perc <-> pct and common business synonyms.
    """
    def alias_set(s: str):
        alts = {s}
        if 'percentage' in s:
            alts.update({s.replace('percentage','pct'), s.replace('percentage','percent')})
        if 'percent' in s:
            alts.update({s.replace('percent','pct'), s.replace('percent','percentage')})
        if 'perc' in s:
            alts.update({s.replace('perc','pct'), s.replace('perc','percent'), s.replace('perc','percentage')})
        if 'pct' in s:
            alts.update({s.replace('pct','percent'), s.replace('pct','percentage'), s.replace('pct','perc')})
        return alts

    # semantic aliases -> preferred df columns
    semantic_aliases = {
        'price':        ['unit_price','price','avg_price','averageprice'],
        'revenue':      ['net_revenue','revenue','sales','totalrevenue','netsales'],
        'quantity':     ['units','quantity','qty','unitssold','unit_sold','units_sold'],
        'qty':          ['units','quantity','qty','unitssold','units_sold'],
        'date':         ['purchase_date','order_date','date','purchasedate','orderdate'],
        'customer':     ['customer_id','customerid','cust_id','custid','id'],
        'city':         ['city','town'],
        'state':        ['state_province','state','province','region','stateprovince'],
        'province':     ['state_province','province','state'],
        'country':      ['country','nation'],
        'income':       ['household_income','income','householdincome'],
        'maritalstatus':['marital_status','maritalstatus'],
        'children':     ['number_children','children','numchildren','childcount'],
        'childrenages': ['children_ages','childrenages','kidsages'],
        'subscription': ['subscription_member','member','ismember','loyalty'],
        'marketing':    ['marketing_source','marketingsource','utm_source','source'],
        'acquisition':  ['acquisition_channel','acquisitionchannel','channel','utm_medium'],
        'format':       ['preferred_format','format'],
        'discount':     ['discount_pct','discount','discountpercent','discountpercentage','pctdiscount'],
        'coupon':       ['coupon_used','coupon','promocode','discountcode'],
        'gift':         ['gifting','gift','gifted'],
        'feedbackscore':['feedback_score','rating','score','satisfaction'],
        'feedback':     ['feedback_text','comment','review','feedback'],
        'recommend':    ['would_recommend','recommend','nps','promoter','wouldrecommend'],
        'nps':          ['would_recommend','nps'],
        'repeat':       ['repeat_buyer','repeat','returningcustomer','returning'],
        'timetofinish': ['time_to_finish_days','timetofinish','completiontimedays','completiontime'],
        'occupation':   ['occupation','job','profession'],
        'age':          ['age'],
    }

    # map df norm -> actual
    norm2actual = { _norm(c): c for c in df.columns }

    target = _norm(name)
    target_alts = alias_set(target)

    # 1) exact / alias exact
    for ta in [target, *list(target_alts)]:
        if ta in norm2actual: return norm2actual[ta]

    # 2) contains
    for ta in [target, *list(target_alts)]:
        for cn, actual in norm2actual.items():
            if ta in cn or cn in ta:
                return actual

    # 3) semantic
    def concept_keys(t: str):
        concept_map = {
            'price': ['price','unitprice','avgprice','averageprice'],
            'revenue': ['revenue','netsales','sales','totalrevenue'],
            'quantity': ['quantity','qty','units'],
            'qty': ['qty','quantity','units'],
            'date': ['date','orderdate','purchasedate'],
            'customer': ['customer','customerid','custid'],
            'state': ['state','province','stateprovince','region'],
            'province': ['province','state'],
            'country': ['country','nation'],
            'income': ['income','householdincome'],
            'maritalstatus': ['maritalstatus','marital'],
            'children': ['children','numchildren','childcount'],
            'childrenages': ['childrenages','kidsages'],
            'subscription': ['subscription','member','loyalty'],
            'marketing': ['marketing','utm','source'],
            'acquisition': ['acquisition','channel','utm'],
            'format': ['format','preferredformat'],
            'discount': ['discount','discountpercent','discountpercentage','pctdiscount'],
            'coupon': ['coupon','promocode','discountcode'],
            'gift': ['gift','gifting','gifted'],
            'feedbackscore': ['feedbackscore','rating','score','satisfaction'],
            'feedback': ['feedback','comment','review'],
            'recommend': ['recommend','wouldrecommend','nps','promoter'],
            'nps': ['nps','recommend'],
            'repeat': ['repeat','returning'],
            'timetofinish': ['timetofinish','completiontime','completiontimedays'],
            'occupation': ['occupation','job','profession'],
            'age': ['age'],
        }
        keys=[]
        for k,hints in concept_map.items():
            if t==k or any(h in t for h in hints):
                keys.append(k)
        return keys

    hits = []
    for key in concept_keys(target): hits.append(key)
    for ta in target_alts:
        hits += concept_keys(ta)
    seen=set(); hits=[h for h in hits if not (h in seen or seen.add(h))]

    for concept in hits:
        if concept in semantic_aliases:
            for cand in semantic_aliases[concept]:
                cn = _norm(cand)
                if cn in norm2actual: return norm2actual[cn]
                for nn, actual in norm2actual.items():
                    if cn in nn or nn in cn:
                        return actual

    # 4) last resort: soft contains across all
    for nn, actual in norm2actual.items():
        if any(tok in nn for tok in [target,*list(target_alts)]):
            return actual
    return None

def _first_datetime_col(df: pd.DataFrame):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        try:
            pd.to_datetime(df[c], errors='raise')
            return c
        except Exception:
            continue
    return None

# -------------------------
# Data summary for RAG
# -------------------------
def create_data_summary(df: pd.DataFrame) -> str:
    parts=[]
    parts.append(f"Dataset Overview: The dataset contains {len(df)} records and {df.shape[1]} columns.")
    parts.append(f"Columns: {', '.join(df.columns.tolist())}")
    numeric=df.select_dtypes(include=['float64','int64','float32','int32']).columns
    if len(numeric)>0:
        parts.append("\nNumeric Columns Analysis:")
        for col in numeric:
            try:
                parts.append(f"\n{col}:")
                parts.append(f"  - Mean: {df[col].mean():.2f}")
                parts.append(f"  - Median: {df[col].median():.2f}")
                parts.append(f"  - Std Dev: {df[col].std():.2f}")
                parts.append(f"  - Min: {df[col].min():.2f}")
                parts.append(f"  - Max: {df[col].max():.2f}")
            except Exception: pass
    cats=df.select_dtypes(include=['object']).columns
    if len(cats)>0:
        parts.append("\nCategorical Columns Analysis:")
        for col in cats:
            try:
                uniq=df[col].nunique()
                parts.append(f"\n{col}:")
                parts.append(f"  - Unique values: {uniq}")
                if uniq<=10:
                    vc=df[col].value_counts()
                    parts.append(f"  - Distribution: {vc.to_dict()}")
            except Exception: pass
    missing=df.isnull().sum()
    if missing.sum()>0:
        parts.append("\nMissing Data:")
        for col in missing[missing>0].index:
            parts.append(f"  - {col}: {missing[col]} missing values")
    completeness=(1 - df.isnull().sum().sum() / max(1,(df.shape[0]*df.shape[1]))) * 100
    parts.append("\nKey Insights:")
    parts.append(f"  - Total rows: {len(df)}")
    parts.append(f"  - Data completeness: {completeness:.1f}%")
    return "\n".join(parts)

# -------------------------
# RAG setup (no chains/memory)
# -------------------------
def setup_rag_system(df: pd.DataFrame, api_key: str) -> bool:
    try:
        os.environ["OPENAI_API_KEY"]=api_key
        summary=create_data_summary(df)
        st.session_state.data_summary=summary
        data_insights=f"""
{summary}

Sample Data Records (first 20):
{df.head(20).to_string()}

Column Data Types and Info:
{df.dtypes.to_string()}
"""
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks=splitter.split_text(data_insights)
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore=FAISS.from_texts(chunks, embeddings)
        retriever=vectorstore.as_retriever(search_kwargs={"k":3})
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        st.session_state.vectorstore=vectorstore
        st.session_state.retriever=retriever
        st.session_state.llm=llm
        return True
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        return False

def answer_with_rag(question: str):
    retriever=st.session_state.retriever
    source_docs=[]; context=""
    if retriever is not None:
        try:
            source_docs=retriever.invoke(question)  # Runnable retriever
            context="\n\n".join(getattr(doc,"page_content",str(doc)) for doc in source_docs)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            source_docs=[]; context=""
    history_tail=st.session_state.chat_history[-6:]
    history_text="\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history_tail])
    system_text=(
        "You are a helpful business data analyst. Use the provided context (from the uploaded CSV) "
        "to answer the user's question. If the answer is not in the context, say you don't have enough "
        "information and suggest what to compute or inspect."
    )
    prompt=(f"{system_text}\n\nChat history:\n{history_text}\n\nContext:\n{context}\n\n"
            f"User question: {question}\n\nAnswer clearly and concisely:")
    try:
        resp=st.session_state.llm.invoke(prompt)
        answer=getattr(resp,"content",str(resp))
        return answer, source_docs
    except Exception as e:
        return f"Error generating response: {e}", source_docs

# -------------------------
# Structured Analytics Engine
# -------------------------
def _resolve_condition_phrase(df: pd.DataFrame, phrase: str):
    """
    Infer a condition column + value/polarity from natural language like:
      - "customers who recommend the books"
      - "users who do not recommend"
      - "coupon used = yes"
      - "subscribers" / "subscription members"
    Returns: (cond_col, explicit_value, polarity)
      - cond_col: matched column name or None
      - explicit_value: string value if explicitly provided (e.g., "yes"), else None
      - polarity: "truthy" or "falsy" if implied (no explicit value)
    """
    p_raw = phrase.strip()
    p = p_raw.lower()

    # 0) Strip common leading noise that can bias matching to customer_id
    p = re.sub(r'^(customers?|users?|people)\s+(who|that|which)\s+', '', p).strip()

    # 1) If there's an explicit equality, pull it out first.
    #    Handles: '=', ' is ', ' equals '
    eq_match = re.search(r'(=| is | equals )\s*([^\s].+)$', p)
    explicit_value = None
    if eq_match:
        explicit_value = eq_match.group(2).strip().strip('"\'').lower()
        p = p[:eq_match.start()].strip()

    # 2) Polarity hints for boolean-like columns
    falsy_hints = (" do not ", " don't ", " not ", " no ", " non ", " without ")
    polarity = "truthy"
    if any(h in f" {p} " for h in falsy_hints):
        polarity = "falsy"

    # 3) HIGH-PRIORITY intent keywords → map directly to their likely columns first
    #    These are predicate-like fields we *want* to pick for phrases like "who recommend".
    intent_priority = [
        ("recommend", ["recommend", "recommended", "recommends", "would recommend", "nps", "promoter"]),
        ("coupon",    ["coupon", "promo", "code", "discount code", "coupon used", "used coupon"]),
        ("subscription", ["subscription", "subscriber", "member", "membership", "loyalty"]),
        ("repeat",    ["repeat", "returning", "again"]),
        ("gift",      ["gift", "gifting", "gifted"]),
        ("feedback",  ["feedback", "rating", "review", "satisfaction"]),
    ]

    # helper to try a concept directly
    def _try_concept(concept: str):
        # Reuse your alias matcher to resolve the column for a concept word
        col = _find_col(df, concept)
        return col

    for concept, tokens in intent_priority:
        if any(t in p for t in tokens):
            col = _try_concept(concept)
            if col:
                return col, explicit_value, polarity

    # 4) If no priority concept matched, try the whole phrase → column
    col = _find_col(df, p)
    if col:
        return col, explicit_value, polarity

    # 5) Token scan fallback (prefer longer tokens)
    tokens = re.findall(r'[a-z0-9_%]+', p)
    tokens_sorted = sorted(tokens, key=len, reverse=True)
    for tok in tokens_sorted:
        col = _find_col(df, tok)
        if col:
            return col, explicit_value, polarity

    # 6) Bigram fallback
    for i in range(len(tokens_sorted) - 1):
        bg = tokens_sorted[i] + " " + tokens_sorted[i + 1]
        col = _find_col(df, bg)
        if col:
            return col, explicit_value, polarity

    return None, explicit_value, polarity

def _normalize_boolish(series: pd.Series) -> pd.Series:
    """
    Return a Series of strings normalized to boolean-like categories:
      - truthy set: {'1','true','t','yes','y','recommended','recommend','member','used'}
      - falsy set:  {'0','false','f','no','n','notrecommended','dontrecommend','do_not_recommend','unused'}
    Non-matching values are returned as lower-cased strings (so we can still do explicit equality if needed).
    """
    s = series.astype(str).str.lower().str.strip()
    truthy = {'1','true','t','yes','y','recommended','recommend','member','used'}
    falsy  = {'0','false','f','no','n','notrecommended','dontrecommend','do_not_recommend','unused'}

    def map_val(v):
        if v in truthy: return '___truthy___'
        if v in falsy:  return '___falsy___'
        return v

    return s.map(map_val)

def handle_analytics_query(question: str, df: pd.DataFrame):
    """
    Detect and execute common analytics directly on the dataframe.
    Returns (handled: bool, message: str).
    """
    q=question.lower().strip()

    # --- Correlation between X and Y ---
    m=re.search(r'correlat\w*\s+.*\bbetween\b\s+(.+?)\s+\b(and|&)\b\s+(.+)', q)
    if m:
        raw_x=m.group(1); raw_y=m.group(3)
        col_x=_find_col(df, raw_x); col_y=_find_col(df, raw_y)
        if not col_x or not col_y:
            return True, (f"I couldn't match both columns.\nMatched X: `{col_x or 'None'}` | "
                          f"Matched Y: `{col_y or 'None'}`.\nColumns: {list(df.columns)}")
        x=pd.to_numeric(df[col_x], errors='coerce'); y=pd.to_numeric(df[col_y], errors='coerce')
        valid=x.notna() & y.notna()
        if valid.sum()<3:
            return True, f"Not enough overlapping numeric values to compute correlation between `{col_x}` and `{col_y}`."
        r=x[valid].corr(y[valid], method='pearson')
        msg=f"**Pearson r** between `{col_x}` and `{col_y}`: **{r:.3f}** (n={valid.sum()})"
        st.write(msg)
        try:
            fig=px.scatter(pd.DataFrame({col_x:x[valid], col_y:y[valid]}), x=col_x, y=col_y,
                           title=f"Scatter: {col_x} vs {col_y}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception: pass
        return True, msg

    # --- Explicit Pearson request ---
    m2=re.search(r'(pearson|corr(?!elation)\b).*?(between|for)\s+[\'"]?([\w %_]+)[\'"]?\s+(and|,)\s+[\'"]?([\w %_]+)[\'"]?', q)
    if m2:
        raw_x=m2.group(3); raw_y=m2.group(5)
        col_x=_find_col(df, raw_x); col_y=_find_col(df, raw_y)
        if not col_x or not col_y:
            return True, (f"I couldn't match both columns.\nMatched X: `{col_x or 'None'}` | "
                          f"Matched Y: `{col_y or 'None'}`.\nColumns: {list(df.columns)}")
        x=pd.to_numeric(df[col_x], errors='coerce'); y=pd.to_numeric(df[col_y], errors='coerce')
        valid=x.notna() & y.notna()
        if valid.sum()<3:
            return True, f"Not enough overlapping numeric values to compute correlation between `{col_x}` and `{col_y}`."
        r=x[valid].corr(y[valid], method='pearson')
        msg=f"**Pearson r** between `{col_x}` and `{col_y}`: **{r:.3f}** (n={valid.sum()})"
        st.write(msg)
        try:
            fig=px.scatter(pd.DataFrame({col_x:x[valid], col_y:y[valid]}), x=col_x, y=col_y,
                           title=f"Scatter: {col_x} vs {col_y}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception: pass
        return True, msg

    # --- Conditional average/mean: natural language friendly ---
    m_avg = re.search(
        r'\b(average|avg|mean)\b\s+(of\s+|for\s+)?(?P<metric>[\w %_]+?)\s+(among|for|where)\s+(?P<condphrase>.+)$',
        q
    )
    if m_avg:
        raw_metric = m_avg.group('metric').strip()
        cond_phrase = m_avg.group('condphrase').strip()

        metric_col = _find_col(df, raw_metric)
        if not metric_col:
            return True, (f"I couldn't resolve the requested metric column.\n"
                          f"Metric: `{raw_metric}`\nColumns: {list(df.columns)}")

        cond_col, explicit_val, polarity = _resolve_condition_phrase(df, cond_phrase)
        if not cond_col:
            return True, (f"I couldn't resolve the condition column from: `{cond_phrase}`.\n"
                          f"Columns: {list(df.columns)}")

        # Build filter
        raw_series = df[cond_col]
        norm_series = _normalize_boolish(raw_series)

        if explicit_val:
            # explicit string equality (post-normalization)
            target = explicit_val.lower().strip()
            cond_mask = norm_series.eq(target) | raw_series.astype(str).str.lower().str.strip().eq(target)
        else:
            flag = '___truthy___' if polarity == 'truthy' else '___falsy___'
            cond_mask = norm_series.eq(flag)

            # If nothing matched: try booleans or 0/1 numerics directly
            if not cond_mask.any():
                if raw_series.dtype == bool:
                    cond_mask = raw_series.fillna(False) if polarity == "truthy" else ~raw_series.fillna(False)
                else:
                    try:
                        num_series = pd.to_numeric(raw_series, errors='coerce').fillna(0)
                        cond_mask = (num_series != 0) if polarity == "truthy" else (num_series == 0)
                    except Exception:
                        pass

        metric_vals = pd.to_numeric(df.loc[cond_mask, metric_col], errors='coerce').dropna()
        n = metric_vals.shape[0]
        if n == 0:
            pretty_val = (f"= {explicit_val}" if explicit_val else (" is truthy" if polarity=="truthy" else " is falsy"))
            return True, (f"No matching rows for **{cond_col}** {pretty_val}.")

        mean_val = metric_vals.mean()
        pretty_val = (f"= {explicit_val}" if explicit_val else (" is truthy" if polarity=="truthy" else " is falsy"))
        msg = (f"**Average {metric_col}** for rows where **{cond_col}** {pretty_val}: "
               f"**{mean_val:.2f}** (n={n})")
        st.write(msg)

        try:
            fig = px.histogram(metric_vals, nbins=20, title=f"Distribution of {metric_col} (filtered)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        return True, msg
        
    # --- "top 3 insights" ---
    if re.search(r'\btop\s*3\s*insight', q):
        insights=[]
        num_df=df.select_dtypes(include=[np.number])
        if num_df.shape[1]>=2:
            corr=num_df.corr().abs()
            np.fill_diagonal(corr.values,0)
            max_pair=divmod(corr.values.argmax(), corr.shape[1])
            a,b=num_df.columns[max_pair[0]], num_df.columns[max_pair[1]]
            insights.append(f"Strongest numeric relationship: **{a}** vs **{b}** (|r| ≈ **{corr.values.max():.3f}**).")
        cat_cols=df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            vc=df[cat_cols[0]].value_counts(dropna=True).head(3)
            insights.append(f"Top categories in **{cat_cols[0]}**: " + ", ".join([f'{k} ({v})' for k,v in vc.items()]))
        if num_df.shape[1]>=1 and cat_cols:
            grp=df.groupby(cat_cols[0])[num_df.columns[0]].mean().sort_values(ascending=False).head(3)
            insights.append(f"Highest average **{num_df.columns[0]}** by **{cat_cols[0]}**: " +
                            ", ".join([f"{idx} ({val:.2f})" for idx,val in grp.items()]))
        if not insights:
            return True, "I couldn't derive quick insights—dataset might lack numeric/categorical variety."
        msg="### Top 3 Insights\n" + "\n".join([f"{i+1}. {insights[i]}" for i in range(min(3,len(insights)))])
        return True, msg

    # --- "trends or patterns" ---
    if re.search(r'\btrends?\b|\bpatterns?\b', q):
        bullets=[]
        num_df=df.select_dtypes(include=[np.number])
        date_col=_first_datetime_col(df)
        if date_col:
            try:
                df2=df.copy()
                df2[date_col]=pd.to_datetime(df2[date_col], errors='coerce')
                if num_df.shape[1]>=1:
                    ycol=num_df.columns[0]
                    ts=df2.dropna(subset=[date_col, ycol]).sort_values(date_col)
                    if len(ts)>=3:
                        ts['__period']=ts[date_col].dt.to_period('M').dt.to_timestamp()
                        agg=ts.groupby('__period')[ycol].mean()
                        direction="increasing" if agg.iloc[-1]>agg.iloc[0] else "decreasing"
                        bullets.append(f"**Time trend** in **{ycol}**: appears {direction} from first to last observed month.")
                        try:
                            fig=px.line(agg.reset_index(), x='__period', y=ycol, title=f"Trend of {ycol} over time")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception: pass
            except Exception: pass
        if num_df.shape[1]>=2:
            corr=num_df.corr(); corr_abs=corr.abs()
            np.fill_diagonal(corr_abs.values,0)
            max_pair=divmod(corr_abs.values.argmax(), corr_abs.shape[1])
            a,b=num_df.columns[max_pair[0]], num_df.columns[max_pair[1]]
            bullets.append(f"**Strongest numeric relationship**: {a} vs {b} (|r| ≈ {corr_abs.values.max():.3f}).")
        if not bullets:
            bullets.append("No clear trends/patterns detected automatically. Try asking about specific columns or groups.")
        msg="### Detected Trends & Patterns\n" + "\n".join([f"- {b}" for b in bullets])
        return True, msg

    return False, ""

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=150)
    else:
        st.sidebar.markdown("### GROTHKO CONSULTING")

    st.title("Navigation")

    # API Key input (prefer secrets)
    st.subheader("🗝️ API Configuration")
    
    # --- Navigation ---
    pages = ["Dashboard", "Data Analysis", "AI Assistant", "Visualizations"]
    current_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
    selection = st.radio("Select Page:", pages, index=current_index)
    st.session_state.page = selection

    # Prefer Streamlit Secrets (or env) first
    api_key = get_openai_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("💪 API key verified")
    else:
        st.info("ℹ️ You can store your key in Streamlit Secrets as `OPENAI_API_KEY` (or under `openai.api_key`).")

    st.subheader("🗄️ Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)", type=['csv'],
        help="Upload a CSV file containing your business data"
    )

    if uploaded_file is not None and api_key:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"☑️ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            if st.session_state.vectorstore is None:
                with st.spinner("Setting up AI system..."):
                    if setup_rag_system(df, api_key):
                        st.success("🦾 AI system ready!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif uploaded_file is not None and not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key first")

# -------------------------
# Main content
# -------------------------
st.markdown('<h1 class="main-header">Grothko Consulting Business Intelligence Generator</h1>', unsafe_allow_html=True)

# Dashboard Page
page = st.session_state.get("page", "Dashboard")
if page == "Dashboard":
    st.header("📈 BI Dashboard")
    if st.session_state.data_loaded:
        df = st.session_state.df
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric(label="Total Records", value=f"{len(df):,}", delta="Active")
        with c2: st.metric(label="Data Columns", value=df.shape[1], delta="Features")
        with c3:
            numeric_cols = df.select_dtypes(include=['float64','int64']).columns
            if len(numeric_cols)>0:
                st.metric(label=f"Avg {numeric_cols[0]}", value=f"{df[numeric_cols[0]].mean():.2f}")
        with c4:
            completeness=(1 - df.isnull().sum().sum() / max(1,(df.shape[0]*df.shape[1]))) * 100
            st.metric(label="Data Quality", value=f"{completeness:.1f}%", delta="Complete")
        st.divider()
        st.subheader("🗃️ Data Preview"); st.dataframe(df.head(10), use_container_width=True)
        st.subheader("💡 Quick Statistics"); st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("👈 Please upload a dataset and enter your OpenAI API Key to get started.")
        st.markdown("""
        ### Getting Started
        1. Create an API key at the OpenAI Platform
        2. Enter the API key in the sidebar
        3. Upload your CSV file
        4. Explore AI-powered insights!
        """)

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🛰️ Advanced Data Analysis")
    if st.session_state.data_loaded:
        df = st.session_state.df
        tab1,tab2,tab3 = st.tabs(["Column Analysis","Missing Data","Correlations"])
        with tab1:
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        with tab2:
            st.subheader("Missing Data Analysis")
            missing = df.isnull().sum()
            missing = missing[missing>0].sort_values(ascending=False)
            if len(missing)>0:
                fig = px.bar(x=missing.values, y=missing.index, orientation='h',
                             title="Missing Values by Column",
                             labels={'x':'Number of Missing Values','y':'Column'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing data found! 🎉")
        with tab3:
            st.subheader("Correlation Analysis")
            numeric_df = df.select_dtypes(include=['float64','int64'])
            if len(numeric_df.columns)>1:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Correlation Heatmap", color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
    else:
        st.warning("Please upload a dataset first.")

# AI Assistant Page
elif page == "AI Assistant":
    st.header("🧬 AI Assistant")
    if st.session_state.data_loaded and api_key and st.session_state.retriever and st.session_state.llm:
        st.info("💡 Ask questions about your data. I can compute correlations, conditional averages, top insights, and basic trends.")
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]): st.write(m["content"])
        user_question = st.chat_input("Ask a question about your data...")
        if user_question:
            st.session_state.chat_history.append({"role":"user","content":user_question})
            with st.chat_message("user"): st.write(user_question)
            handled, msg = handle_analytics_query(user_question, st.session_state.df)
            if handled:
                with st.chat_message("assistant"): st.write(msg)
                st.session_state.chat_history.append({"role":"assistant","content":msg})
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your data..."):
                        answer, src = answer_with_rag(user_question)
                        st.write(answer)
                        if src:
                            with st.expander("📚 View Source Context"):
                                for i,doc in enumerate(src):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(getattr(doc,"page_content",str(doc))[:300] + "...")
                        st.session_state.chat_history.append({"role":"assistant","content":answer})
        c1,c2 = st.columns([6,1])
        with c2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history=[]; st.rerun()
    else:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API Key in the sidebar.")
        elif not st.session_state.data_loaded:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            st.warning("⚠️ AI system is not ready. Please reload the page.")

# Visualizations Page
elif page == "Visualizations":
    st.header("📊 Data Visualizations")
    if st.session_state.data_loaded:
        df = st.session_state.df
        viz_type = st.selectbox("Select Visualization Type:",
                                ["Line Chart","Bar Chart","Scatter Plot","Pie Chart","Histogram","Box Plot"])
        c1,c2 = st.columns(2)
        with c1:
            numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if viz_type in ["Line Chart","Bar Chart"]:
                x_col = st.selectbox("Select X-axis:", df.columns.tolist())
                y_col = st.selectbox("Select Y-axis:", numeric_cols) if numeric_cols else None
            elif viz_type=="Scatter Plot":
                x_col = st.selectbox("Select X-axis:", numeric_cols)
                y_col = st.selectbox("Select Y-axis:", numeric_cols)
            elif viz_type=="Pie Chart":
                x_col = st.selectbox("Select Category:", categorical_cols if categorical_cols else df.columns.tolist())
                y_col = st.selectbox("Select Value:", numeric_cols) if numeric_cols else None
            elif viz_type in ["Histogram","Box Plot"]:
                x_col = st.selectbox("Select Column:", numeric_cols if numeric_cols else df.columns.tolist())
                y_col = None
        with c2:
            color_col = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist())
            color_col = None if color_col=="None" else color_col
        st.subheader(f"{viz_type} Visualization")
        try:
            if viz_type=="Line Chart" and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
            elif viz_type=="Bar Chart" and y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            elif viz_type=="Scatter Plot":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            elif viz_type=="Pie Chart" and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution of {y_col}")
            elif viz_type=="Histogram":
                fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
            elif viz_type=="Box Plot":
                fig = px.box(df, y=x_col, color=color_col, title=f"Box Plot of {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.warning("Please upload a dataset first.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Grothko Consulting B.I.G - Business Intelligence Generator</p>
</div>
""", unsafe_allow_html=True)
