# app.py — Percipient Finance (drop-in)

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
import json
from pathlib import Path

# ============== Page configuration & logo ==============

REPO_DIR = Path(__file__).parent
LOGO_PATH = REPO_DIR / "assets" / "header_logo.png"

def page_icon_value():
    return str(LOGO_PATH) if LOGO_PATH.exists() else "📈"

st.set_page_config(
    page_title="Grothko Consulting's Business Intelligence Generator (BIG)",
    page_icon=page_icon_value(),
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== Custom CSS ==============
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

# ============== API key loader ==============
def get_openai_api_key():
    """
    Precedence:
      1) st.secrets["OPENAI_API_KEY"]
      2) st.secrets["openai"]["api_key"]
      3) os.environ["OPENAI_API_KEY"]
    """
    key = None
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        try:
            key = st.secrets["openai"]["api_key"]
        except Exception:
            key = os.environ.get("OPENAI_API_KEY")
    return key

# ============== Session State ==============
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
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
if 'finance_mapping' not in st.session_state:
    st.session_state.finance_mapping = {}
if 'mapping_confirmed' not in st.session_state:
    st.session_state.mapping_confirmed = False
if 'budget_df' not in st.session_state:
    st.session_state.budget_df = None

# ============== Finance canonical fields (Suggestion 0) ==============
# Canonical -> likely CSV column aliases
FINANCE_CANONICAL_FIELDS = {
    "date": ["date","order_date","invoice_date","transaction_date","purchase_date"],
    "customer_id": ["customer","customer_id","cust_id","account_id"],
    "vendor_id": ["vendor","vendor_id","supplier_id"],
    "sku": ["sku","product_id","item_id"],
    "quantity": ["quantity","qty","units","units_sold","unit_sold"],
    "revenue": ["revenue","sales","net_revenue","net_sales","amount"],
    "cogs": ["cogs","cost_of_goods_sold","cost"],
    "op_ex": ["opex","operating_expense","expense","operating_expenses"],
    "invoice_due_date": ["due_date","invoice_due","terms_due"],
    "invoice_paid_date": ["paid_date","payment_date","cash_date"],
    "ar_amount": ["ar","accounts_receivable","ar_amount"],
    "ap_amount": ["ap","accounts_payable","ap_amount"],
    "inventory_value": ["inventory","inventory_value","stock_value"],
    "cash_in": ["cash_in","collections","receipts"],
    "cash_out": ["cash_out","disbursements","payments"],
    # SaaS:
    "contract_start": ["contract_start","start_date"],
    "contract_end": ["contract_end","end_date","renewal_date"],
    "mrr": ["mrr","monthly_recurring_revenue"],
    "arr": ["arr","annual_recurring_revenue"],
    "logo_id": ["logo_id","account_id","customer_id"],
    "plan": ["plan","package","tier"],
    "is_churned": ["churn","is_churned","churned"],
    "is_expansion": ["expansion","upsell","add_on"],
    "discount": ["discount","discount_pct","discount_percent"],
    "s&m_expense": ["sales_marketing","sales_and_marketing","sm_expense"],

    # Extended keys (Suggestion 0):
    "unit_price": ["unit_price","price","avg_price","net_price","average_price"],
    "period": ["period","month","year_month","fiscal_period","ym","yyyy_mm"],
    "account": ["account","gl_account","budget_account","pnl_account"],
    "amount": ["amount","value","usd_amount"]  # for Budget CSV
}

MAPPING_STORE = REPO_DIR / ".if_mapping.json"

# Attempt to load persisted mapping
try:
    if MAPPING_STORE.exists():
        persisted = json.loads(MAPPING_STORE.read_text())
        if isinstance(persisted, dict):
            st.session_state.finance_mapping.update(persisted)
except Exception:
    pass

# ============== Helpers ==============
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _find_col(df: pd.DataFrame, name: str):
    """Fuzzy/semantic matcher."""
    def alias_set(s: str):
        alts = {s}
        if 'percentage' in s:
            alts.update({s.replace('percentage','pct'), s.replace('percentage','percent')})
    # ... (trimmed minor alias expansions for brevity)
        return alts

    semantic_aliases = {
        'price':        ['unit_price','price','avg_price','averageprice'],
        'revenue':      ['net_revenue','revenue','sales','totalrevenue','netsales'],
        'quantity':     ['units','quantity','qty','unitssold','units_sold'],
        'qty':          ['units','quantity','qty','unitssold','units_sold'],
        'date':         ['purchase_date','order_date','date','purchasedate','orderdate'],
        'customer':     ['customer_id','customerid','cust_id','custid','id'],
        'state':        ['state','province','stateprovince','region'],
        'country':      ['country','nation'],
        'discount':     ['discount_pct','discount','discountpercent','discountpercentage','pctdiscount'],
    }
    norm2actual = {_norm(c): c for c in df.columns}
    target = _norm(name)
    target_alts = alias_set(target)

    # exact/alias
    for ta in [target, *list(target_alts)]:
        if ta in norm2actual: return norm2actual[ta]

    # contains
    for ta in [target, *list(target_alts)]:
        for cn, actual in norm2actual.items():
            if ta in cn or cn in ta:
                return actual

    # semantic
    concept_map = {
        'price': ['price','unitprice','avgprice','averageprice'],
        'revenue': ['revenue','netsales','sales','totalrevenue'],
        'quantity': ['quantity','qty','units'],
        'date': ['date','orderdate','purchasedate'],
        'customer': ['customer','customerid','custid'],
        'discount': ['discount','pct','percent'],
    }
    hits = []
    for k,hints in concept_map.items():
        if target==k or any(h in target for h in hints):
            hits.append(k)
    seen=set(); hits=[h for h in hits if not (h in seen or seen.add(h))]
    for concept in hits:
        if concept in semantic_aliases:
            for cand in semantic_aliases[concept]:
                cn = _norm(cand)
                if cn in norm2actual: return norm2actual[cn]
                for nn, actual in norm2actual.items():
                    if cn in nn or nn in cn:
                        return actual

    # last resort
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

# ============== Data summary for RAG ==============
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

# ============== RAG setup ==============
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
            source_docs=retriever.invoke(question)
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

# ============== Structured Analytics Engine ==============

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
# >>> END: Structured Analytics <<<

# -----------------------
# Finance KPI Engine (existing baseline)
# -----------------------
def _col(name):  # convenience
    return st.session_state.finance_mapping.get(name)

def _parse_dates(df, cols):
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def compute_dso(df):
    inv = _col("date") or _col("invoice_date")
    paid = _col("invoice_paid_date")
    ar_amt = _col("ar_amount")
    rev = _col("revenue")
    if not inv or not (paid or ar_amt or rev):
        return None, "Missing required fields for DSO."
    df = df.copy()
    _parse_dates(df, [inv, paid])
    if ar_amt in df.columns and rev in df.columns:
        avg_ar = df[ar_amt].mean(skipna=True)
        total_rev = df[rev].sum(skipna=True)
        dso = (avg_ar / (total_rev/365.0)) if total_rev else np.nan
        return dso, None
    if paid in df.columns:
        mask = df[paid].notna() & df[inv].notna()
        if mask.sum()==0: return None, "No paid vs. invoice dates to compute realized DSO."
        durs = (df.loc[mask, paid] - df.loc[mask, inv]).dt.days
        return durs.mean(), None
    return None, "Insufficient data for DSO."

def compute_dpo(df):
    inv = _col("date")
    paid = _col("invoice_paid_date")
    ap_amt = _col("ap_amount")
    cogs = _col("cogs")
    if ap_amt in df.columns and cogs in df.columns:
        avg_ap = df[ap_amt].mean(skipna=True)
        total_cogs = df[cogs].sum(skipna=True)
        dpo = (avg_ap / (total_cogs/365.0)) if total_cogs else np.nan
        return dpo, None
    if inv and paid and inv in df.columns and paid in df.columns:
        _parse_dates(df, [inv, paid])
        mask = df[paid].notna() & df[inv].notna()
        if mask.sum()==0: return None, "No paid vs. bill dates to compute realized DPO."
        durs = (df.loc[mask, paid] - df.loc[mask, inv]).dt.days
        return durs.mean(), None
    return None, "Insufficient data for DPO."

def compute_dio(df):
    inv_val = _col("inventory_value")
    cogs = _col("cogs")
    if inv_val in df.columns and cogs in df.columns:
        avg_inv = df[inv_val].mean(skipna=True)
        total_cogs = df[cogs].sum(skipna=True)
        dio = (avg_inv / (total_cogs/365.0)) if total_cogs else np.nan
        return dio, None
    return None, "Insufficient data for DIO."

def compute_ccc(df):
    dso, e1 = compute_dso(df)
    dpo, e2 = compute_dpo(df)
    dio, e3 = compute_dio(df)
    if None not in (dso,dpo,dio):
        return (dso + dio - dpo), None
    return None, "Insufficient data for full CCC."

def gross_margin(df):
    rev = _col("revenue"); cogs = _col("cogs")
    if rev in df.columns and cogs in df.columns:
        total_rev = df[rev].sum(skipna=True)
        gm = (total_rev - df[cogs].sum(skipna=True))
        return gm, (gm/total_rev*100 if total_rev else np.nan)
    return None, None

def contribution_margin_by(df, by):
    rev = _col("revenue"); cogs = _col("cogs")
    if not (rev in df.columns and cogs in df.columns and by in df.columns):
        return None
    gp = df.groupby(by).agg({rev:"sum", cogs:"sum"})
    gp["gross_profit"] = gp[rev]-gp[cogs]
    gp["gm_pct"] = np.where(gp[rev]!=0, gp["gross_profit"]/gp[rev]*100, np.nan)
    return gp.sort_values("gross_profit", ascending=False)

def vendor_pareto(df):
    v = _col("vendor_id"); ap = _col("ap_amount")
    if v in df.columns and ap in df.columns:
        s = df.groupby(v)[ap].sum().sort_values(ascending=False)
        s_pct = s.cumsum()/s.sum()*100 if s.sum()!=0 else s.cumsum()
        return s, s_pct
    return None, None

def churn_and_retention(df):
    lid = _col("logo_id"); churn = _col("is_churned")
    if lid in df.columns and churn in df.columns:
        total = df[lid].nunique()
        churned = df.loc[df[churn]==True, lid].nunique()
        grr = (1 - churned/total)*100 if total else np.nan
        return {"logos": total, "churned": churned, "GRR_pct": grr}
    return None

def ltv_simple(arpu, gross_margin_pct, months):
    return arpu * (gross_margin_pct/100.0) * months

def cac_payback_months(cac, arpu, gm_pct):
    monthly_gp = arpu * (gm_pct/100.0)
    return cac / monthly_gp if monthly_gp else np.nan

def magic_number(delta_arr_quarter, prior_q_sm_expense):
    return (delta_arr_quarter*4.0) / prior_q_sm_expense if prior_q_sm_expense else np.nan

# -----------------------
# Optional-but-powerful features (Suggestions 1-4)
# -----------------------

# (1) 13-week cash view
def weekly_13_week_cash(df, opening_cash=0.0, min_cash_threshold=0.0):
    dt = _col("date"); ci = _col("cash_in"); co = _col("cash_out")
    if not dt or dt not in df.columns or not (ci in df.columns or co in df.columns):
        return None, "Map at least 'date' and one of 'cash_in' or 'cash_out' to compute 13-week cash."
    w = df.copy()
    _parse_dates(w, [dt])
    if ci not in w.columns: w[ci] = 0.0
    if co not in w.columns: w[co] = 0.0
    # bucket by Mon
    w["week"] = w[dt] - w[dt].dt.weekday.astype("timedelta64[D]")
    agg = w.groupby("week").agg(
        cash_in=(ci,"sum"),
        cash_out=(co,"sum")
    ).sort_index()
    agg["net"] = agg["cash_in"] - agg["cash_out"]
    agg["ending_cash"] = opening_cash + agg["net"].cumsum()
    agg["flag_low_cash"] = agg["ending_cash"] < float(min_cash_threshold)
    # show the most recent 13 weeks window anchored at current week
    start = pd.Timestamp.today().normalize() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit="D")
    future = agg[agg.index >= start]
    past = agg[agg.index < start].tail(3)  # small context
    agg = pd.concat([past, future]).head(13)
    return agg, None

# (2) BvA + forecast error
def join_budget_actuals(actuals_df, budget_df):
    per = _col("period"); acc = _col("account"); amt = _col("amount")
    dt  = _col("date"); rev = _col("revenue"); cogs = _col("cogs"); opx = _col("op_ex")

    if budget_df is None:
        return None, "Upload a Budget CSV."
    if per not in budget_df.columns or acc not in budget_df.columns or amt not in budget_df.columns:
        return None, "Budget CSV must have mapped 'period', 'account', and 'amount'."

    a = actuals_df.copy()
    if per not in a.columns:
        if dt in a.columns:
            a = _parse_dates(a, [dt])
            a[per] = a[dt].dt.to_period("M").astype(str)
        else:
            return None, "Map either 'period' or 'date' in actuals."

    pieces = []
    if rev in a.columns: pieces.append(a.groupby(per)[rev].sum().rename("amount").reset_index().assign(account="Revenue"))
    if cogs in a.columns: pieces.append(a.groupby(per)[cogs].sum().rename("amount").reset_index().assign(account="COGS"))
    if opx in a.columns: pieces.append(a.groupby(per)[opx].sum().rename("amount").reset_index().assign(account="Opex"))
    if not pieces:
        return None, "Need at least one of revenue, cogs, or op_ex in actuals to compute BvA."
    act = pd.concat(pieces, ignore_index=True)
    act = act.rename(columns={per:"period"})
    act["account"] = act["account"].astype(str)

    bud = budget_df[[per, acc, amt]].copy()
    bud = bud.rename(columns={per:"period", acc:"account", amt:"budget"})
    bud["account"] = bud["account"].astype(str)

    m = pd.merge(act, bud, on=["period","account"], how="left")
    m["variance"] = m["amount"] - m["budget"]
    m["var_pct"] = np.where(m["budget"]!=0, m["variance"]/m["budget"]*100, np.nan)
    return m, None

def forecast_error_metrics(df_period_vs, group_by_account=True):
    df = df_period_vs.dropna(subset=["amount","budget"]).copy()
    df["abs_pct"] = np.where(df["budget"]!=0, (df["amount"]-df["budget"]).abs()/df["budget"], np.nan)
    df["abs_err"] = (df["amount"]-df["budget"]).abs()
    df["abs_act"] = df["amount"].abs()
    if group_by_account:
        out = df.groupby("account").agg(
            MAPE=("abs_pct","mean"),
            WAPE=("abs_err","sum")
        )
        denom = df.groupby("account")["abs_act"].sum()
        out["WAPE"] = out["WAPE"] / denom.replace(0,np.nan)
        out["MAPE"] = out["MAPE"]*100; out["WAPE"]=out["WAPE"]*100
        return out.sort_values("WAPE", ascending=False)
    else:
        mape = df["abs_pct"].mean()*100
        wape = df["abs_err"].sum()/df["abs_act"].sum()*100 if df["abs_act"].sum()!=0 else np.nan
        return pd.DataFrame({"MAPE":[mape],"WAPE":[wape]})

# (3) PVM waterfall
def pvm_decomposition(df, period_a, period_b):
    sku = _col("sku"); qty = _col("quantity"); price = _col("unit_price") or _find_col(df,"price"); per = _col("period")
    if not all([sku, qty, price, per]) or any(c not in df.columns for c in [sku, qty, price, per]):
        return None, "Map 'sku', 'quantity', 'unit_price' (or 'price'), and 'period' to run PVM."
    d = df.copy()
    d = d[d[per].astype(str).isin([str(period_a), str(period_b)])]
    if d.empty: return None, "Selected periods not found."

    agg = d.groupby([per, sku]).agg(Q=(qty,"sum"), P=(price,"mean")).reset_index()
    a = agg[agg[per].astype(str)==str(period_a)].set_index(sku)[["Q","P"]].rename(columns={"Q":"Qa","P":"Pa"})
    b = agg[agg[per].astype(str)==str(period_b)].set_index(sku)[["Q","P"]].rename(columns={"Q":"Qb","P":"Pb"})
    j = a.join(b, how="outer").fillna(0.0)

    RevA = (j["Qa"]*j["Pa"]).sum()
    RevB = (j["Qb"]*j["Pb"]).sum()

    price_effect  = (j["Qb"]*(j["Pb"]-j["Pa"])).sum()
    volume_effect = (j["Pa"]*(j["Qb"]-j["Qa"])).sum()
    mix_effect    = ((j["Qb"]-j["Qa"])*(j["Pb"]-j["Pa"])).sum()

    out = pd.DataFrame({
        "Component":["Revenue A","Price","Volume","Mix","Revenue B"],
        "Value":[RevA, price_effect, volume_effect, mix_effect, RevB]
    })
    out["Delta"] = out["Value"].diff()
    return out, None

# (4) SaaS cohorts & NRR
def cohort_table_nrr(df):
    lid = _col("logo_id") or _col("customer_id")
    dt  = _col("date")
    mrr = _col("mrr") or _col("arr")
    if not (lid and dt and mrr) or any(c not in df.columns for c in [lid, dt, mrr]):
        return None, None, "Map 'logo_id' (or 'customer_id'), 'date', and 'mrr' (or 'arr')."
    x = df[[lid, dt, mrr]].copy()
    _parse_dates(x, [dt])
    x["ym"] = x[dt].dt.to_period("M").astype(str)
    first = x.groupby(lid)["ym"].min().rename("cohort")
    x = x.join(first, on=lid)
    cube = x.groupby(["cohort","ym"])[mrr].sum().unstack(fill_value=0.0)
    first_col = cube.columns[0] if len(cube.columns) else None
    if first_col is None:
        return cube, cube, None
    nrr = cube.divide(cube[first_col].replace(0,np.nan), axis=0)
    return cube, nrr, None

def overall_retention_curves(df):
    lid = _col("logo_id") or _col("customer_id")
    dt  = _col("date")
    mrr = _col("mrr") or _col("arr")
    if not (lid and dt and mrr):
        return None, None, "Map 'logo_id'/'customer_id', 'date', and 'mrr'/'arr'."
    d = df[[lid, dt, mrr]].copy()
    _parse_dates(d, [dt])
    d["ym"] = d[dt].dt.to_period("M").astype(str)
    by = d.groupby(["ym", lid])[mrr].sum().reset_index()
    pivot = by.pivot(index=lid, columns="ym", values=mrr).fillna(0.0)
    months = sorted(pivot.columns)
    grr, nrr = [], []
    for i in range(1, len(months)):
        prev, cur = months[i-1], months[i]
        m_prev = pivot[prev]; m_cur = pivot[cur]
        base_prev = m_prev.sum()
        grr_base = np.minimum(m_prev, m_cur).sum() / (base_prev if base_prev!=0 else np.nan)
        nrr_base = (m_cur.sum() / (base_prev if base_prev!=0 else np.nan))
        grr.append({"month": cur, "GRR": grr_base*100})
        nrr.append({"month": cur, "NRR": nrr_base*100})
    return pd.DataFrame(grr), pd.DataFrame(nrr), None

# ============== Sidebar ==============
with st.sidebar:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=150)
    else:
        st.sidebar.markdown("### GROTHKO CONSULTING")

    st.title("Navigation")

    # Navigation list now includes Finance KPIs
    pages = ["Dashboard", "Finance KPIs", "Data Analysis", "AI Assistant", "Visualizations"]
    current_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
    selection = st.radio("Select Page:", pages, index=current_index)
    st.session_state.page = selection

    # API Key
    st.subheader("🗝️ API Configuration")
    api_key = get_openai_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("💪 API key verified")
    else:
        st.info("ℹ️ Store your key in Streamlit Secrets as `OPENAI_API_KEY` (or `openai.api_key`).")

    # Data upload
    st.subheader("🗄️ Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)", type=['csv'],
        help="Upload a CSV file containing your business data"
    )

    # Budget upload (Suggestion 2)
    with st.expander("📥 Budget Upload (for BvA)", expanded=False):
        budget_file = st.file_uploader("Upload Budget CSV", type=["csv"], key="budget_csv")
        if budget_file is not None:
            try:
                st.session_state.budget_df = pd.read_csv(budget_file)
                st.success(f"Loaded budget with {len(st.session_state.budget_df):,} rows.")
            except Exception as e:
                st.error(f"Failed to read budget CSV: {e}")

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

            # Finance Field Mapping (persisted)  (Suggestion 5)
            with st.expander("🔧 Finance Field Mapping (optional but recommended)", expanded=False):
                df = st.session_state.df
                for canon, guesses in FINANCE_CANONICAL_FIELDS.items():
                    # Try persisted mapping first
                    persisted = st.session_state.finance_mapping.get(canon)
                    default_guess = None
                    # find a reasonable default if not persisted
                    if not persisted:
                        for c in df.columns:
                            normc = c.lower().replace(" ","").replace("_","")
                            if normc in [g.replace("_","") for g in guesses]:
                                default_guess = c
                                break
                    options = ["(none)"] + df.columns.tolist()
                    preselect = 0
                    if persisted and persisted in df.columns:
                        preselect = options.index(persisted)
                    elif default_guess and default_guess in df.columns:
                        preselect = options.index(default_guess)
                    choice = st.selectbox(
                        f"Map **{canon}** to:",
                        options,
                        index=preselect,
                        key=f"map_{canon}"
                    )
                    st.session_state.finance_mapping[canon] = None if choice=="(none)" else choice

                if st.button("✅ Confirm Finance Mapping"):
                    st.session_state.mapping_confirmed = True
                    try:
                        MAPPING_STORE.write_text(json.dumps(st.session_state.finance_mapping, indent=2))
                        st.success("Mapping saved for future sessions.")
                    except Exception as e:
                        st.warning(f"Could not persist mapping: {e}")

        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif uploaded_file is not None and not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key first")

# ============== Header ==============
st.markdown('<h1 class="main-header">Grothko Consulting Business Intelligence Generator</h1>', unsafe_allow_html=True)

# ============== Page Routing (fixed) ==============
page = st.session_state.get("page", "Dashboard")

# ---- Dashboard ----
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

# ---- Finance KPIs (includes Suggestions 1-4 UIs) ----
elif page == "Finance KPIs":
    st.header("💼 Finance KPIs")
    if not st.session_state.data_loaded:
        st.warning("Upload a dataset first.")
    elif not st.session_state.mapping_confirmed:
        st.info("Open the sidebar ➜ Finance Field Mapping and click **Confirm** to enable calculations.")
    else:
        df = st.session_state.df

        # Working capital tiles
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dso, e = compute_dso(df); st.metric("DSO (days)", f"{dso:.1f}" if dso is not None else "—")
            if e: st.caption(f"Note: {e}")
        with c2:
            dpo, e = compute_dpo(df); st.metric("DPO (days)", f"{dpo:.1f}" if dpo is not None else "—")
            if e: st.caption(f"Note: {e}")
        with c3:
            dio, e = compute_dio(df); st.metric("DIO (days)", f"{dio:.1f}" if dio is not None else "—")
            if e: st.caption(f"Note: {e}")
        with c4:
            ccc, e = compute_ccc(df); st.metric("Cash Conversion Cycle", f"{ccc:.1f}" if ccc is not None else "—")
            if e: st.caption(f"Note: {e}")

        st.divider()
        # Profitability
        gp, gm_pct = gross_margin(df)
        colA, colB = st.columns(2)
        with colA:
            st.metric("Gross Profit", f"{gp:,.0f}" if gp is not None else "—")
        with colB:
            st.metric("Gross Margin %", f"{gm_pct:.1f}%" if gm_pct is not None else "—")

        # Contribution by customer/product if mapped
        by_choice = st.selectbox("Contribution margin by:", ["(none)","customer_id","sku"])
        if by_choice != "(none)" and _col(by_choice) in st.session_state.df.columns:
            cm = contribution_margin_by(df, _col(by_choice))
            if cm is not None:
                st.subheader(f"Contribution margin by {_col(by_choice)}")
                st.dataframe(cm.head(25))
                try:
                    fig = px.bar(cm.reset_index().head(15), x=_col(by_choice), y="gross_profit", title="Top Contribution")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

        st.divider()
        # Vendor Pareto
        s, s_pct = vendor_pareto(df)
        if s is not None:
            st.subheader("Vendor Concentration (Pareto)")
            st.dataframe(pd.DataFrame({"AP_Amount": s, "Cum%": s_pct}).head(25))
            try:
                # simple rank-based chart
                fig = px.line(pd.DataFrame({"rank": range(1, len(s_pct)+1), "cum_pct": s_pct.values}),
                              x="rank", y="cum_pct", title="Vendor Pareto (cum%)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        st.divider()
        # SaaS quick stats
        saas = churn_and_retention(df)
        if saas:
            st.subheader("SaaS Retention (basic)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Logos", saas["logos"])
            col2.metric("Churned", saas["churned"])
            col3.metric("GRR %", f"{saas['GRR_pct']:.1f}%")
            st.caption("Cohorts and NRR below if mapped fields exist.")

        # ---------- Suggestion (1): 13-week cash ----------
        st.divider()
        st.subheader("🗓️ 13-Week Cash View")
        oc = st.number_input("Opening Cash (starting point)", value=0.0, step=1000.0)
        th = st.number_input("Low-Cash Threshold (warn below)", value=0.0, step=1000.0)
        tbl, err = weekly_13_week_cash(df, opening_cash=oc, min_cash_threshold=th)
        if err:
            st.caption(f"Note: {err}")
        elif tbl is not None:
            st.dataframe(tbl)
            try:
                fig = px.bar(tbl.reset_index(), x="week", y="net", title="Weekly Net Cash (13 weeks)")
                st.plotly_chart(fig, use_container_width=True)
                fig2 = px.line(tbl.reset_index(), x="week", y="ending_cash", title="Ending Cash by Week")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass
            if tbl["flag_low_cash"].any():
                st.warning("Weeks below threshold are flagged in the table.")
            else:
                st.success("No low-cash weeks flagged.")

        # ---------- Suggestion (2): BvA ----------
        st.divider()
        st.subheader("📊 Budget vs. Actuals")
        if st.session_state.budget_df is None:
            st.caption("Upload a Budget CSV in the sidebar to enable BvA.")
        else:
            bva, err = join_budget_actuals(df, st.session_state.budget_df)
            if err:
                st.caption(f"Note: {err}")
            else:
                st.dataframe(bva.head(50))
                try:
                    fig = px.bar(bva, x="period", y="variance", color="account", title="Variance by Period & Account")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
                st.markdown("**Forecast error metrics (lower is better):**")
                st.dataframe(forecast_error_metrics(bva))

        # ---------- Suggestion (3): PVM ----------
        st.divider()
        st.subheader("💸 Price-Volume-Mix Waterfall")
        per_col = _col("period")
        if per_col and per_col in df.columns:
            periods = sorted(df[per_col].astype(str).unique())
            if len(periods) >= 2:
                a = st.selectbox("Baseline period", periods, index=max(0,len(periods)-2))
                b = st.selectbox("Compare to period", periods, index=max(0,len(periods)-1))
                res, err = pvm_decomposition(df, a, b)
                if err:
                    st.caption(f"Note: {err}")
                else:
                    st.dataframe(res)
                    try:
                        fig = px.waterfall(res, x="Component", y="Value", title=f"PVM: {a} ➜ {b}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
            else:
                st.caption("Need at least two distinct 'period' values for PVM.")
        else:
            st.caption("Map or derive 'period' to enable PVM.")

        # ---------- Suggestion (4): Cohorts & NRR ----------
        st.divider()
        st.subheader("🧩 SaaS Cohorts & NRR")
        cohort, nrr_tbl, err = cohort_table_nrr(df)
        if err:
            st.caption(f"Note: {err}")
        else:
            st.markdown("**Cohort MRR (rows=cohort start month, cols=month):**")
            st.dataframe(cohort)
            st.markdown("**Cohort NRR (each cohort normalized to 100% at start):**")
            st.dataframe((nrr_tbl*100).round(1))
            try:
                heat = (nrr_tbl*100).round(1).reset_index().melt(id_vars="cohort", var_name="month", value_name="NRR")
                fig = px.density_heatmap(heat, x="month", y="cohort", z="NRR", title="Cohort NRR Heatmap",
                                         nbinsx=len(heat["month"].unique()))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        grr_curve, nrr_curve, err2 = overall_retention_curves(df)
        if not err2 and grr_curve is not None and nrr_curve is not None and not grr_curve.empty and not nrr_curve.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.line(grr_curve, x="month", y="GRR", title="GRR by Month")
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.line(nrr_curve, x="month", y="NRR", title="NRR by Month")
                st.plotly_chart(fig2, use_container_width=True)

# ---- Data Analysis ----
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

# ---- AI Assistant ----
elif page == "AI Assistant":
    st.header("🧬 AI Assistant")
    api_key = get_openai_api_key()
    if st.session_state.data_loaded and api_key and st.session_state.retriever and st.session_state.llm:
        st.info("💡 Ask questions about your data. I can compute correlations, conditional averages, top insights, and basic trends.")
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]): st.write(m["content"])
        user_question = st.chat_input("Ask a question about your data...")
        if user_question:
            st.session_state.chat_history.append({"role":"user","content":user_question})
            with st.chat_message("user"): st.write(user_question)
            # NOTE: ensure your handle_analytics_query is defined (see placeholder above)
            try:
                handled, msg = handle_analytics_query(user_question, st.session_state.df)
            except NameError:
                handled, msg = (False, "")
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

# ---- Visualizations ----
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

# ============== Footer ==============
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Grothko Consulting B.I.G - Business Intelligence Generator</p>
</div>
""", unsafe_allow_html=True)

