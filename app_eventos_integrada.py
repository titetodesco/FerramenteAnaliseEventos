import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from docx import Document
from io import BytesIO
import re, os, tempfile
from unidecode import unidecode
from langdetect import detect
from rapidfuzz import fuzz
import plotly.express as px

# ===== Embeddings =====
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ====== CONFIG INICIAL ======
st.set_page_config(page_title="Análise Integrada de Eventos (ESO)", layout="wide")
st.title("🛡️ Análise Integrada de Eventos (ESO) – WS × Precursores (HTO) × Fatores (TaxonomiaCP)")

# ---- URLs dos dados no GitHub (RAW) ----
URL_WS = "https://raw.githubusercontent.com/titetodesco/CorrelacaoWS-PREC/main/DicionarioWaekSignals.xlsx"
URL_PRECS = "https://raw.githubusercontent.com/titetodesco/CorrelacaoWS-PREC/main/precursores_expandido.xlsx"
URL_TAXO = "https://raw.githubusercontent.com/titetodesco/CondicionantesPerformance/main/TaxonomiaCP_Por.xlsx"
URL_TRIPLO = "https://raw.githubusercontent.com/titetodesco/CorrelacaoWS-PREC/main/MapaTriplo_tratado.xlsx"

# ---- Thresholds padrão ----
DEFAULT_WS_TH = 0.50
DEFAULT_PREC_TH = 0.50
DEFAULT_TAXO_TH = 0.55  # ligeiramente mais alto por termos mais genéricos

# ===== Helpers =====
def load_xlsx(url_or_file, sheet_name=None):
    return pd.read_excel(url_or_file, sheet_name=sheet_name)

def simple_sent_split(text:str):
    # Sem dependências do NLTK: quebra por pontuação . ! ? e novas linhas
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    parts = [p.strip() for p in parts if p and len(p.strip())>2]
    return parts

def extract_text(uploaded):
    suffix = os.path.splitext(uploaded.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name
    if suffix == ".pdf":
        doc = fitz.open(path)
        text = "\n".join([page.get_text("text") for page in doc])
    elif suffix == ".docx":
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif suffix == ".txt":
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    else:
        text = ""
    # normalização leve para matching
    return text

@st.cache_resource(show_spinner=False)
def get_model():
    # Modelo leve que roda bem no Streamlit Cloud
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(model, texts:list[str]):
    if not texts:
        return np.empty((0,384), dtype=np.float32)
    return model.encode(texts, normalize_embeddings=True)

def normalize_ws_name(s):
    # remove "(0.xx)" no final
    return re.sub(r"\s*\([-+]?\d*\.?\d+\)\s*$", "", str(s)).strip()

def prepare_ws_list(df_ws):
    # Espera uma coluna com a lista de sinais fracos. Adapte se o seu arquivo tiver outro layout.
    # Vamos aceitar primeira coluna não vazia como sinal fraco.
    col = df_ws.columns[0]
    ws = df_ws[col].dropna().astype(str).str.strip()
    # remover duplicados (normalizados)
    ws_norm = ws.apply(normalize_ws_name)
    return ws_norm.drop_duplicates().tolist()

def prepare_precs_list(df_prec):
    # Espera colunas: ["Categoria","Precursor_PT","Precursor_EN"] OU coluna única "Precursor" com PT/EN unidos
    cols = [c.lower() for c in df_prec.columns]
    df = df_prec.copy()
    df.columns = cols

    if "precursor" in df.columns:
        # formato já concatenado "falha hidráulica/pipe rupture"
        out = []
        for _, r in df.iterrows():
            cat = r.get("categoria","")
            prec = str(r["precursor"]).strip()
            out.append((prec, cat))
        # dedup
        uniq = {}
        for p,c in out:
            uniq[(p.lower(), c)] = (p,c)
        return list(uniq.values())
    else:
        # usar precursor_pt e precursor_en
        if "precursor_pt" in df.columns and "precursor_en" in df.columns:
            df["precursor"] = df["precursor_pt"].fillna("").astype(str).str.strip() + "/" + df["precursor_en"].fillna("").astype(str).str.strip()
            out = []
            for _, r in df.iterrows():
                cat = r.get("categoria","")
                prec = r["precursor"]
                if prec.strip("/ ").strip():
                    out.append((prec, cat))
            # dedup
            uniq = {}
            for p,c in out:
                uniq[(p.lower(), c)] = (p,c)
            return list(uniq.values())
        else:
            # fallback: junta tudo que houver em texto
            col = df.columns[1]
            out = df[col].dropna().astype(str).str.strip().tolist()
            uniq = {}
            for p in out:
                uniq[p.lower()] = p
            return [(v,"") for v in uniq.values()]

def prepare_taxo_terms(df_taxo, idioma="pt"):
    # Colunas esperadas na taxonomia: "Bag de termos"(pt) e "Bag of terms"(en)
    bag_col = "Bag de termos" if idioma == "pt" else "Bag of terms"
    if bag_col not in df_taxo.columns:
        # tenta achar uma das duas
        if "Bag de termos" in df_taxo.columns:
            bag_col = "Bag de termos"
        elif "Bag of terms" in df_taxo.columns:
            bag_col = "Bag of terms"
        else:
            return []
    bag_series = df_taxo[bag_col].fillna("").astype(str)
    terms = []
    for x in bag_series:
        parts = [t.strip() for t in x.split(";") if t.strip()]
        terms.extend(parts)
    # dedup
    uniq = {}
    for t in terms:
        uniq[t.lower()] = t
    return list(uniq.values())

def jaccard(a:set, b:set):
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

# ===== Sidebar: dados & parâmetros =====
st.sidebar.header("⚙️ Parâmetros")
ws_th = st.sidebar.slider("Limiar de Similaridade para Weak Signals", 0.30, 0.85, DEFAULT_WS_TH, 0.01)
prec_th = st.sidebar.slider("Limiar de Similaridade para Precursores", 0.30, 0.85, DEFAULT_PREC_TH, 0.01)
taxo_th = st.sidebar.slider("Limiar de Similaridade para Fatores (TaxonomiaCP)", 0.30, 0.85, DEFAULT_TAXO_TH, 0.01)

st.sidebar.header("📚 Fontes no GitHub (opcional)")
url_ws = st.sidebar.text_input("URL Dicionário WS (.xlsx)", URL_WS)
url_prec = st.sidebar.text_input("URL Precursores (.xlsx)", URL_PRECS)
url_taxo = st.sidebar.text_input("URL TaxonomiaCP (.xlsx)", URL_TAXO)
url_triple = st.sidebar.text_input("URL MapaTriplo (.xlsx)", URL_TRIPLO)

# ===== Carregar dicionários & taxonomia =====
with st.spinner("Carregando dicionários e taxonomia..."):
    df_ws_raw = load_xlsx(url_ws)
    df_prec_raw = load_xlsx(url_prec)
    df_taxo_raw = load_xlsx(url_taxo)
    try:
        df_triplo = load_xlsx(url_triple)
    except Exception:
        df_triplo = None

ws_list = prepare_ws_list(df_ws_raw)                               # lista de strings
precs_list = prepare_precs_list(df_prec_raw)                       # lista de (prec, HTO)
# para taxonomia, usaremos PT+EN combinados (mais chance de match)
taxo_terms_pt = prepare_taxo_terms(df_taxo_raw, "pt")
taxo_terms_en = prepare_taxo_terms(df_taxo_raw, "en")
taxo_terms = list(dict.fromkeys(taxo_terms_pt + taxo_terms_en))

# Embeddings dos vocabulários
model = get_model()
ws_emb = embed_texts(model, ws_list)
precs_emb = embed_texts(model, [p for (p, _) in precs_list])
taxo_emb = embed_texts(model, taxo_terms)

# ===== Upload de documentos =====
st.header("1) 📂 Envie o(s) documento(s) do evento")
files = st.file_uploader("PDF, DOCX ou TXT (1..N)", type=["pdf","docx","txt"], accept_multiple_files=True)

if not files:
    st.info("Envie um ou mais documentos para iniciar a análise.")
    st.stop()

# ===== Processar cada documento =====
st.header("2) 🔎 Extração e Identificação")
all_rows = []
for file in files:
    text_raw = extract_text(file)
    # detect idioma com fallback
    try:
        lang = detect(text_raw)  # 'pt' ou 'en' (ou 'es' etc.)
    except Exception:
        lang = "pt" if any(w in text_raw.lower() for w in ["segurança","falha","trabalho"]) else "en"

    # normalizar para matching
    text_norm = unidecode(text_raw.lower())

    # dividir em "parágrafos/sentenças" usáveis
    paras = simple_sent_split(text_raw)

    # embeddings dos parágrafos
    para_emb = embed_texts(model, paras)

    # similaridade com WS / Precs / Taxo
    if len(para_emb) > 0:
        sim_ws = cosine_similarity(para_emb, ws_emb) if len(ws_list) else np.zeros((len(paras),0))
        sim_prec = cosine_similarity(para_emb, precs_emb) if len(precs_list) else np.zeros((len(paras),0))
        sim_taxo = cosine_similarity(para_emb, taxo_emb) if len(taxo_terms) else np.zeros((len(paras),0))
    else:
        sim_ws = sim_prec = sim_taxo = np.zeros((0,0))

    # coletar hits acima do threshold
    for i, paragraph in enumerate(paras):
        ws_hits = []
        if sim_ws.size:
            idx_ws = np.where(sim_ws[i] >= ws_th)[0]
            for j in idx_ws:
                ws_hits.append((ws_list[j], float(sim_ws[i, j])))

        prec_hits = []
        if sim_prec.size:
            idx_p = np.where(sim_prec[i] >= prec_th)[0]
            for j in idx_p:
                prec_hits.append((precs_list[j][0], precs_list[j][1], float(sim_prec[i, j])))  # (prec, HTO, score)

        taxo_hits = []
        if sim_taxo.size:
            idx_t = np.where(sim_taxo[i] >= taxo_th)[0]
            for j in idx_t:
                taxo_hits.append((taxo_terms[j], float(sim_taxo[i, j])))

        if ws_hits or prec_hits or taxo_hits:
            all_rows.append({
                "Arquivo": file.name,
                "Idioma": lang,
                "Paragrafo": paragraph.strip(),
                "WS_list": ws_hits,                 # [(ws, score)]
                "Precs_list": prec_hits,            # [(prec, HTO, score)]
                "Taxo_list": taxo_hits              # [(termo, score)]
            })

if not all_rows:
    st.warning("Nenhuma correspondência encontrada com os limiares atuais. Tente reduzir os thresholds na barra lateral.")
    st.stop()

df_hits = pd.DataFrame(all_rows)

# ==== Flatten para tabelas legíveis ====
def expand_ws(df):
    rows = []
    for _, r in df.iterrows():
        for ws, sc in r["WS_list"]:
            rows.append([r["Arquivo"], r["Idioma"], r["Paragrafo"], ws, sc])
    return pd.DataFrame(rows, columns=["Arquivo","Idioma","Paragrafo","WeakSignal","WS_Sim"])

def expand_prec(df):
    rows = []
    for _, r in df.iterrows():
        for prec, hto, sc in r["Precs_list"]:
            rows.append([r["Arquivo"], r["Idioma"], r["Paragrafo"], hto, prec, sc])
    return pd.DataFrame(rows, columns=["Arquivo","Idioma","Paragrafo","HTO","Precursor","Prec_Sim"])

def expand_taxo(df):
    rows = []
    for _, r in df.iterrows():
        for termo, sc in r["Taxo_list"]:
            rows.append([r["Arquivo"], r["Idioma"], r["Paragrafo"], termo, sc])
    return pd.DataFrame(rows, columns=["Arquivo","Idioma","Paragrafo","Fator_Termo","Taxo_Sim"])

df_ws = expand_ws(df_hits)
df_prec = expand_prec(df_hits)
df_taxo = expand_taxo(df_hits)

st.success(f"✅ Encontrados: {len(df_ws)} WS • {len(df_prec)} Precursores • {len(df_taxo)} Fatores/Termos (Taxonomia)")

# ===== 3) Visualizações =====
tabs = st.tabs(["📄 Tabelas", "🌳 Árvore HTO→Precursor→WS", "🗺️ Treemap", "🧬 Relatórios Pregressos Similares", "⬇️ Exportar"])

with tabs[0]:
    st.subheader("Weak Signals")
    st.dataframe(df_ws, use_container_width=True, height=300)
    st.subheader("Precursores (HTO)")
    st.dataframe(df_prec, use_container_width=True, height=300)
    st.subheader("Fatores (TaxonomiaCP)")
    st.dataframe(df_taxo, use_container_width=True, height=300)

with tabs[1]:
    st.subheader("Árvore (colapsável) HTO → Precursor → Weak Signals")
    # construir “tabela de frequência” WS por precursor
    ws_clean = df_ws.assign(WeakSignal=df_ws["WeakSignal"].apply(normalize_ws_name))
    join = df_prec.merge(ws_clean[["Arquivo","Paragrafo","WeakSignal"]], on=["Arquivo","Paragrafo"], how="inner")
    if join.empty:
        st.info("Sem interseção entre parágrafos com WS e Precursores nos thresholds atuais.")
    else:
        freq = (join.groupby(["HTO","Precursor","WeakSignal"], as_index=False)
                     .size()
                     .rename(columns={"size":"Frequencia"}))
        st.caption("Clique nos nós para expandir/colapsar (visual simples por tabela):")
        st.dataframe(freq, use_container_width=True)

with tabs[2]:
    st.subheader("Treemap Hierárquico")
    if df_prec.empty:
        st.info("Sem dados de Precursores para o Treemap.")
    else:
        # preparar WS por parágrafo que também tem precursor (como acima)
        ws_clean = df_ws.assign(WeakSignal=df_ws["WeakSignal"].apply(normalize_ws_name))
        join = df_prec.merge(ws_clean[["Arquivo","Paragrafo","WeakSignal"]], on=["Arquivo","Paragrafo"], how="inner")
        if join.empty:
            st.info("Sem interseção entre parágrafos com WS e Precursores para o treemap.")
        else:
            fig = px.treemap(
                join,
                path=["HTO","Precursor","WeakSignal"],
                values=[1]*len(join),
                hover_data=["Arquivo","Paragrafo"],
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Relatórios de investigação pregressos mais similares")
    if df_triplo is None or df_triplo.empty:
        st.info("MapaTriplo não carregado. Informe a URL correta na barra lateral.")
    else:
        # Normalizar colunas esperadas (Top_WS, Top_Precursores, Report, Text)
        cols = {c.lower(): c for c in df_triplo.columns}
        # tentar achar nomes padrões
        def find_col(name_candidates):
            for c in df_triplo.columns:
                if c.lower() in [n.lower() for n in name_candidates]:
                    return c
            return None
        col_ws = find_col(["Top_WS","weak signals","ws"])
        col_prec = find_col(["Top_Precursores","precursores","precursors"])
        col_rep = find_col(["Report","Relatorio","Arquivo"])
        if not (col_ws and col_prec and col_rep):
            st.info("A planilha do MapaTriplo precisa ter colunas equivalentes a Top_WS, Top_Precursores e Report.")
        else:
            # WS/Prec encontrados agora:
            found_ws = set(ws_clean["WeakSignal"].unique().tolist())
            found_prec = set(df_prec["Precursor"].unique().tolist())

            # extrair ws/prec na triplo (normalizando similares "(0.xx)")
            def parse_ws_cell(cell):
                if not isinstance(cell,str): return []
                out=[]
                for part in cell.split(";"):
                    nm = normalize_ws_name(part.strip())
                    if nm: out.append(nm)
                return out
            def parse_prec_cell(cell):
                if not isinstance(cell,str): return []
                out=[]
                for part in cell.split(";"):
                    m = re.match(r"(.+?)\s*\[([^\]]+)\]\s*(?:\([-+]?\d*\.?\d+\))?$", part.strip())
                    if m:
                        out.append(m.group(1).strip())
                    else:
                        out.append(part.strip())
                return out

            sims = []
            for _, r in df_triplo.iterrows():
                ws_set = set(parse_ws_cell(r[col_ws]))
                prec_set = set(parse_prec_cell(r[col_prec]))
                score_ws = jaccard(found_ws, ws_set)
                score_prec = jaccard(found_prec, prec_set)
                final = 0.6*score_ws + 0.4*score_prec
                sims.append((r[col_rep], score_ws, score_prec, final))
            df_sim = pd.DataFrame(sims, columns=["Report","Jaccard_WS","Jaccard_Prec","Score_Final"]).sort_values("Score_Final", ascending=False)
            st.dataframe(df_sim.head(15), use_container_width=True)

with tabs[4]:
    st.subheader("Downloads")
    def to_xlsx_bytes(df_in: pd.DataFrame, sheet="dados") -> bytes:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as wr:
            df_in.to_excel(wr, sheet_name=sheet, index=False)
        bio.seek(0)
        return bio.read()

    colA, colB, colC = st.columns(3)
    with colA:
        st.download_button("⬇️ WS encontrados (xlsx)", data=to_xlsx_bytes(df_ws, "ws"),
                           file_name="ws_encontrados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with colB:
        st.download_button("⬇️ Precursores encontrados (xlsx)", data=to_xlsx_bytes(df_prec, "precursores"),
                           file_name="precursores_encontrados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with colC:
        st.download_button("⬇️ Fatores / Taxonomia (xlsx)", data=to_xlsx_bytes(df_taxo, "taxo"),
                           file_name="taxonomia_encontrada.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Dica: ajuste os limiares na barra lateral para aumentar/diminuir a sensibilidade dos matches.")
