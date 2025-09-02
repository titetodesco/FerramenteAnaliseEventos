import io
import json
import requests
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import plotly.express as px
st.cache_data.clear()
st.set_page_config(page_title="Árvore HTO → Precursores → Weak Signals", layout="wide")
st.title("🌳 Árvore: HTO → Precursores → Weak Signals")

# ===== 1) Fonte dos dados (XLSX no GitHub) =====
URL_XLSX = "https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main/MapaTriplo_tratado.xlsx"

@st.cache_data(ttl=300, show_spinner=True)
def load_excel(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    # Se houver várias abas, use a primeira (ou troque por sheet_name="nome_da_sua_aba")
    return pd.read_excel(bio)

try:
    df = load_excel(URL_XLSX)
except Exception as e:
    st.error(f"Falha ao baixar/ler o Excel: {e}")
    st.stop()

# Esperado: colunas com esses nomes (já tratados anteriormente)
# ['HTO','Precursor','WeakSignal','Report','Text']
required = {"HTO","Precursor","WeakSignal","Report","Text"}
missing = required - set(df.columns)
if missing:
    st.error(f"Planilha não contém as colunas obrigatórias: {missing}")
    st.dataframe(df.head())
    st.stop()

# ===== 2) Filtros =====
cA, cB, cC = st.columns([2,2,1])
with cA:
    reports_sel = st.multiselect("Filtrar por Report", sorted(df["Report"].dropna().unique().tolist()))
with cB:
    min_freq = st.number_input("Frequência mínima (WS por precursor)", min_value=1, max_value=100, value=1, step=1)
with cC:
    init_depth = st.slider("Profundidade inicial", 1, 3, 2)

df_f = df.copy()
if reports_sel:
    df_f = df_f[df_f["Report"].isin(reports_sel)]

# ===== 3) Agregação para árvore e índice para drill-down =====
# índice: mapeia uma chave de nó -> linhas do df_f
node_index = {}

def add_index(key, rows_idx):
    node_index.setdefault(key, set()).update(rows_idx)

# criar dicionário hierárquico {HTO: {Precursor: {WS: [linhas]}}}
tree_dict = {}
for i, r in df_f.iterrows():
    h = str(r["HTO"])
    p = str(r["Precursor"])
    w = str(r["WeakSignal"])

    tree_dict.setdefault(h, {}).setdefault(p, {}).setdefault(w, []).append(i)

# filtrar por frequência mínima no nível WS
for h in list(tree_dict.keys()):
    for p in list(tree_dict[h].keys()):
        for w in list(tree_dict[h][p].keys()):
            if len(tree_dict[h][p][w]) < int(min_freq):
                del tree_dict[h][p][w]
        if not tree_dict[h][p]:
            del tree_dict[h][p]
    if not tree_dict[h]:
        del tree_dict[h]

if not tree_dict:
    st.info("Sem dados após os filtros atuais.")
    st.stop()

# ===== 4) Converter para o formato de árvore do ECharts =====
def make_node(name, children=None, value=None, extra=None):
    node = {"name": name}
    if value is not None:
        node["value"] = value
    if extra is not None:
        node["extra"] = extra  # guardamos metadados que usamos no click
    if children:
        node["children"] = children
    return node

def build_echarts_tree(tree_dict):
    echarts_root_children = []
    for hto, precs in sorted(tree_dict.items()):
        prec_children = []
        for prec, ws_dict in sorted(precs.items()):
            ws_children = []
            for ws, idx_list in sorted(ws_dict.items()):
                # nó de WS: valor = frequência, extra = chave-índice para drilldown
                key = f"WS::{hto}::{prec}::{ws}"
                add_index(key, idx_list)
                ws_children.append(make_node(
                    f"{ws}",
                    value=len(idx_list),
                    extra={"type": "ws", "key": key}
                ))
            key_prec = f"PREC::{hto}::{prec}"
            # índice do precursor (todas linhas de seus ws)
            all_idx = [i for ws_idxs in ws_dict.values() for i in ws_idxs]
            add_index(key_prec, all_idx)
            prec_children.append(make_node(
                f"{prec}",
                children=ws_children,
                value=len(all_idx),
                extra={"type": "prec", "key": key_prec}
            ))
        key_hto = f"HTO::{hto}"
        all_idx_hto = []
        for prec, ws_dict in precs.items():
            for idxs in ws_dict.values():
                all_idx_hto.extend(idxs)
        add_index(key_hto, all_idx_hto)
        echarts_root_children.append(make_node(
            f"{hto}",
            children=prec_children,
            value=len(all_idx_hto),
            extra={"type": "hto", "key": key_hto}
        ))
    # raiz
    root = make_node("ROOT", children=echarts_root_children, value=sum(len(v) for v in node_index.get("ROOT", [])) if "ROOT" in node_index else None)
    return root

root_data = build_echarts_tree(tree_dict)

# ===== 5) Desenhar ÁRVORE ECharts =====
st.subheader("🌿 Árvore interativa (colapsável)")

options = {
    "tooltip": {
        "trigger": "item",
        "triggerOn": "mousemove",
        "formatter": """function(p) {
            var v = (p.value !== undefined) ? ("<br/>Freq: " + p.value) : "";
            return "<b>" + p.name + "</b>" + v;
        }"""
    },
    "series": [{
        "type": "tree",
        "data": [root_data],
        "left": "2%",
        "right": "20%",
        "top": "2%",
        "bottom": "2%",
        "symbol": "circle",
        "symbolSize": 10,
        "expandAndCollapse": True,
        "initialTreeDepth": int(init_depth),
        "animationDuration": 300,
        "animationDurationUpdate": 300,
        "label": {
            "position": "left",
            "verticalAlign": "middle",
            "align": "right",
            "fontSize": 12
        },
        "leaves": {
            "label": {"position": "right", "align": "left"}
        },
        "emphasis": {"focus": "descendant"},
        "roam": True  # permite pan/zoom
    }]
}

events = {
    "click": "function (params) { return params; }"
}
event = st_echarts(options=options, height="650px", events=events)

# ===== 6) Drill-down ao clicar no nó =====
st.subheader("🔎 Detalhes do nó selecionado")
if event and "name" in event:
    data = event.get("data", {}) or {}
    extra = data.get("extra", {})
    key = extra.get("key")
    if key and key in node_index:
        idxs = sorted(node_index[key])
        detail = df_f.loc[idxs, ["HTO","Precursor","WeakSignal","Report","Text"]].copy()
        st.write(f"**Nó:** `{event['name']}` — **linhas:** {len(detail)}")
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "📥 Baixar CSV deste nó",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="detalhes_no.csv",
            mime="text/csv"
        )
    else:
        st.info("Clique em um nó de **HTO**, **Precursor** ou **WeakSignal** para ver os detalhes.")

# ===== 7) Treemap (alternativa visual) =====
st.subheader("🧩 Treemap (alternativa)")
# cada linha vale 1 ocorrência
df_f["_one_"] = 1
fig = px.treemap(
    df_f,
    path=["HTO","Precursor","WeakSignal"],
    values="_one_",
    hover_data=["Report","Text"],
)
st.plotly_chart(fig, use_container_width=True)
