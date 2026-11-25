import json
from pathlib import Path

import joblib
import streamlit as st


pasta_projeto = Path(__file__).resolve().parent
arquivo_modelo = pasta_projeto / "models" / "model.joblib"
arquivo_metricas = pasta_projeto / "reports" / "metricas.json"

st.title("Filtro antispam de SMS")
st.write("Digite uma mensagem em inglês para verificar se ela parece spam.")
st.caption("Protótipo educacional. O modelo pode errar.")

if not arquivo_modelo.is_file() or not arquivo_metricas.is_file():
    st.error("Modelo ou métricas ausentes. Execute python train.py e recarregue a página.")
    st.stop()

modelo = joblib.load(arquivo_modelo)
relatorio = json.loads(arquivo_metricas.read_text(encoding="utf-8"))
algoritmo = relatorio["modelo_escolhido"]
f1 = relatorio["resultados"][algoritmo]["f1"]

st.write(f"Modelo: **{algoritmo}**")
st.metric("F1 de spam no conjunto de teste", f"{f1:.1%}".replace(".", ","))

mensagem = st.text_area("Mensagem SMS")

if st.button("Fazer previsão"):
    texto = mensagem.strip()

    if not texto:
        st.warning("Digite uma mensagem antes de fazer a previsão.")
    else:
        previsao = modelo.predict([texto])[0]

        if previsao == "spam":
            st.warning("Provável spam. O texto tem padrões associados a spam no treino.")
        else:
            st.success("Provável mensagem legítima. O modelo classificou o texto como legítimo.")