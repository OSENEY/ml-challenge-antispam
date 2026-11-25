import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Usar caminhos relativos ao próprio script permite executar o comando de qualquer pasta.
pasta_projeto = Path(__file__).resolve().parent

# 1. Ler o arquivo. Cada linha tem um rótulo, uma tabulação e a mensagem.
arquivo = pasta_projeto / "data" / "SMSSpamCollection"
if not arquivo.is_file():
    raise SystemExit(
        "Base de dados não encontrada em data/SMSSpamCollection. "
        "Consulte a seção 'Obter os dados' do README.md."
    )

registros = []

with arquivo.open(encoding="utf-8") as base:
    for linha in base:
        rotulo, separador, mensagem = linha.partition("\t")
        if separador:
            registros.append((rotulo, mensagem.strip()))

dados = pd.DataFrame(registros, columns=["rotulo", "mensagem"])
dados["tamanho"] = dados["mensagem"].str.len()

# 2. Calcular os valores que vamos usar para escrever os achados.
contagem = dados["rotulo"].value_counts()
percentual_spam = 100 * contagem.get("spam", 0) / len(dados)
medianas = dados.groupby("rotulo")["tamanho"].median()
duplicadas = dados.duplicated(subset=["rotulo", "mensagem"]).sum()

print(f"Total de mensagens: {len(dados)}")
print(contagem.to_string())
print(f"\nPercentual de spam: {percentual_spam:.1f}%")
print("\nTamanho mediano por classe (em caracteres):")
print(medianas.to_string())
print(f"\nCópias repetidas com o mesmo rótulo: {duplicadas}")
print(f"Mensagens vazias: {dados['mensagem'].eq('').sum()}")

# 3. Salvar os dois gráficos.
pasta_graficos = pasta_projeto / "reports" / "graficos"
pasta_graficos.mkdir(parents=True, exist_ok=True)

figura, eixo = plt.subplots()
contagem.rename({"ham": "Legítima", "spam": "Spam"}).plot.bar(
    ax=eixo, rot=0
)
eixo.set(title="Quantidade de mensagens por classe", xlabel="", ylabel="Mensagens")
figura.tight_layout()
figura.savefig(pasta_graficos / "quantidade_por_classe.png", dpi=150)
plt.close(figura)

figura, eixo = plt.subplots()
eixo.boxplot([
    dados.loc[dados["rotulo"] == "ham", "tamanho"],
    dados.loc[dados["rotulo"] == "spam", "tamanho"],
])
eixo.set_xticks([1, 2], ["Legítima", "Spam"])
eixo.set(title="Comprimento das mensagens por classe", ylabel="Caracteres")
figura.tight_layout()
figura.savefig(pasta_graficos / "comprimento_por_classe.png", dpi=150)
plt.close(figura)

print(f"\nGráficos salvos em {pasta_graficos}")

# Remover textos vazios, rótulos contraditórios e cópias repetidas.
base_treino = dados.loc[dados["mensagem"].ne(""), ["mensagem", "rotulo"]]
rotulos_por_texto = base_treino.groupby("mensagem")["rotulo"].transform("nunique")
base_treino = base_treino.loc[rotulos_por_texto.eq(1)]
base_treino = base_treino.drop_duplicates(subset="mensagem")

# Separar 80% para treino e 20% para teste.
X = base_treino["mensagem"]
y = base_treino["rotulo"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTreino: {len(X_treino)} mensagens | Teste: {len(X_teste)} mensagens")

# Cada modelo aprende a transformar o texto em números com TF-IDF.
modelos = {
    "Naive Bayes": make_pipeline(TfidfVectorizer(), MultinomialNB()),
    "Regressão Logística": make_pipeline(
        TfidfVectorizer(), LogisticRegression(max_iter=1000)
    ),
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)

    avaliacao = classification_report(
        y_teste, previsoes, labels=["ham", "spam"], output_dict=True, zero_division=0
    )
    por_classe = {
        classe: {
            "precisao": avaliacao[classe]["precision"],
            "recall": avaliacao[classe]["recall"],
            "f1": avaliacao[classe]["f1-score"],
            "quantidade": int(avaliacao[classe]["support"]),
        }
        for classe in ("ham", "spam")
    }
    precisao = por_classe["spam"]["precisao"]
    recall = por_classe["spam"]["recall"]
    f1 = por_classe["spam"]["f1"]
    matriz = confusion_matrix(y_teste, previsoes, labels=["ham", "spam"])

    resultados[nome] = {
        "precisao": float(precisao),
        "recall": float(recall),
        "f1": float(f1),
        "matriz_confusao": matriz.tolist(),
        "por_classe": por_classe,
    }

    print(f"\n{nome}")
    print(f"Precisão: {precisao:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}")
    print(f"Mensagens legítimas marcadas como spam: {matriz[0, 1]}")
    print(f"Mensagens spam que passaram pelo filtro: {matriz[1, 0]}")

# Escolher pelo maior F1 da classe spam e salvar o modelo completo.
escolhido = max(resultados, key=lambda nome: resultados[nome]["f1"])

pasta_modelos = pasta_projeto / "models"
pasta_relatorios = pasta_projeto / "reports"
pasta_modelos.mkdir(exist_ok=True)
pasta_relatorios.mkdir(exist_ok=True)
joblib.dump(modelos[escolhido], pasta_modelos / "model.joblib")

relatorio = {
    "modelo_escolhido": escolhido,
    "criterio": "Maior F1 para a classe spam",
    "ordem_classes_na_matriz": ["ham", "spam"],
    "resultados": resultados,
}
(pasta_relatorios / "metricas.json").write_text(
    json.dumps(relatorio, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\nModelo escolhido: {escolhido}")
print("Modelo salvo em models/model.joblib")
print("Métricas salvas em reports/metricas.json")
