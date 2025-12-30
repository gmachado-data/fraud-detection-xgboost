# %%

# FRAUD DETECTION PIPELINE — BUSINESS & RISK ORIENTED APPROACH
# Context:
# Este projeto simula um cenário real de uma fintech / instituição de pagamentos,
# cujo objetivo é identificar transações com alto risco de fraude.

# Do ponto de vista de negócio, fraudes geram:
# - perdas financeiras diretas
# - risco operacional
# - impacto em liquidez e funding
# - deterioração da experiência do cliente

# O pipeline está estruturado nas seguintes etapas:
# 1) Entendimento do problema de risco
# 2) Análise exploratória orientada a comportamento financeiro
# 3) Preparação dos dados para modelagem
# 4) Benchmark de modelos supervisionados
# 5) Ajuste de threshold com base em custo de erro
# 6) Conclusões para tomada de decisão


# FASE 1 — DEFINIÇÃO DO PROBLEMA DE RISCO

# O objetivo do projeto é identificar transações com comportamento anômalo
# que indiquem potencial fraude.
# Em um contexto real, esse tipo de modelo apoia decisões como:
# - bloqueio preventivo de transações
# - acionamento de regras antifraude
# - priorização de análises manuais

# A modelagem considera padrões relacionados a:
# Comportamento do cliente
# Comportamento do merchant
# Valores transacionados fora do padrão
# Concentração temporal de transações

# Tudo isso é importante para evitar prejuízos, detectar anomalias antes que se tornem fraudes reais, antecipar riscos em merchants e categorias.

#FASE 2 - CARREGAMENTO + ANÁLISE EXPLORATÓRIA DOS DADOS 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho relativo esperado para o dataset
DATA_PATH = "data/transactions.csv"

df = pd.read_csv(DATA_PATH)
print(df.head())


# %%
df.info()
#Podemos notar que algumas colunas categóricas vêm como strings, inclusive colunas que parecem numéricas como age e zip codes. 
#É interessante padronizar e codificar essas variáveis antes de modelar, porque o modelo não vai conseguir interpretá-las corretamente se continuarem como texto.

# %%
df.describe()
#Aqui, avaliamos alguns valores para buscar comportamentos, possíveis erros e outliers
# A distribuição de step indica um horizonte temporal completo que pode ser explorado para detectar sazonalidade de risco e picos de transação.
#Pela análise na coluna amount, identificamos uma assimetria muito forte, indicando a presença de outliers


# %%
df["fraud"].value_counts(normalize=True)
# A variável target (fraud) é altamente desbalanceada.
# Apenas ~1.2% das transações são fraudes.

# Do ponto de vista de negócio, isso significa que:
# - Accuracy não é uma métrica confiável
# - O custo de um falso negativo (fraude não detectada) é alto

# Por isso, métricas como Recall e AUC são priorizadas, pois refletem melhor a capacidade do modelo em capturar risco.

# %%
#Distribuição dos valores transacionados

sns.histplot(df["amount"], bins=50)
plt.title("Distribuição de Valores")
plt.show()

#Plotando um gráfico histograma para observar a distribuição dos amounts, mostrando graficamente que existem outliers

# %%
#Calculando o volume de transações no tempo
#Esse gráfico mostra picos e vales e serve para antecipar picos de demanda, garantir liquidez e entender sazonalidade

df.groupby("step")["amount"].sum().plot(figsize=(12,4))
plt.title("Fluxo de Transações ao Longo do Tempo")
plt.show()

#A análise do gráfico mostra um aumento no número de transações com o passar do tempo, com oscilações mantendo um padrão
#Também observamos muitos picos e valores rapidos, sem quedas ou altas muito bruscas.
#Essa oscilação pode representar comportamento heterogeneo dos merchants e/ou muitos clientes fazendo transações simultaneas


# %%
#Cálculo do volume por categoria

df["category"].value_counts().plot(kind="bar")
plt.title("Transações por Categoria")
plt.show()


# %%
#FASE 3: Etapa de pré processamento

# Nesta etapa, preparamos os dados para garantir que o modelo consiga capturar padrões de risco de forma consistente.
# Variáveis categóricas são transformadas em numéricas, permitindo que o modelo aprenda relações entre categorias, como:
# Tipo de transação, canal e comportamento do merchant.

# Uma cópia do dataset original foi mantida para garantir rastreabilidade e evitar perda de informação.

import sklearn 
from sklearn.preprocessing import LabelEncoder

df_prep = df.copy()
cat_cols = df_prep.select_dtypes(include="object").columns
cat_cols

for col in cat_cols:
    df_prep[col] = LabelEncoder().fit_transform(df_prep[col])


#Garantindo que a coluna target fraud é numérica:

df_prep["fraud"] = df_prep["fraud"].astype(int)

#Remover colunas que não queremos usar no modelo
cols_to_remove = ["gender", "customer", "merchant","zipcodeOri", "zipMerchant", "age"]
df_prep = df_prep.drop(columns=cols_to_remove)

# Algumas colunas são removidas por não agregarem valor preditivo ou por introduzirem risco de overfitting.

# IDs como customer e merchant não representam comportamento, apenas identificação.
# Variáveis como gênero, idade e zipcode foram removidas por não serem determinantes diretas de fraude neste contexto e para evitar vieses indesejados.


# %%
#Separação das features e da variável alvo (target)
#Primeiro, removemos a coluna "fraud" para que ela não entre como feature no modelo e armazenamos em X
#Agora, X tem todas as colunas numéricas e categorizadas, exceto fraud
#Em resumo:
#X → valor, horário, merchant, categoria, idade, etc.
#y → fraude ou não fraude

X = df_prep.drop("fraud", axis=1)
y = df_prep["fraud"]

# %%
#Verificando se o número de linhas é igual, se X tem todas as features esperadas e se y tem apenas 1 coluna:
X.shape, y.shape

# %%
#DATA SPLIT - TRAIN/TEST

#Divisão dos dados em treino e teste
#Estamos dividindo as features (X) e target (y) em conjuntos de treino e teste
#80% dos dados vão treinar o modelo e 20% vão avaliar a performance real
#Usamos stratify=y para garantir que a proporção de fraudes e não fraudes seja preservada nos dois conjuntos
#Isso é fundamental para evitar viés, já que datasets de fraude tendem a ser altamente desbalanceados
#O random_state torna a divisão reprodutível, o que permite comparar resultados de forma consistente independente de quando ou onde rodar o código

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %%

#FASE 4: BENCHMARK DE MODELOS SUPERVISIONADOS
#Queremos saber qual é o modelo ideal a ser usado para supervisionamento.
#Primeiro, importamos os modelos e métricas

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import pandas as pd

# %%
#Depois, criamos um dicionario com todos os modelos para benchmark

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced"
    )
}

# %%
#Definimos uma função para treinar e avaliar cada modelo

def evaluate_model(model, X_train, y_train, X_test, y_test):

    # Treinar
    model.fit(X_train, y_train)

    # Previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Métricas
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return results

# %%
#E por fim rodamos todos os modelos para comparar

results_table = []

for name, model in models.items():
    
    print(f"\n🔍 Treinando modelo: {name}...\n")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    row = {
        "Modelo": name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "AUC": metrics["AUC"]
    }

    results_table.append(row)

    print("Matriz de Confusão:")
    print(metrics["Confusion Matrix"])
    print("-" * 60)

# DataFrame final com os resultados
results_df = pd.DataFrame(results_table)
results_df


# %%
#Conclusões sobre o benchmark

# Nesta etapa, comparamos diferentes modelos supervisionados para entender o trade-off entre:
#- capacidade de detecção de fraude (recall)
#- custo operacional de falsos positivos

# O objetivo não é apenas maximizar métricas, mas escolher um modelo viável em produção.

#Entre elas, XGBoost tem maior recall, seguido por random forest e depois KNN, o qual não performa bem em grandes datasets
#Logo, em conclusão, os melhores modelos a serem usados são XGBoost ou Random Forest
#XGBoost: lida bem com problemas complexos e também desbalanceados, tem alta performance e controle fino de hiperparâmetros
#Random Forest não será utilizado pois pode ser lento com datasets grandes, e foi lento no benchmark em questão
#Portanto, o modelo utilizado será XGBoost.

# %%
#Confeccionando o modelo XGBoost após o benchmark

from xgboost import XGBClassifier


# %%
model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)


# %%
model_xgb.fit(X_train, y_train)


# %%
y_pred = model_xgb.predict(X_test)


# %%
y_proba = model_xgb.predict_proba(X_test)[:, 1]


# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))


# %%
#Fase 5: Ajuste de threshold com base em custo de erro

#Os resultados nos mostram ótimo valor de f1-score, o que indica bons valores de precisão e recall.
#Podemos ajustar o valor de threshold para aumentar o recall, visto que isso muda o comportamento do modelo
# O ajuste de threshold é uma decisão de negócio.

# Threshold menor:
# - aumenta recall
# - reduz fraudes não detectadas
# - aumenta custo operacional (falsos positivos)

# Threshold maior:
# - reduz intervenções
# - aumenta risco financeiro

# Neste contexto, priorizamos recall, pois o custo da fraude é maior que o custo do falso positivo.

# %%
#Agora, vamos testar alguns valores de threshold, entre 0.3 a 0.7 para observar  qual é o ideal
#Calcularemos precisão, recall e F1 para cada threshold
#Também mostraremos graficamente como precision e recall mudam conforme alteramos threshold


# %%
y_proba = model_xgb.predict_proba(X_test)[:, 1]

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.30, 0.71, 0.05)  # thresholds de 0.30 a 0.70
results = []

for t in thresholds:
    y_pred_adj = (y_proba >= t).astype(int)

    precision = precision_score(y_test, y_pred_adj, zero_division=0)
    recall = recall_score(y_test, y_pred_adj)
    f1 = f1_score(y_test, y_pred_adj)

    results.append([t, precision, recall, f1])

# Mostrar resultados em tabela
import pandas as pd
threshold_df = pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1-score"])

threshold_df


# %%
#Gráfico de precision e recall de acordo com a variação de threshold

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Geração das curvas de precision, recall e thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Criar figura
plt.figure(figsize=(10, 6))

# Plotar Precision e Recall em função do Threshold
plt.plot(thresholds, precision[:-1], label='Precision', linewidth=2)
plt.plot(thresholds, recall[:-1], label='Recall', linewidth=2)

# Detalhes do gráfico
plt.title("Curva Precision x Recall x Threshold", fontsize=16)
plt.xlabel("Threshold", fontsize=14)
plt.ylabel("Valor", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

plt.show()


# %%
#Portanto, threshold igual a 0.3 é o que apresentou melhor equilíbrio entre as métricas estudadas.
#Esse valor reduz falsos negativos sem comprometer muito os falsos positivos
#O gráfico acima mostra que o ponto de equilíbrio entre precisão e recall fica próximo de 0.25 à 0.27
#É importante ressaltar que esse ponto é só uma referência, e não necessariamente maximiza nossa métrica mais importante (recall)

# %%
#Agora vamos ajustar o código final do modelo XGBoost para threshold=0.3

# TREINAMENTO DO MODELO XGBOOST

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# 1) Criar e treinar o modelo
model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model_xgb.fit(X_train, y_train)

# PREVISÕES COM THRESHOLD PERSONALIZADO

# 2) Probabilidades da classe 1 (fraude)
y_proba = model_xgb.predict_proba(X_test)[:, 1]

# 3) Aplicar o threshold = 0.30
threshold = 0.30
y_pred_adj = (y_proba >= threshold).astype(int)

# AVALIAÇÃO DO MODELO AJUSTADO

# 4) Métricas finais
precision = precision_score(y_test, y_pred_adj)
recall = recall_score(y_test, y_pred_adj)
f1 = f1_score(y_test, y_pred_adj)
auc = roc_auc_score(y_test, y_proba)

print(f"Threshold utilizado: {threshold}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# 5) Classification Report
print("\nClassification Report (com threshold ajustado):")
print(classification_report(y_test, y_pred_adj))

# 6) Matriz de confusão
print("Matriz de confusão (com threshold = 0.30):")
print(confusion_matrix(y_test, y_pred_adj))



