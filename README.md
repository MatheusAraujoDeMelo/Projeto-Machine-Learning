# 💳 Modelagem de Risco de Crédito em Duas Camadas

> **Projeto de Machine Learning Clássico — Centro Universitário UNISATC**  
> *Predição de Inadimplência (Default) e Estimativa de Exposição Financeira usando dados do UCI Machine Learning Repository.*

---

## 📌 Sobre o Projeto

Este projeto desenvolve e avalia um sistema preditivo avançado em **duas camadas** focado no gerenciamento de risco de crédito bancário. A abordagem vai além da simples classificação de *default*, integrando a estimativa do valor financeiro em risco para dar suporte à tomada de decisão de instituições financeiras.

### 🏗️ Arquitetura do Sistema
* **Camada 1 (Classificação Binária - Probabilidade de Default):**
  * **Objetivo:** Identificar clientes em risco de inadimplência no mês subsequente (`default.payment.next.month`).
  * **Algoritmos:** Comparação entre **Regressão Logística** (`class_weight='balanced'`) e ***K-Nearest Neighbors* (KNN, K=5)**.
  * **Destaque:** Calibração do limiar de decisão (\\(\tau = 0,42\\)) para contornar o desbalanceamento da base (78% vs. 22%) e priorizar a taxa de captura de inadimplentes (**Recall**).
* **Camada 2 (Regressão Contínua - Exposição Financeira):**
  * **Objetivo:** Estimar o valor contínuo da fatura corrente em risco (`BILL_AMT1`).
  * **Algoritmos:** Comparação entre **Regressão Linear Múltipla (OLS)** e **Regressão Ridge (Regularização \\(\mathcal{L}_2\\))** para conter a multicolinearidade de faturas históricas.

---

## 📊 Principais Resultados

### Camada 1: Classificação de Inadimplência (Conjunto de Teste \\(N=6.000\\))

| Modelo / Configuração | Acurácia Global | Precisão (Classe 1) | Recall (Inadimplentes) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN (\\(K=5\\), \\(\tau=0,50\\))** | 79,00% | 56,00% | 33,00% | 0,4200 | 0,6988 |
| **Regressão Logística (\\(\tau=0,50\\))** | 77,52% | 49,26% | 55,54% | 0,5221 | 0,7585 |
| **Regressão Logística (\\(\tau=0,42\\)) 🏆** | **70,33%** | **39,70%** | **65,79%** | **0,4952** | **0,7585** |

* **Conclusão:** O ajuste do limiar para \\(\tau = 0,42\\) na Regressão Logística permitiu capturar **65,79% dos inadimplentes reais** (dobrando o desempenho do KNN, que capturava apenas 33%), mantendo uma taxa de aprovação saudável para 72% dos adimplentes.

### Camada 2: Regressão da Exposição Financeira (`BILL_AMT1`)

| Modelo | MAE (Erro Médio Absoluto) | RMSE | \\(R^2\\) (Coeficiente de Determinação) |
| :--- | :---: | :---: | :---: |
| **Regressão Linear Múltipla (OLS)** | NT\$ 8.162,10 | NT\$ 18.235,50 | 93,02% |
| **Regressão Ridge (\\(\mathcal{L}_2\\)) 🏆** | **NT\$ 8.154,30** | **NT\$ 18.210,45** | **93,08%** |

---

## 🏛️ Matriz de Decisão Bancária

| Probabilidade Predita (\\(\hat{p}\\)) | Nível de Risco | Ação Operacional Recomendada |
| :---: | :---: | :--- |
| **\\(\hat{p} < 30\%\\)** | Baixo Risco | Concessão/renovação automática de limite sem atrito. |
| **\\(30\% \le \hat{p} < 42\%\\)** | Risco Moderado | Manutenção do limite atual e inclusão em régua de monitoramento. |
| **\\(42\% \le \hat{p} < 60\%\\)** | Risco Elevado | Bloqueio de aumentos de limite e alertas preventivos por SMS/E-mail. |
| **\\(\hat{p} \ge 60\%\\)** | Risco Crítico | Suspensão temporária do cartão e acionamento de renegociação prévia. |

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Manipulação de Dados:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Pipelines, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, `KNeighborsClassifier`, `Ridge`)
* **Visualização:** `matplotlib`, `seaborn`
* **Ambiente de Desenvolvimento:** VS Code / Jupyter Notebook

---

## 📂 Estrutura do Repositório

```text
.
├── data/
│   └── UCI_Credit_Card.csv   # Dataset bruto do UCI
├── notebooks/
│   └── 01_eda_e_preprocessamento.ipynb      # Análise exploratória e Modelagem
└── README.md                            