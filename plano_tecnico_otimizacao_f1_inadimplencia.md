# Plano Técnico — Otimização do F1 Binário para Inadimplência

## 1. Objetivo

Modificar experimentalmente o pipeline de classificação de inadimplência para tentar atingir simultaneamente os seguintes critérios de validação:

- **ROC-AUC ≥ 0,75**
- **Recall da classe `default = 1` ≥ 0,60**
- **F1 binário da classe `default = 1` ≥ 0,65**

### Estado atual de referência

- ROC-AUC: **0,7841**
- Recall `default=1`: **0,6121**
- F1 binário `default=1`: **0,5447**
- Macro F1: **0,6971**
- Weighted F1: **0,7821**

O objetivo principal é elevar o **F1 binário**, preservando os dois gates já atingidos.

### Base documental

O material **Como Avaliar Modelos de Machine Learning** estabelece para a Trilha A / A1 — Inadimplência:

- AUC-ROC ≥ 0,75
- F1 ≥ 0,65
- Recall ≥ 0,60

O mesmo material define F1 como a métrica que equilibra Precisão e Recall e diferencia explicitamente `F1-Score macro` quando pretende utilizar macro averaging. Portanto, este plano adota como gate o **F1 binário da classe positiva `default=1`**.

---

## 2. Regra principal da otimização

A partir deste ponto, a seleção do melhor modelo **não deve ser feita apenas pelo maior ROC-AUC**.

Utilizar como regra de decisão:

> Entre todas as configurações com ROC-AUC ≥ 0,75 e Recall da classe 1 ≥ 0,60, selecionar aquela que apresentar o maior F1 binário da classe `default=1`.

### Função objetivo

Maximizar:

```text
F1_binary(default=1)
```

Sujeito a:

```text
ROC_AUC >= 0.75
Recall(default=1) >= 0.60
```

Meta final:

```text
F1_binary(default=1) >= 0.65
```

---

## 3. Restrições obrigatórias

Antes de qualquer alteração:

- manter `default = 1` como classe positiva;
- não inverter os rótulos;
- não utilizar Macro F1 ou Weighted F1 como substitutos do F1 binário;
- continuar calculando Macro F1 e Weighted F1 apenas para diagnóstico;
- não utilizar o conjunto de teste para tuning;
- não selecionar threshold utilizando o conjunto de teste;
- não realizar oversampling/undersampling antes da divisão dos conjuntos;
- não permitir data leakage;
- manter seeds fixas para reprodutibilidade;
- preservar o pipeline atual funcionando antes de implementar mudanças;
- criar uma baseline reproduzível com as métricas atuais antes de iniciar os experimentos.

---

## 4. FASE 0 — Auditoria do pipeline atual

Antes de modificar qualquer coisa, localizar no código:

1. carregamento do target;
2. definição da classe positiva;
3. separação train/validation/test;
4. treinamento dos modelos;
5. geração de `predict_proba`;
6. aplicação do threshold;
7. cálculo de Precision;
8. cálculo de Recall;
9. cálculo do F1;
10. cálculo de ROC-AUC;
11. rotina de tuning;
12. lógica utilizada para selecionar o melhor modelo.

Confirmar explicitamente:

```python
positive_class = 1
```

e que o F1 utilizado para o gate equivale conceitualmente a:

```python
f1_score(y_true, y_pred, pos_label=1, average="binary")
```

Não modificar código nesta fase.

Gerar um pequeno relatório de auditoria indicando:

- arquivo;
- função;
- linha aproximada;
- comportamento encontrado;
- necessidade ou não de alteração.

---

## 5. FASE 1 — Otimização de threshold

Esta é a primeira intervenção e deve ser executada antes de qualquer retreinamento pesado.

### 5.1 Utilizar probabilidades

Para cada modelo treinado, obter:

```python
y_score = model.predict_proba(X_val)[:, 1]
```

O ROC-AUC deve continuar sendo calculado sobre `y_score`.

Não calcular AUC a partir das classes já thresholdadas.

### 5.2 Construir threshold search

Testar diferentes thresholds utilizando exclusivamente o conjunto de validação.

Preferencialmente utilizar os thresholds derivados da curva Precision-Recall ou uma grade suficientemente fina.

Para cada threshold `t`:

```python
y_pred = (y_score >= t).astype(int)
```

Calcular:

- threshold;
- Precision classe 1;
- Recall classe 1;
- F1 binário classe 1;
- Macro F1;
- Weighted F1;
- Accuracy;
- TP;
- FP;
- TN;
- FN.

Registrar cada resultado.

### 5.3 Regra de seleção

Considerar elegível somente threshold com:

```text
Recall >= 0.60
ROC-AUC >= 0.75
```

Entre os elegíveis:

```text
best_threshold = threshold com maior F1_binary
```

Em caso de empate:

1. maior Recall;
2. maior Precision;
3. threshold mais próximo de 0,5.

### 5.4 Resultado esperado da fase

Gerar uma tabela semelhante a:

| threshold | precision_1 | recall_1 | f1_binary | macro_f1 | auc |
|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... |

Informar:

```text
Baseline F1:
Best threshold F1:
Gain:
Best threshold:
Recall:
Precision:
AUC:
```

### Gate de decisão

Se:

```text
F1 >= 0.65
Recall >= 0.60
AUC >= 0.75
```

interromper a busca de modificações estruturais.

Não realizar mudanças desnecessárias no modelo.

Caso contrário, avançar para a Fase 2.

---

## 6. FASE 2 — Tuning orientado ao F1

O tuning atual deve ser revisado.

Não assumir que o modelo com maior AUC produz automaticamente o melhor F1.

Para cada configuração candidata:

1. treinar o modelo;
2. gerar probabilidades na validação;
3. calcular ROC-AUC;
4. executar threshold optimization;
5. obter o máximo F1 binário possível mantendo Recall ≥ 0,60;
6. registrar resultado.

A unidade avaliada passa a ser:

```text
modelo + hiperparâmetros + threshold
```

e não apenas:

```text
modelo + hiperparâmetros
```

---

## 7. FASE 3 — Ajuste de pesos da classe positiva

Para modelos que suportem pesos de classe, investigar aumento da importância da classe `default=1`.

No XGBoost, localizar o uso de:

```python
scale_pos_weight
```

Calcular primeiro a razão de referência:

```python
n_negative / n_positive
```

Não assumir automaticamente que essa razão é o melhor valor.

Testar valores ao redor dela.

Exemplo conceitual:

```text
0.50 × razão
0.75 × razão
1.00 × razão
1.25 × razão
1.50 × razão
```

Também incluir:

```text
scale_pos_weight = 1
```

como controle.

Para cada configuração:

- treinar;
- otimizar threshold;
- calcular F1 binário;
- verificar Recall;
- verificar AUC.

A configuração vencedora deve continuar sendo escolhida pelo maior F1 sob as restrições dos gates.

---

## 8. FASE 4 — Hiperparâmetros do XGBoost

Depois de investigar `scale_pos_weight`, realizar tuning controlado dos principais hiperparâmetros que afetam generalização:

- `learning_rate`
- `n_estimators`
- `max_depth`
- `min_child_weight`
- `gamma`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`
- `scale_pos_weight`

Evitar grid search combinatorial excessivamente grande.

Preferir:

- mecanismo de tuning já existente no projeto;
- random search;
- busca otimizada já disponível nas dependências do projeto.

Não adicionar novas bibliotecas apenas para tuning sem necessidade.

---

## 9. Função de avaliação única

Criar ou adaptar uma função central de avaliação para evitar inconsistências.

Estrutura esperada:

```python
evaluate_binary_classifier(
    y_true,
    y_score,
    threshold
)
```

Ela deve retornar pelo menos:

```python
{
    "threshold": ...,
    "roc_auc": ...,
    "precision_positive": ...,
    "recall_positive": ...,
    "f1_binary": ...,
    "f1_macro": ...,
    "f1_weighted": ...,
    "accuracy": ...,
    "tn": ...,
    "fp": ...,
    "fn": ...,
    "tp": ...
}
```

Todos os experimentos devem utilizar essa mesma função.

---

## 10. FASE 5 — Análise Precision-Recall

Gerar a curva Precision-Recall do melhor modelo.

Identificar visualmente:

- região onde Recall ≥ 0,60;
- Precision correspondente;
- máximo F1;
- threshold correspondente.

Criar também uma curva:

```text
Threshold × F1
```

e, se simples de implementar:

```text
Threshold × Precision
Threshold × Recall
```

O objetivo é verificar se existe uma região de threshold capaz de produzir simultaneamente Precision e Recall suficientes.

---

## 11. Diagnóstico de teto do modelo

Após threshold tuning + class weighting + hyperparameter tuning, determinar o melhor F1 encontrado.

### Cenário A

```text
F1 >= 0.65
Recall >= 0.60
AUC >= 0.75
```

Resultado:

**Meta atingida.**

Encerrar otimização.

### Cenário B

```text
F1 entre aproximadamente 0.60 e 0.65
```

Resultado:

Modelo está próximo da meta.

Avançar para engenharia de atributos e/ou ensemble.

### Cenário C

```text
F1 permanece próximo da baseline
```

Resultado:

Threshold e hiperparâmetros não são o principal gargalo.

Provável limitação de separabilidade ou representação das features.

Avançar para análise de atributos.

---

## 12. FASE 6 — Engenharia e seleção de atributos

Somente executar esta fase se as anteriores não atingirem a meta.

Analisar:

- feature importance;
- SHAP, caso já exista no projeto;
- correlações;
- atributos pouco informativos;
- possíveis interações;
- atributos categóricos e sua codificação;
- distribuição dos atributos por classe;
- possíveis outliers;
- possíveis transformações.

Não remover features exclusivamente porque a importância individual ficou baixa sem testar o impacto.

Criar experimentos comparáveis:

```text
features baseline
vs.
features modificadas
```

Sempre repetir:

```text
treino
→ predict_proba
→ threshold optimization
→ avaliação dos gates
```

---

## 13. FASE 7 — Estratégias de balanceamento

Se necessário, testar técnicas de balanceamento exclusivamente no conjunto de treinamento.

Ordem preferencial:

1. pesos de classe;
2. undersampling controlado;
3. oversampling;
4. técnicas sintéticas, apenas se justificadas.

Validação e teste devem manter a distribuição original.

Nunca aplicar resampling antes da separação train/validation/test.

Cada experimento deve ser comparado com a baseline utilizando exatamente o mesmo conjunto de validação.

---

## 14. FASE 8 — Comparação entre algoritmos

Reavaliar os algoritmos já existentes no pipeline utilizando o novo critério.

Para cada modelo:

```text
treinar
↓
predict_proba
↓
threshold optimization
↓
best binary F1 subject to Recall >= 0.60
↓
comparar
```

Não comparar modelos utilizando thresholds diferentes sem registrar explicitamente cada threshold.

Criar tabela final:

| modelo | threshold | AUC | Precision 1 | Recall 1 | F1 binary | status |
|---|---:|---:|---:|---:|---:|---|
| XGB | | | | | | |
| HGB | | | | | | |
| RF | | | | | | |

Ordenar por:

1. gates satisfeitos;
2. maior F1 binário;
3. maior AUC.

---

## 15. Ensemble — somente se necessário

Caso modelos diferentes apresentem erros complementares, testar ensemble simples de probabilidades.

Exemplo:

```python
p_final = (
    w1 * p_xgb +
    w2 * p_hgb +
    w3 * p_rf
)
```

com:

```text
w1 + w2 + w3 = 1
```

O ensemble também deve passar pela busca de threshold.

Não manter ensemble caso o ganho seja irrelevante.

Preferir o modelo mais simples quando os resultados forem equivalentes.

---

## 16. Prevenção de overfitting na validação

Como serão testadas várias combinações, não utilizar o conjunto de teste durante a busca.

O teste deve permanecer completamente isolado.

Fluxo correto:

```text
TRAIN
    ↓
treinamento

VALIDATION
    ↓
hyperparameter selection
threshold selection
model selection

TEST
    ↓
uma única avaliação final
```

Se o pipeline já possuir cross-validation dentro do treinamento, preservá-la.

Não alterar a metodologia de validação sem necessidade.

---

## 17. Resultado final a ser persistido

Persistir:

```text
best_model
best_hyperparameters
best_threshold
validation_auc
validation_precision
validation_recall
validation_f1_binary
validation_macro_f1
validation_weighted_f1
confusion_matrix
```

O threshold deve fazer parte do artefato/configuração do modelo.

Não deixar um valor fixo `0.5` espalhado pelo código.

Exemplo conceitual:

```python
MODEL_THRESHOLD = best_threshold
```

ou arquivo de configuração equivalente.

---

## 18. Testes obrigatórios

Adicionar ou atualizar testes garantindo:

### Target

```text
default = 1 é a classe positiva.
```

### Threshold

Verificar:

```python
prediction = (probability >= threshold)
```

### F1

Garantir que o gate utiliza:

```text
F1 binário da classe 1
```

e não:

```text
macro
weighted
micro
```

### Leakage

Garantir que:

```text
test não participa do tuning.
```

### Reprodutibilidade

Executar novamente com a mesma seed e verificar estabilidade dos resultados.

Não quebrar os testes existentes.

---

## 19. Artefatos de experimentação

Criar um diretório de resultados, respeitando a estrutura atual do projeto, contendo algo equivalente a:

```text
experiments/
    threshold_search.csv
    model_comparison.csv
    tuning_results.csv
    best_configuration.json
    metrics_summary.json
```

Também gerar, quando a estrutura atual permitir:

```text
precision_recall_curve.png
f1_vs_threshold.png
confusion_matrix.png
```

Não adicionar arquivos redundantes se já existir uma estrutura equivalente.

---

## 20. Registro de cada experimento

Cada execução deve registrar:

```text
experiment_id
timestamp
model
hyperparameters
class_weight/scale_pos_weight
threshold
precision
recall
f1_binary
macro_f1
weighted_f1
roc_auc
seed
```

Isso é obrigatório para evitar decisões baseadas em execuções isoladas.

---

## 21. Critério formal de sucesso

Considerar o trabalho tecnicamente bem-sucedido somente se houver pelo menos uma configuração de validação com:

```python
roc_auc >= 0.75
and recall_positive >= 0.60
and f1_binary >= 0.65
```

Criar explicitamente:

```python
meets_requirements = (
    roc_auc >= 0.75
    and recall_positive >= 0.60
    and f1_binary >= 0.65
)
```

Não usar arredondamento para fazer uma métrica atingir artificialmente o requisito.

---

## 22. Ordem obrigatória dos experimentos

Não executar tudo simultaneamente.

Seguir esta sequência:

### Experimento 0
Reproduzir baseline.

### Experimento 1
Threshold optimization com o modelo atual.

### Experimento 2
Threshold + pesos de classe.

### Experimento 3
Threshold + pesos + hyperparameter tuning.

### Experimento 4
Comparação dos algoritmos existentes usando a nova estratégia.

### Experimento 5
Feature engineering/selection, somente se necessário.

### Experimento 6
Resampling, somente se necessário.

### Experimento 7
Ensemble, somente se necessário.

Interromper assim que houver uma solução estável que satisfaça os três gates.

---

## 23. Não fazer

Não:

- trocar F1 binário por Macro F1 para declarar aprovação;
- utilizar Weighted F1 como gate;
- inverter a classe positiva;
- utilizar dados do teste para selecionar threshold;
- maximizar Recall indefinidamente sacrificando Precision;
- otimizar apenas ROC-AUC;
- aplicar SMOTE no dataset inteiro;
- alterar várias etapas simultaneamente sem medir contribuição individual;
- manter alterações que não produzam ganho;
- refatorar partes não relacionadas do projeto;
- mudar contratos públicos desnecessariamente;
- quebrar testes já existentes.

---

## 24. Relatório final do agente

Ao terminar, responder utilizando exatamente esta estrutura:

### STATUS

`TARGET_REACHED` ou `TARGET_NOT_REACHED`

### BASELINE

```text
AUC:
Precision:
Recall:
F1 binary:
Threshold:
```

### BEST RESULT

```text
Model:
AUC:
Precision:
Recall:
F1 binary:
Macro F1:
Weighted F1:
Threshold:
```

### REQUIREMENTS

```text
AUC >= 0.75: PASS/FAIL
Recall >= 0.60: PASS/FAIL
F1 binary >= 0.65: PASS/FAIL
```

### IMPROVEMENT

```text
Baseline F1:
Final F1:
Absolute gain:
```

### BEST CONFIGURATION

Listar hiperparâmetros e threshold.

### CHANGES

Listar arquivos modificados.

### EXPERIMENTS

Mostrar tabela resumida dos principais experimentos.

### TESTS

Informar quantidade de testes aprovados/falhos.

### CONCLUSION

Explicar objetivamente:

- se a meta foi atingida;
- qual mudança produziu maior ganho;
- se o ganho veio principalmente de threshold, pesos, tuning, features ou ensemble;
- se existe evidência de overfitting;
- se recomenda aceitar a solução como modelo final.

---

## 25. Princípio de implementação

Priorizar:

> **menor alteração → maior ganho mensurável → menor risco de leakage → maior reprodutibilidade**

Não procurar tornar o projeto mais complexo.

O objetivo é descobrir de forma experimental e rastreável se o modelo consegue elevar o F1 binário de **0,5447 para ≥ 0,65**, mantendo simultaneamente **Recall ≥ 0,60 e ROC-AUC ≥ 0,75**.
