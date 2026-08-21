# Relatorio Final — Credit Default A1

## 1. Introducao e objetivo

Este trabalho implementa um pipeline reproduzivel para identificar clientes com probabilidade de inadimplencia no dataset UCI Default of Credit Card Clients. O objetivo foi avaliar um candidato congelado sob o protocolo A1, usando `default = 1` como classe positiva e os gates oficiais de ROC-AUC, Recall e Macro F1.

## 2. Problema e dados

O problema e uma classificacao binaria de risco de default. O dataset UCI id=350 possui aproximadamente 30 mil observacoes, com limite de credito, dados demograficos, historico de atraso, faturas e pagamentos. O target e normalizado explicitamente para `default payment next month`, com valor `1` como evento positivo.

O split e estratificado e deterministico em 70% TRAIN, 15% VALIDATION e 15% TEST. O preprocessing e ajustado somente no TRAIN. A feature pipeline final usa A3: `ROUND1 + BILL + PAYMENT`, preservando variaveis semanticas, agregacoes de faturas e pagamentos e atributos de atraso.

## 3. Metricas e protocolo

Durante o desenvolvimento, VALIDATION foi usada para analise, threshold e comparacao. A Round 4 adotou selecao prioritariamente baseada em TRAIN/OOF para reduzir reuso de VALIDATION. O candidato final foi congelado antes do holdout.

Na correcao A1, a definicao oficial passou a ser `f1_macro`. O Binary F1 da classe positiva continua sendo reportado como diagnostico e nao foi substituido silenciosamente. Os gates sao:

| Metrica | Gate |
|---|---:|
| ROC-AUC | >= 0.75 |
| Recall de `default=1` | >= 0.60 |
| Macro F1 | >= 0.65 |

## 4. Desenvolvimento experimental

### Round 1

Foram avaliados o pipeline inicial, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting e XGBoost, com threshold search e tratamento controlado de desbalanceamento. O principal resultado foi a identificacao de um plateau de Binary F1 proximo de 0.55.

### Round 2

Foram testados atributos de BILL, PAYMENT, PAY, razoes, tendencias, ablations, reamostragem e ensembles. A representacao `ROUND1 + BILL + PAYMENT` foi a melhor entre as alternativas avaliadas. O XGBoost alcancou Binary F1 aproximadamente 0.5501, sem atingir 0.65.

### Round 3

Hard Negative Mining, pesos, cascade, thresholds segmentados e diagnosticos de separabilidade foram investigados. Houve reducao de falsos positivos em algumas variantes, mas sem ganho material de Binary F1. Foram observados 27 grupos de duplicatas exatas, 11 grupos com labels conflitantes e taxa de vizinhos opostos de 26.55%; isso indica sobreposicao relevante, mas nao prova um limite matematico do dataset.

### Round 4

Foi testada representacao temporal explicita e CatBoost. A alternativa temporal nao produziu ganho material generalizavel. O candidato CatBoost A3 foi promovido com base em selecao OOF e depois reavaliado sob a correcao A1.

## 5. Correcao metodologica A1 e modelo final

O modelo final e `CatBoostClassifier` com feature set A3 (`ROUND1 + BILL + PAYMENT`). A configuracao congelada e: 250 iteracoes, profundidade 6, learning rate 0.05, `l2_leaf_reg=5.0`, `random_strength=1.0`, `bagging_temperature=1.0`, seed 42. O threshold congelado e `0.247743`.

O vencedor foi escolhido por maior Macro F1 entre candidatos que atendiam AUC e Recall. CatBoost nao e apresentado como universalmente superior ao XGBoost; a diferenca entre candidatos foi pequena e depende do protocolo.

## 6. Resultados

| Metrica | Validation | Test | Gate A1 | Test status |
|---|---:|---:|---:|---|
| ROC-AUC | 0.7811 | 0.7865 | >= 0.75 | PASS |
| Recall `default=1` | 0.6050 | 0.6104 | >= 0.60 | PASS |
| Macro F1 | 0.7025 | 0.7019 | >= 0.65 | PASS |
| Binary F1 `default=1` | 0.5498 | 0.5502 | diagnostico | — |
| Weighted F1 | 0.7877 | 0.7865 | diagnostico | — |

### Matriz de confusao no TEST

| | Predito 0 | Predito 1 |
|---|---:|---:|
| Real 0 | TN 2898 | FP 606 |
| Real 1 | FN 388 | TP 608 |

O modelo identificou corretamente 608 inadimplentes e deixou de identificar 388. Tambem classificou 606 clientes sem inadimplencia como positivos e classificou corretamente 2898 negativos. Esses numeros descrevem o comportamento observado; nao estabelecem causalidade.

## 7. Diagnostico do Binary F1

O Binary F1 da classe positiva permaneceu aproximadamente 0.55, apesar de AUC e Recall atenderem aos criterios. Esse resultado evidencia a dificuldade especifica de obter simultaneamente precision alta e recall alto para `default=1` sob o protocolo adotado. O Macro F1 oficial inclui o desempenho das duas classes e foi explicitamente confirmado como criterio A1; Macro F1 e Weighted F1 nao devem ser reinterpretados como Binary F1.

## 8. Generalizacao

As diferencas TEST menos VALIDATION foram: AUC +0.0054, Precision -0.0029, Recall +0.0054, Binary F1 +0.0005, Macro F1 -0.0006 e Weighted F1 -0.0012. Os resultados permaneceram proximos, indicando estabilidade do candidato congelado no holdout final sob este protocolo, sem afirmar generalizacao perfeita.

## 9. Leakage e controle experimental

- O target nao foi usado na criacao das features.
- O preprocessing foi fitado somente no TRAIN.
- Reamostragem, quando utilizada em experimentos, ocorreu somente no TRAIN.
- OOF e thresholds da Round 4 foram derivados dentro do TRAIN.
- Modelo, feature set, threshold e metrica oficial foram congelados antes do TEST.
- TEST nao participou de selecao, tuning, threshold ou feature selection.

O acesso fisico ao TEST foi contado como 2: o primeiro abriu a avaliacao final do candidato congelado; o segundo foi uma recuperacao deterministica causada pela ausencia da chave `micro_f1` na persistencia. Nenhuma decisao de modelagem foi tomada a partir do TEST, nenhum runner-up foi avaliado e nao houve retuning.

## 10. Eficiencia e interpretacao

A busca de threshold foi vetorizada de aproximadamente 91.5 s para aproximadamente 0.03 s, speedup aproximado de 3050x. Isso e uma melhoria operacional, nao uma melhoria estatistica do modelo.

Os grupos de variaveis utilizados representam historico PAY, comportamento de BILL, comportamento de PAYMENT, atrasos acumulados e tendencias/agregacoes. Importancia de variavel significa associacao com a decisao do modelo; nao significa efeito causal.

## 11. Limitacoes

As principais limitacoes sao a sobreposicao relevante entre classes, o Binary F1 positivo proximo de 0.55, o reuso de VALIDATION nas Rounds 1–3, a diferenca pequena entre CatBoost e XGBoost, a ambiguidade inicial sobre a definicao de F1, labels conflitantes em parte dos vetores duplicados e a dependencia dos resultados ao dataset e ao protocolo. A Round 4 usou OOF para reduzir o risco de selecao excessiva, mas isso nao apaga o historico experimental anterior.

## 12. Reprodutibilidade

As instrucoes estao em [GUIA_REPRODUCAO.md](GUIA_REPRODUCAO.md). Os artefatos fonte ficam em `artifacts/final/`, incluindo o vencedor, metricas, auditoria de leakage, comparacao Validation/Test, predictions do holdout e figuras. O TEST ja foi consumido como holdout final e nao deve ser executado novamente por padrao.

## 13. Conclusao

O `CatBoostClassifier` com A3 satisfez no TEST os tres criterios A1: ROC-AUC 0.7865, Recall de `default=1` 0.6104 e Macro F1 0.7019. O candidato, as features, o threshold e a definicao das metricas estavam congelados antes do holdout. Os resultados permaneceram proximos aos de VALIDATION, sem tuning posterior ao TEST. Binary F1 aproximadamente 0.55 permanece como diagnostico da dificuldade especifica da classe inadimplente.

