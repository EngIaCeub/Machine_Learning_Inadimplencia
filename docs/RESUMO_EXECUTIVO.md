# Resumo Executivo — Credit Default A1

Foi desenvolvido um pipeline de classificacao para prever `default payment next month` no dataset UCI id=350. `default=1` e a classe positiva. O candidato final, congelado antes do holdout, e um `CatBoostClassifier` com feature set A3 (`ROUND1 + BILL + PAYMENT`) e threshold `0.247743`.

| Metrica TEST | Resultado | Gate |
|---|---:|---|
| ROC-AUC | 0.7865 | PASS |
| Recall `default=1` | 0.6104 | PASS |
| Macro F1 | 0.7019 | PASS |
| Overall | 3/3 | PASS |

Validation apresentou AUC 0.7811, Recall 0.6050 e Macro F1 0.7025. O Binary F1 da classe positiva foi 0.5502 no TEST e permanece diagnostico, nao sendo o gate oficial A1. Os resultados Validation/Test ficaram proximos.

O TEST foi aberto somente apos o congelamento. Houve um segundo acesso fisico deterministico para corrigir a persistencia da chave `micro_f1`; nao houve retuning, selecao de runner-up ou alteracao de features/modelo/threshold. A principal limitacao e a sobreposicao entre classes, que manteve Binary F1 positivo proximo de 0.55.

**Conclusao:** sob o protocolo A1 adotado, os tres gates oficiais foram atendidos e a entrega esta pronta para submissao. Consulte [RELATORIO_FINAL.md](RELATORIO_FINAL.md) e [GUIA_REPRODUCAO.md](GUIA_REPRODUCAO.md).

