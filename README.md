# Project 1

## 1. O que é Machine Learning
Machine learning é uma área da inteligência artificial que se concentra no desenvolvimento de algoritmos e modelos que permitem que as máquinas aprendam com os dados fornecidos e melhorem suas habilidades de tomada de decisão ao longo do tempo. Em outras palavras, é uma técnica que permite que os computadores aprendam a partir de exemplos e experiências passadas, sem serem explicitamente programados para cada tarefa específica. O objetivo final do machine learning é permitir que as máquinas realizem tarefas complexas e tomem decisões precisas e automatizadas, sem a intervenção humana. O Machine Learning faz parte de uma das 8 subáreas da inteligência artificial.

## 2. Tipos de Machine Learning
O Machine Learning pode ser dividido em 4 categorias, são elas:
  * Supervisionado - Quando o modelo de conhecimento é construído a partir dos dados apresentados na forma de pares ordenados, como entrada e saída desejada
  * Não Supervisionado - Quando não existe a informação da saída desejada e o processo de aprendizado busca identificar regularidades entre os dados, a fim de agrupá-los em função das similaridades que apresentam entre si
  * Semissupervisionado - Combina o aprendizado supervisionado e o não supervisionado
  * Por Reforço - Quando a máquina é capaz de “perceber” o estado do ambiente, executar ações de acordo com ele, receber “recompensas” pelas ações executadas e trocar de estado, se apropriado 

### 3. Supervisionado
Como mencionado a cima, no aprendizado supervisionado, o modelo é construído a partir dos dados de entrada (também chamados de dataset), que são apresentados para um algoritmo na forma de pares ordenados (entrada-saída desejados). Diz-se que esses dados são rotulados, pois sabemos de antemão a saída esperada para cada entrada. Um exemplo prático seria a classificação de animais como "gato" ou "cachorro" a partir de duas informações: peso e altura. Neste exemplo, os dados de entrada são as características do animal (peso e altura) e a saída desejada é o animal (gato ou cachorro), em que seria-se descoberto a partir do peso e altura de um animal se ele é um gato ou um cachorro com base em informações anteriores de peso e altura de gatos e cachorros. Em suma, os pares ordenados seriam: ((peso, altura), 'Cachorro / Gato'). 
O aprendizado supervisionado pode ser divido em duas categorias:
  * Classificação - Usada para prever a classe ou categoria de um objeto. Exemplo: "Deve-se conceder ou não crédito para um cliente de um banco?", neste caso a variável a ser predita é categórica, pois a resposta deverá ser sim ou não (**saída sempre categórica**)
  * Regressão - Usada para prever um valor numérico ou contínuo. Exemplo: "Deve-se conceder qual valor de crédito a um cliente de um banco?", neste caso, a variável a ser predita é numérica (contínua ou disreta) (**saída sempre numérica discreta ou contínua)**

Além disso, é comum que particionar os dados de entrada (rotulados) em dois conjuntos:
  * De treinamento - Servirá para construir o modelo
  * De teste - Também chamado na literatura de conjunto de validação, servirá para verificar como o modelo se comportaria em dados não vistos, de forma que possamos ajustá-lo, se necessário, para a construção final do modelo a ser aplicado a novos dados em que ainda não conhecemos a saída esperada

#### 3.1 Classificação
A classificação é uma das tarefas de machine learning mais importantes e populares. **Pode-se definir um problema de classificação, informalmente, como a busca por uma função matemática que permita associar corretamente cada exemplo Xi de um conjunto de dados a um único rótulo categórico (Yi), denominado classe.** A classificação pode ser utilizada quando é o objetivo é prever a classe ou categoria de um determinado exemplo com base em suas características ou atributos. Fluxo:

 1. A partir de uma base de dados rotulada , gera-se dois subconjuntos: base de treino (70% dos dados) e base de teste (30% dos dados)
 2. É retirada da base de treino os rótulos (ou as categorias que deseja-se descobrir) e é submetida ao classificador para o treinamento do modelo que é calibrado de acordo com os dados apresentados
 3. Apresentam-se os exemplos da base de teste para o modelo que deverá realizar predições de suas classes
 4. Compara-se as classes preditas com as classes verdadeiras da base de teste, irá se medir a qualidade do modelo, isto é, sua habilidade em classificar corretamente exemplos nao vistos durante o treinamento 

Observação: o classificador é o algoritmo usado para predição

*Holdout* - (separação do dataset em bases de treino e teste): o modelo é construído com um conjunto de dados e seu erro de generalização é avaliado com outro, não utilizado para o treinamento.
*overfitting* - quando o classificador se ajustou em excesso ao conjunto de treinamento
*underfitting* - o classificador se ajustou pouco ao conjunto de treinamento, não sendo adequado para realizar predições no conjunto de teste.

Para evitar problemas com overliftting e underfitting, temos:
 * **Dilema bias x variância**, em que: **viés (Bias)** representa a simplificação feita pelo modelo durante o treinamento, um modelo com alto viés pode subestimar a complexidade dos dados e, portanto, pode não capturar adequadamente os padrões nos dados; e **variância** representa a sensibilidade do modelo às flutuações nos dados de treinamento, um modelo com alta variância é muito sensível aos dados de treinamento específicos e pode não generalizar bem para novos dados. **O objetivo é encontrar um ponto de equilíbrio onde o modelo tenha baixo viés e baixa variância**
 * **Validação cruzada**, que é uma técnica usada para avaliar a capacidade de generalização de um modelo de aprendizado de máquina. Em vez de simplesmente dividir os dados em conjuntos de treinamento e teste uma única vez, a validação cruzada divide os dados em k partes (folds), treina o modelo em k-1 partes e avalia o desempenho no fold restante. Isso é repetido k vezes, cada vez com um fold diferente reservado para teste. O desempenho médio do modelo em todos os folds de teste é então calculado

#### 3.1.1 Métricas de Avaliação
A cada problema, os algoritmos disponíveis devem ser experimentados, com diversas configurações, para identificar os que obtêm melhor desempenho. Para isso, precisamos utilizar métricas de avaliação apropriadas. Existem diversas medidas para estimar o desempenho de um modelo de aprendizado supervisionado, ou seja, para avaliar o modelo. 
Para calcular as principais métricas de avaliação, é necessário calcular **matriz de confusão**, em que mostra a contagem de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos produzidos pelo modelo. A partir dessa matriz, várias métricas de avaliação, como precisão, recall, F1-score, entre outras, podem ser calculadas para entender o desempenho do modelo em diferentes cenários de classificação.

##### 3.1.1.1 Acurácia
Mede a proporção de todas as previsões corretas do modelo em relação ao número total de previsões
Acurácia = (Verdadeiros Positivos + Verdadeiros Negativos) / (Verdadeiros Positivos + Falsos Positivos + Falsos Negativos + Verdadeiros Negativos)

##### 3.1.1.2 Precisão
Proporção de todas as previsões corretas feitas pelo modelo em relação ao número total de previsões
Precisão = (VP + VN) / (VP + VN + FP + FN)

##### 3.1.1.3 Recall
Mede a capacidade do modelo de encontrar todos os casos positivos.
Recall = VP / (VP + FN)

##### 3.1.1.4 ROC
A curva ROC (receiver operating characteristic) contrasta os benefícios de uma classificação correta (TVP, sensibilidade ou recall) e o custo de uma classificação incorreta (TFP ou 1-especificidade), que varia entre 0 (predições 100% incorretas) e 1 (predições 100% corretas), em que:
 * Sensibilidade = TVP = Recall
 * Especificidade = 1 - Especificidade = TFP = FP / (FP + VN)

#### 3.1.2 Algoritmos
Algoritmos de machine learning que podem ser usados para problemas de classificação.

##### 3.1.2.1 - KNN (k-nearest neighbours)
O algoritmo KNN (k-nearest neighbours, ou k-vizinhos mais próximos) é um algoritmo não paramétrico que não assume premissas sobre a distribuição dos dados. O KNN utiliza uma métrica de distância para encontrar as k instâncias mais semelhantes nos dados de treinamento para uma nova instância. Além disso, considera a classe mais comum entre os vizinhos como a predição da classe da nova instância.
Limitações: a performance de predição pode ser lenta em datasets grandes; é sensível a características irrelevantes, uma vez que todas as características contribuem para o cálculo da distância e, consequentemente, para a predição; e é necessário testar diferentes valores de k e a métrica de distância a utilizar.

##### 3.1.2.2 - Árvore de Decisão
A Árvore de Decisão é um dos modelos preditivos mais simples de ser interpretado, e é inspirada na forma como humanos tomam decisões. Uma de suas principais vantagens é apresentar a informação visualmente, de uma forma fácil de entender. Há diferentes algoritmos para a elaboração de uma Árvore de Decisão, como:
 * ID3
 * CTree
 * C4.5
 * C5.0
 * CART

##### 3.1.2.3 - Naive Bayes
O Naive Bayes, ou Bayes Ingênuo, é um classificador genérico e de aprendizado dinâmico. É um dos métodos mais utilizados para classificação – especialmente em aplicações de text mining, previsões em tempo real e/ou sistemas embarcados –, por ser rápido computacionalmente e só necessitar de um pequeno número de dados de treinamento. Ele é especialmente adequado quando o problema tem um grande número de atributos (características), e determina a probabilidade de um exemplo pertencer a uma determinada classe. Esse método é chamado de ingênuo (naive, em inglês) porque desconsidera completamente a correlação entre os atributos (características), tratando cada um de forma independente. Além disso, o nome contém a palavra Bayes por ser baseado no teorema de Bayes, que determina a probabilidade de um evento com base em um conhecimento prévio (a priori) que pode estar relacionado a ele.

##### 3.1.2.4 - SVM
O SVM (support vector machine, ou máquina de vetor de suporte) é um dos algoritmos mais efetivos para classificação. Pode ser aplicado em dados lineares ou não lineares.




































