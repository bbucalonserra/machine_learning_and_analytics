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
  * Classificação - Usada para prever a classe ou categoria de um objeto. Exemplo: "Deve-se conceder ou não crédito para um cliente de um banco?", neste caso a variável a ser predita é categórica, pois a resposta deverá ser sim ou não
  * Regressão - Usada para prever um valor numérico ou contínuo. Exemplo: "Deve-se conceder qual valor de crédito a um cliente de um banco?", neste caso, a variável a ser predita é numérica (contínua ou disreta)
Além disso, é comum que particionar os dados de entrada (rotulados) em dois conjuntos:
  * De treinamento - Servirá para construir o modelo
  * De teste - Também chamado na literatura de conjunto de validação, servirá para verificar como o modelo se comportaria em dados não vistos, de forma que possamos ajustá-lo, se necessário, para a construção final do modelo a ser aplicado a novos dados em que ainda não conhecemos a saída esperada.


