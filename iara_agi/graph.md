Modelo Líquido Ternário — princípios naturais
hardware de padaria · inteligência de universo
Pesos ternários {-1, 0, +1} — BitLinear
i8 comprimido em 2 bits · substitui f32 · operações viram XOR/POPCNT
memória 16x menor · multiplicação vira soma/subtração de inteiros
KAN — Kolmogorov-Arnold Networks
funções B-spline nas arestas, não nos nós · aprende a forma da função
mesma acurácia com 100x menos parâmetros que MLP · interpretável
Superposição de ondas — mecânica quântica
token = ψ(amplitude, fase) · significado = interferência entre tokens
colapso da função de onda = escolha do próximo token
Ressonância estocástica — neurociência
ruído controlado melhora detecção de sinal · cérebro usa isso nativamente
dropout ternário = ruído útil · calibra temperatura de amostragem
Princípio da ação mínima — Lagrange / Fermat
o caminho natural é o de menor custo energético
ILPA busca o caminho semântico de menor "tensão de fase"
Redes livre de escala — Barabási · biologia
poucos hubs conectam tudo · lei de potência no vocabulário
tokens raros ligam a hubs comuns · grafo esparso = cache eficiente
Máxima entropia — Jaynes · termodinâmica
distribuição mais honesta dado o que se sabe
loss = KL-divergência ternária · gradiente em {-1,0,+1} apenas
Plasticidade hebbiana — neurociência
"neurônios que disparam juntos, se ligam" · aprendizado local
sem backprop global · cada camada KAN aprende sozinha · online
Resultado: modelo que roda em CPU comum
2-bit pesos · KAN esparso · ondas ternárias · busca por ressonância · aprendizado local
treinamento: sem GPU obrigatória · inferência: kilobytes por camada