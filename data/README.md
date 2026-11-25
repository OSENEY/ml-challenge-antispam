# SMS Spam Collection

Tiago Almeida e José Hidalgo reuniram mensagens de SMS em inglês para pesquisa sobre spam. A UCI distribui a base sob licença CC BY 4.0. Consulta em 25/11/2026. [Página do conjunto de dados](https://doi.org/10.24432/C5CC84)

Na página da UCI, clique em **Download**, extraia o ZIP e copie o arquivo sem extensão `SMSSpamCollection` para esta pasta. O treinamento procura `data/SMSSpamCollection`.

O arquivo tem cerca de 478 kB e contém 5.574 registros em UTF-8, sem cabeçalho. Cada linha contém dois campos separados por uma tabulação.

| Campo | Tipo e uso |
|---|---|
| Rótulo | Categoria a prever. `ham` identifica uma mensagem legítima e `spam` identifica uma mensagem indesejada. |
| Mensagem | Texto do SMS. É a única entrada que o modelo recebe para prever a categoria. |
