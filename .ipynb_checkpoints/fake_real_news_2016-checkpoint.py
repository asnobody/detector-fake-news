import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Criar conjunto de dados
data = {
    'titulo': [
        # Notícias Falsas (0)
        "Papa Francisco Apoia Donald Trump para Presidente",
        "Hillary Clinton Gerencia Rede Secreta de Tráfico de Crianças",
        "Agente do FBI Encontrado Morto Após Expor Corrupção de Clinton",
        "NASA Confirma que Terra Ficará Escura por 15 Dias em Novembro",
        "Obama Assina Ordem Executiva Proibindo o Juramento de Fidelidade",
        "KFC Usa Galinhas Geneticamente Modificadas Sem Cabeça",
        "Coca-Cola Recolhe Garrafas Contendo Sangue Humano",
        "Líder do ISIS Se Converte ao Cristianismo Após Milagre",
        "Trump Promete Construir Muro ao Redor da Califórnia",
        "Campanha de Clinton Pagou US$10M a Beyoncé por Apoio",
        "Alienígenas Aterrissam em Nevada e Apoiam Trump",
        "Homem da Flórida Come o Próprio Rosto Após Usar Nova Droga",
        "Putin Alerta América: Vote em Trump ou Enfrente Guerra Nuclear",
        "Michelle Obama Grávida do Terceiro Filho",
        "Disney World Fecha Permanentemente Devido ao Vírus Zika",
        "Hillary Clinton Tem Mal de Parkinson, Dizem Médicos",
        "Trump Promete Proibir Todos os Muçulmanos de Entrar nos EUA",
        "Máquina do Tempo Secreta Encontrada na Área 51",
        "Rainha Elizabeth Declara Apoio ao Brexit",
        "E-mails de Clinton Revelam Plano para Iniciar a Terceira Guerra Mundial",
        # Notícias Verdadeiras (1)
        "Economia dos EUA Adiciona  Gabriella",
        "Acordo Climático de Paris Entra em Vigor",
        "Forças Sírias Lançam Ofensiva em Aleppo",
        "Federal Reserve Mantém Taxas de Juros Inalteradas",
        "Tribunal do Reino Unido Decide que Brexit Precisa de Aprovação Parlamentar",
        "Fidel Castro, de Cuba, Morre aos 90 Anos",
        "Presidente da Coreia do Sul Enfrenta Voto de Impeachment",
        "Primeiro-Ministro da Itália Renuncia Após Referendo",
        "Colômbia Assina Acordo de Paz com Rebeldes FARC",
        "Sonda Cassini da NASA Inicia Órbita Final em Saturno",
        "Alemanha Proíbe Burca em Locais Públicos",
        "China Lança Novo Porta-Aviões",
        "EUA Aprovam US$3,8 Bi em Ajuda Militar a Israel",
        "Turquia Detém Milhares Após Tentativa de Golpe",
        "Senado do Brasil Aprova Impeachment de Dilma Rousseff",
        "Vazamento dos Panama Papers Expõe Evasão Fiscal Global",
        "Vírus Zika Se Espalha Rapidamente na América Latina",
        "Apple Lança iPhone 7 Sem Conector de Fone de Ouvido",
        "Coreia do Norte Realiza Quinto Teste Nuclear",
        "Theresa May Torna-se Primeira-Ministra do Reino Unido"
    ],
    'texto': [
        # Trechos de Notícias Falsas
        "Em uma declaração chocante, o Papa Francisco anunciou seu apoio a Donald Trump, dizendo que ele é a ‘escolha de Deus’ para a América. O Vaticano confirmou o apoio…",
        "E-mails vazados revelam que Hillary Clinton opera uma rede de tráfico de crianças em uma pizzaria em Washington, D.C., afirmam fontes…",
        "Um agente do FBI que investigava os e-mails de Clinton foi encontrado morto em um aparente assassinato-suicídio, levantando suspeitas de acobertamento…",
        "A NASA anunciou que um evento celestial raro mergulhará a Terra na escuridão por 15 dias a partir de 15 de novembro…",
        "O presidente Obama assinou uma ordem executiva proibindo o Juramento de Fidelidade nas escolas, alegando que é divisivo…",
        "O KFC tem usado galinhas geneticamente modificadas sem cabeça para reduzir custos, revelou um denunciante…",
        "A Coca-Cola recolheu milhares de garrafas após testes encontrarem vestígios de sangue humano em seus produtos…",
        "O líder do ISIS, Abu Bakr al-Baghdadi, converteu-se ao cristianismo após testemunhar um milagre divino, dizem fontes…",
        "Donald Trump prometeu construir um muro ao redor da Califórnia para ‘manter os liberais fora’ se eleito…",
        "A campanha de Hillary Clinton pagou US$10 milhões a Beyoncé por um show de apoio, mostram documentos vazados…",
        "Um OVNI pousou em Nevada, e extraterrestres anunciaram seu apoio à campanha de Donald Trump…",
        "Um homem da Flórida foi preso após comer o próprio rosto, supostamente sob a influência de uma nova droga sintética…",
        "O presidente russo Vladimir Putin alertou que uma vitória de Clinton levaria à guerra nuclear com a Rússia…",
        "Fontes afirmam que Michelle Obama está grávida de seu terceiro filho aos 52 anos, chocando a nação…",
        "A Disney World anunciou seu fechamento permanente devido ao surto do vírus Zika na Flórida…",
        "Especialistas médicos afirmam que Hillary Clinton está escondendo um diagnóstico de Parkinson, com base em suas aparições públicas…",
        "A campanha de Trump anunciou uma proibição total de muçulmanos entrando nos EUA, efetiva imediatamente se ele vencer…",
        "Documentos desclassificados revelam que uma máquina do tempo funcional foi descoberta na Área 51…",
        "A rainha Elizabeth surpreendeu o mundo ao apoiar publicamente o Brexit, chamando-o de ‘novo amanhecer’ para a Grã-Bretanha…",
        "Os e-mails de Clinton, obtidos pelo WikiLeaks, revelam um plano secreto para provocar a Terceira Guerra Mundial com a Rússia…",
        # Trechos de Notícias Verdadeiras
        "A economia dos EUA adicionou 161.000 empregos em outubro, com o desemprego caindo para 4,9%, informou o Departamento do Trabalho…",
        "O acordo climático de Paris, assinado por 196 países, entrou oficialmente em vigor em 4 de novembro…",
        "Forças do governo sírio, apoiadas pela Rússia, lançaram uma grande ofensiva para recapturar Aleppo controlada por rebeldes…",
        "O Federal Reserve decidiu manter as taxas de juros inalteradas, citando crescimento econômico estável…",
        "Um tribunal do Reino Unido decidiu que o Parlamento deve aprovar o plano do governo para iniciar o Brexit…",
        "Fidel Castro, líder revolucionário de Cuba, morreu aos 90 anos, anunciou a mídia estatal…",
        "A presidente sul-coreana Park Geun-hye enfrenta um voto de impeachment em meio a um escândalo de corrupção…",
        "O primeiro-ministro italiano Matteo Renzi renunciou após perder um referendo sobre reforma constitucional…",
        "O governo da Colômbia e os rebeldes FARC assinaram um acordo de paz revisado para encerrar um conflito de 52 anos…",
        "A sonda Cassini da NASA começou suas órbitas finais ao redor de Saturno, preparando-se para um pouso forçado…",
        "O parlamento da Alemanha votou pela proibição da burca em locais públicos, citando preocupações de segurança…",
        "A China lançou seu segundo porta-aviões, um passo para expandir seu poder naval…",
        "O Congresso dos EUA aprovou um pacote de ajuda militar de US$3,8 bilhões para Israel ao longo de 10 anos…",
        "A Turquia deteve milhares de soldados, juízes e acadêmicos após uma tentativa de golpe fracassada…",
        "O Senado do Brasil votou pelo impeachment da presidente Dilma Rousseff por má gestão fiscal…",
        "O vazamento dos Panama Papers expôs contas offshore usadas por elites globais para evadir impostos…",
        "O vírus Zika, ligado a defeitos congênitos, espalhou-se rapidamente pela América Latina, relatou a OMS…",
        "A Apple lançou o iPhone 7, removendo o conector de fone de ouvido em uma decisão controversa…",
        "A Coreia do Norte realizou seu quinto teste nuclear, desafiando sanções internacionais…",
        "Theresa May tornou-se primeira-ministra do Reino Unido após a renúncia de David Cameron pós-Brexit…",
    ],
    'data': [
        # Datas de Notícias Falsas
        "2016-10-15",
        "2016-11-02",
        "2016-09-20",
        "2016-08-10",
        "2016-07-25",
        "2016-06-30",
        "2016-05-12",
        "2016-04-18",
        "2016-03-22",
        "2016-02-28",
        "2016-01-15",
        "2016-11-05",
        "2016-10-01",
        "2016-09-08",
        "2016-08-14",
        "2016-07-10",
        "2016-06-05",
        "2016-05-01",
        "2016-04-07",
        "2016-03-03",
        # Datas de Notícias Verdadeiras
        "2016-11-04",
        "2016-11-04",
        "2016-10-27",
        "2016-11-02",
        "2016-11-03",
        "2016-11-25",
        "2016-12-09",
        "2016-12-04",
        "2016-11-24",
        "2016-12-01",
        "2016-12-15",
        "2016-04-20",
        "2016-09-14",
        "2016-07-16",
        "2016-08-31",
        "2016-04-03",
        "2016-02-11",
        "2016-09-07",
        "2016-09-09",
        "2016-07-13",
    ],
    'classe': [
        # Rótulos de Notícias Falsas
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Rótulos de Notícias Verdadeiras
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]
}

# Criar DataFrame
df = pd.DataFrame(data)

# Salvar em CSV para uso no seu projeto
df.to_csv("noticias_falsas_verdadeiras_2016.csv", index=False)

# Visualização 1: Distribuição de Classes
plt.figure(figsize=(8, 6))
sns.countplot(x='classe', data=df)
plt.title('Distribuição de Notícias Falsas (0) e Verdadeiras (1)')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Falsa', 'Verdadeira'])
plt.show()

# Visualização 2: Nuvens de Palavras
texto_falso = ' '.join(df[df['classe'] == 0]['texto'])
texto_verdadeiro = ' '.join(df[df['classe'] == 1]['texto'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
wordcloud_falso = WordCloud(width=800, height=400, background_color='white').generate(texto_falso)
plt.imshow(wordcloud_falso, interpolation='bilinear')
plt.axis('off')
plt.title('Nuvem de Palavras - Notícias Falsas')

plt.subplot(1, 2, 2)
wordcloud_verdadeiro = WordCloud(width=800, height=400, background_color='white').generate(texto_verdadeiro)
plt.imshow(wordcloud_verdadeiro, interpolation='bilinear')
plt.axis('off')
plt.title('Nuvem de Palavras - Notícias Verdadeiras')
plt.show()