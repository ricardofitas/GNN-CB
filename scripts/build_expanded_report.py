"""Generate the current Portuguese report from recorded expansion outputs."""
from pathlib import Path
import json
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/GNN_CB_Approach_Report.pdf'
S=json.loads((ROOT/'reports/expansion_summary.json').read_text())
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
navy=colors.HexColor('#17334c');blue=colors.HexColor('#216e98');muted=colors.HexColor('#566773')
styles=getSampleStyleSheet()
for name,size,leading,after in [('P',9.5,14,8),('Small',8,11,7),('H',15,20,12),('Hero',29,34,12),('Code',8,12,9)]:
 styles.add(ParagraphStyle(name='CB'+name,fontName='Bold' if name in ['H','Hero'] else 'Body',fontSize=size,leading=leading,textColor=navy,spaceAfter=after))
story=[]
def p(t,style='P'):story.append(Paragraph(t,styles['CB'+style]))
def h(t):p(t,'H')
def page(t):story.append(PageBreak());h(t)
def table(rows,widths):
 cells=[[Paragraph(str(c),styles['CBSmall']) for c in row] for row in rows]
 t=Table(cells,colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,blue),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d8e0e7')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,8))
def pic(name,width=490,maxheight=365):
 path=ROOT/'reports/figures'/name
 if not path.exists():return
 img=Image(str(path));ratio=min(width/img.imageWidth,maxheight/img.imageHeight)
 img.drawWidth=img.imageWidth*ratio;img.drawHeight=img.imageHeight*ratio;story.append(img);story.append(Spacer(1,8))
def number(x):return f'{x:,}'.replace(',',' ')
def pct(x):return f'{100*x:.2f}%'

p('GNN-CB','Hero')
h('Mais imagens, grafos curados e teste entre ensaios')
p('Relatório atualizado | 9 de outubro de 2026 | Ricardo Fitas','Small')
p('A pesquisa foi alargada aos repositórios públicos e privados disponíveis. Foram recuperados três ensaios com os seus grafos e tabelas originais e quatro vídeos completos. A implementação passou a incluir extração autónoma de nós, arestas e trajetórias a partir de imagens, com máscaras e registo de decisões de qualidade.')
table([['Dados supervisionados principais','Imagens de vídeo processadas','Estados de compressão'],['<b>5 ensaios</b>','<b>8 192 imagens / 4 vídeos</b>','<b>512 estados / 19 252 observações</b>']],[164,164,162])
p(S['headline'])
p(S['augmentation_headline'])
pic('expanded_source_graphs.png',490,250)
p('Três novos ensaios: imagens de referência e grafos originais registados sobre o cartão. As arestas desenhadas são as ligações iniciais; não representam uma malha de elementos finitos.','Small')
h('O que mudou')
p('O conjunto principal aumentou de 2 para 5 espécimes: 841 imagens amostradas, 220 nós iniciais e 321 arestas não orientadas. Os 8 192 frames dos quatro vídeos foram também segmentados e convertidos em grafos. As fotografias estáticas e as figuras de otimização foram classificadas separadamente. A contagem de ficheiros nunca substitui a contagem de ensaios independentes.')
p('O alvo continua a ser o deslocamento 2D dos nós condicionado pela geometria inicial e por uma fase retrospetiva do ensaio. Os valores de referência vêm de tracking de imagem e contêm incerteza; não são medições independentes por DIC nem soluções mecânicas exatas.','Small')

page('1. Inventário e proveniência')
table([['Origem / categoria','Dados encontrados','Uso efetivo'],['2 arquivos públicos do estudo','410 imagens de câmara','2 ensaios de referência, preservados'],['uncertain_cb: Exp3 / Exp4 / Exp5','3 792 frames originais; 431 amostrados','431 JPG verificados por SHA; 3 novos ensaios'],['ML_Corrugated_Boards','17 partes RAR; 4 vídeos','8 192 / 8 192 frames descodificados e processados'],['Opt_CB: fotografias estáticas','22 fotografias; 44 tentativas de extração','Grafos candidatos inspecionados e rejeitados'],['8 repositórios públicos auditados','60 232 caminhos de imagens/vídeos; 28 258 blobs únicos','Maioritariamente figuras e renders de otimização']],[180,155,155])
p('As 431 imagens amostradas são derivadas das 3 792 imagens originais: não se somam como novos instantes independentes. Os espelhos Sample_videos/Exp3-5 têm os mesmos SHAs das pastas D12, D14 e ID03. Versões rate3/rate10, recortes e reextrações mantêm a mesma identidade física.')
p('Foram auditadas ou convertidas em grafos 9 033 imagens de câmara nos nove registos identificados (410 + 431 + 8 192), além das 22 fotografias estáticas. Destas, 8 623 imagens de câmara são adicionais relativamente à primeira entrega. Os 9 registos não equivalem a 9 ensaios admitidos no treino principal.')
h('Uma correção importante de contagem')
p('Os 72 casos / 14 720 frames presentes em Opt_CBv2 são imagens Iteration_*.png de otimização. Não representam 72 ensaios físicos nem 14 720 fotografias de compressão. Foram excluídos da aprendizagem de deslocamentos experimentais.')
h('Identidades e curvas mecânicas')
p('Exp3 corresponde a D12 e à folha E04 / Specimen 1.7; Exp4 corresponde a D14 e a E03 / Specimen 1.6; Exp5 corresponde a ID03 e a E05 / Specimen 1.5. Os vídeos Probe1_2, 1_8, 1_9 e 1_10 pertencem à coleção ML2023. As curvas mecânicas são distintas das anteriores; a numeração Probe1.10 não é uma identidade global. A sincronização força-frame não foi demonstrada, pelo que não foram inventados alvos de força ou tensão.')
p('Fontes fixadas: Graphs_Analysis_Corrugated_Boards @ 16eed74ff025e0e0f40f4f5d81c2b71af58b93b5; uncertain_cb @ 1df856a4f2c71378744d4a73e3cd958f6f6d7a71. O commit da coleção ML2023, os SHAs das 17 partes e os hashes dos vídeos estão em reports/video_source_audit.json.','Small')

page('2. Das imagens aos nós e às arestas')
p('A nova pipeline usa escala de cinzentos, suavização gaussiana, limiarização, filtragem de componentes e mediana. A esqueletização preserva a estrutura do material. Clusters de junção e extremidades tornam-se nós; caminhos de grau dois são contraídos em arestas, conservando as polilinhas. Pequenos ramos espúrios são filtrados; não se criam pontes para fechar material ausente.')
p('O tracking aplica correspondência um-para-um por distância e grau, com opção explícita de ficar sem correspondência. Alternativas ambíguas são recusadas. Nós perdidos durante mais do que a tolerância não são recuperados por uma associação arbitrária. Frames com cobertura insuficiente ficam fora da exportação supervisionada.')
table([['Controlo','Aplicação'],['Geometria e qualidade','ROI, componentes, conectividade, ramos e sobreposição visual em imagens de várias fases.'],['Trajetórias','Distância máxima, rejeição de ambiguidades, máscaras por nó/frame, cobertura e sobrevivência dos nós.'],['Proveniência','Configuração, hashes das imagens, grafos por frame, polilinhas, coordenadas e razões de admissão/exclusão.'],['Separação experimental','Todos os recortes, taxas e reextrações de um ensaio partilham a mesma identidade física.']],[120,370])
pic('automatic_video_graphs.png',490,285)
p(S['automatic_caption'],'Small')
p('A reextração dos dois casos de referência produziu 48 nós em cada caso, em vez dos 40/44 da simplificação original. Em Probe2_7, todos os 44 nós originais ficam a menos de 5 px de um nó extraído; em Probe1_22, 39/40. As diferenças de fronteira foram conservadas e documentadas, sem forçar a topologia a coincidir com os rótulos antigos.','Small')

page('3. Admissão, cobertura e limitações dos alvos')
rows=[['Espécime','Nós / arestas','Frames de compressão','Observações / cobertura']]
for d in S['specimens']:
 rows.append([d['specimen_id'],f"{d['n_nodes']} / {d['n_undirected_edges']}",d['compression_frames'],f"{number(d['compression_observed_node_frames'])} / {100*d['compression_coverage']:.1f}%"])
table(rows,[105,100,120,165])
p('Os três novos arquivos passaram a verificação exata entre geometria inicial, posições no grafo de cada frame e observações das tabelas. Foram mascaradas 447 correspondências com colisão de coordenadas, das quais 227 estavam nas janelas de compressão. Não se misturou a numeração dos nós de exportações incompatíveis e não se interpolaram alvos ausentes.')
p('A consistência entre ficheiros não prova identidade física após encurvadura ou oclusão. Existem reidentificações de origem após intervalos longos sem observação. A análise de sensibilidade retém apenas a parte das trajetórias anterior à primeira lacuna com mais de dois frames intermédios sem observação, isto é, separação entre frames observados superior a três.')
p('<b>Cobertura seletiva:</b> Exp3 tem 15 dos 46 nós sem observação durante toda a compressão. Exp5 tem cerca de 50% das observações possíveis e nenhuma trajetória completa em toda a janela. Os erros descritos neste relatório aplicam-se aos nós observados; não validam a geometria completa.')
h('Decisão sobre os quatro vídeos')
for line in S['automatic_decisions']:p(line,'Small')
h('Fotografias estáticas')
p('Foram extraídos 44 grafos candidatos das 22 fotografias, usando dois métodos de limiarização. A inspeção revelou grupos de tiras, perspetiva, fundos e iluminação que produziam ligações falsas, além de ausência de uma referência antes da deformação. Nenhuma foi usada para supervisionar deslocamentos. Os candidatos e as razões de rejeição foram guardados para inspeção.')

page('4. Modelo e protocolo de avaliação')
p('A GNN mantém três blocos de passagem de mensagens, dimensão latente 32, agregação pela média e atributos de aresta dx, dy e comprimento inicial. Recebe a geometria inicial e s, s²; produz deslocamentos x/y normalizados pela altura inicial. Uma base afim de geometria e fase é ajustada apenas no treino; a rede aprende a correção. A MLP de comparação usa entradas e dimensão semelhantes, mas não comunica pelas arestas.')
table([['Elemento','Protocolo fixado'],['Grupos','5 folds: 1 espécime de teste, o seguinte na ordem fixada para validação, 3 para ajuste.'],['Fases de treino','Toda a janela 0-1 dos espécimes de ajuste. A validação utiliza um espécime inteiro separado.'],['Otimização','Seeds 11, 29, 47; AdamW; lr 0,002; até 300 épocas; paragem antecipada pela validação; época zero elegível.'],['Ponderação','Mesmo peso por espécime físico, independentemente de frames, nós ou reextrações.'],['Métrica principal','RMSE de coordenadas / altura inicial do grafo, com média igual entre espécimes.'],['Métrica secundária','RMSE em píxeis; a resolução muda de câmara para câmara.']],[115,375])
p('As alturas iniciais são 39, 41, 80, 82 e 133 px. Um erro de 10 px não representa a mesma fração da geometria em todos os ensaios. A fase é calculada a partir dos extremos anotados do intervalo observado; não é deformação imposta medida nem uma variável automaticamente disponível em tempo real.')
p('Este benchmark difere da primeira entrega: ali, a validação era o último segmento temporal do único espécime de treino. Por isso, o RMSE agregado atual não deve ser comparado diretamente com 6,35 px como se a população e o protocolo fossem iguais.')
h('Resultados em espécimes reservados')
table([['Modelo','RMSE / altura inicial','RMSE em píxeis']]+S['aggregate_rows'],[210,145,135])
p(S['result_interpretation'])

page('5. Resultados por ensaio e robustez')
pic('expanded_benchmark.png',490,245)
table([['Espécime de teste','GNN (% altura)','Afim (% altura)','Cobertura']]+S['fold_rows'],[150,120,120,100])
p('As barras e os valores ± descrevem variação entre seeds. Não são intervalos de confiança de generalização entre materiais. Cada teste usa um espécime inteiramente reservado; os seus alvos não escolhem épocas nem ajustam os pesos.','Small')
h('Sensibilidade às lacunas de tracking')
p(S['continuity_text'])
h('Modelo anterior em três ensaios novos')
p(S['external_text'])
p('Este teste externo conserva o checkpoint anterior sem qualquer ajuste. A comparação com o novo treino descreve os resultados nos três ensaios, mas não isola causalmente o efeito de mais dados: o conjunto de treino e o protocolo de seleção também mudam.','Small')

page('6. Vídeos: teste adicional e resultado da curadoria')
for text in S['automatic_evaluation']:p(text)
pic('automatic_video_coverage.png',490,230)
p('O arquivo preserva os grafos por frame, incluindo os rejeitados para treino. A perda de nós é informação sobre a qualidade do processo; não foi preenchida com previsões da própria GNN para as fazer passar por medições.','Small')
page('7. Limites e validação futura')
p('A rede não prevê forças, tensões, materiais, dano ou contacto. A topologia inicial permanece fixa na previsão. As correspondências extraídas são rótulos automáticos e partilham erros com o método de imagem. A diversidade experimental continua pequena. Ensaios com baixa cobertura são avaliados apenas no subconjunto visível, e os registos truncados não oferecem automaticamente um fim de compressão comparável.')
p('A próxima validação com maior valor é uma campanha com identidades físicas inequívocas, calibração espacial independente, tempo de câmara sincronizado com a máquina e uma pequena amostra de nós anotada manualmente. Essa amostra permite medir erros de identidade e de posição do próprio tracking antes de avaliar a GNN como modelo mecânico.')

pic('static_photo_curation.jpg',490,340)
p('Fotografias estáticas: sobreposições inspecionadas e rejeitadas para supervisão de deslocamentos. Fundos, perspetiva e vários provetes na mesma imagem produzem grafos sem correspondência física fiável. A auditoria contém as 44 tentativas de extração e as razões de exclusão.', 'Small')

page('8. Entrega, reprodução e verificações')
table([['Ficheiro / diretório','Conteúdo'],['gnn_cb/image_graph.py','Segmentação, esqueletização, nós/arestas e tracking com máscaras.'],['scripts/curate_images.py','Curadoria por sequência/manifesto; configuração e fases explícitas.'],['scripts/extract_rar_videos.py','Extração multi-volume e descodificação verificadas.'],['data/expanded/','5 datasets coerentes, metadados, hashes e imagens de referência.'],['data/automatic/','4 vídeos: grafos, trajetórias, máscaras e sobreposições.'],['gnn_cb/train_expanded.py','Treino com separação por espécime, métricas e sensibilidades.'],['results/expanded/','Checkpoints, protocolo, previsões e resultados gravados.'],['reports/','Inventários, decisões de qualidade, figuras e este relatório.']],[200,290])
p('Executar a avaliação principal com os dados preparados incluídos:','Small')
p('python -m gnn_cb.train_expanded --data data/expanded<br/>--output results/expanded/five_specimens<br/>--epochs 300 --patience 75 --seeds 11 29 47<br/>python -m unittest discover -s tests -v','Code')
p('Juntar as linhas do comando de treino. README.md documenta os comandos completos, as dependências e a origem dos ficheiros privados. Inferência usa apenas a geometria inicial e a fase; não precisa das trajetórias alvo. O checkpoint final ajustado a todos os espécimes é demonstrativo e não tem um teste independente próprio.','Small')
p(S['verification_text'])
h('Referências')
p('Belfekih, T.; Fitas, R.; Schaffrath, H.-J.; Schabel, S. Graph-Based Analysis for the Characterization of Corrugated Board Compression. Materials 17(24), 6083 (2024). DOI: 10.3390/ma17246083.','Small')
p('Pfaff, T. et al. Learning Mesh-Based Simulation with Graph Networks. ICLR 2021. arXiv:2010.03409. A arquitetura inspira-se em passagem de mensagens; esta implementação é um regressor condicionado, não uma reprodução de um solver dinâmico.','Small')
p('Código e dados: repositórios de Ricardo Fitas identificados nos manifestos. A licença MIT dos dois arquivos públicos está preservada em licenses/source-data-MIT.txt. Os novos materiais privados mantêm a proveniência e o acesso do projeto privado GNN-CB.','Small')
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#d8e0e7'));c.line(48,44,547,44);c.setFont('Body',7);c.setFillColor(muted)
 c.drawString(48,30,'GNN-CB | Expansão de dados, curadoria e avaliação | 09.10.2026');c.drawRightString(547,30,str(doc.page))
 c.setTitle('GNN-CB | Dados expandidos, curadoria e avaliação');c.setAuthor('Prepared for Ricardo Fitas')
SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=58).build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
