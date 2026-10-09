"""Build the Portuguese approach report from recorded benchmark outputs."""
from pathlib import Path
import json, math
import numpy as np
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/GNN_CB_Approach_Report.pdf'
d=json.loads((ROOT/'results/loso/metrics.json').read_text())
m=json.loads((ROOT/'data/processed/manifest.json').read_text())
a=d['aggregate'];gnn=a['gnn']['macro_rmse_px_mean_over_seeds'];base=a['phase_affine']['macro_rmse_px'];improve=100*(1-gnn/base)
font_dir=Path('/usr/share/fonts/truetype/dejavu')
if (font_dir/'DejaVuSans.ttf').exists():
 for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
 pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
else:
 pdfmetrics.registerFontFamily('Body',normal='Helvetica',bold='Helvetica-Bold',italic='Helvetica-Oblique',boldItalic='Helvetica-BoldOblique')
 # Standard PDF fonts provide a portable fallback when DejaVu is unavailable.
 from reportlab.pdfbase.pdfmetrics import Font
 pdfmetrics.registerFont(Font('Body','Helvetica','WinAnsiEncoding'));pdfmetrics.registerFont(Font('Bold','Helvetica-Bold','WinAnsiEncoding'))
navy=colors.HexColor('#18324d');blue=colors.HexColor('#246b98');gold=colors.HexColor('#b3965d');gray=colors.HexColor('#4d5b66')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='P',fontName='Body',fontSize=9.7,leading=14,textColor=navy,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallP',parent=styles['P'],fontSize=8.2,leading=11.5,textColor=gray,spaceAfter=6))
styles.add(ParagraphStyle(name='H',fontName='Bold',fontSize=22,leading=28,textColor=navy,spaceAfter=15))
styles.add(ParagraphStyle(name='H2x',fontName='Bold',fontSize=12,leading=16,textColor=blue,spaceBefore=7,spaceAfter=7))
styles.add(ParagraphStyle(name='Hero',fontName='Bold',fontSize=43,leading=49,textColor=navy,spaceAfter=12))
styles.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=7.4,leading=11,textColor=navy,backColor=colors.HexColor('#eef3f7'),borderPadding=8,spaceAfter=12))
story=[]
def p(s,style='P'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'H2x')
def page(title):story.append(PageBreak());p(title,'H')
def pic(name,width=490):
 path=ROOT/'reports/figures'/name
 from PIL import Image as PImage
 w,hh=PImage.open(path).size
 story.append(Image(str(path),width=width,height=width*hh/w));story.append(Spacer(1,6))
def table(rows,widths):
 cells=[[Paragraph(str(c),styles['SmallP']) for c in row] for row in rows]
 t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf3')),('TEXTCOLOR',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,blue),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d8e0e7')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,10))
def fmt(x,n=3):return f'{x:.{n}f}'
def avg(f,kind,key,block='compression_window'):return float(np.mean([r['test'][block][key] for r in f['models'][kind]]))

p('GNN-CB','Hero')
p('Uma GNN funcional para deformação de corrugated boards','H')
p('Relatório de abordagem, implementação e avaliação | 9 de outubro de 2026','SmallP')
p('Preparado para Ricardo Fitas. Implementação e treino executados sobre os dados originais da investigação de caracterização de cartão canelado por grafos [1,2].')
table([['Erro GNN','Melhoria face ao modelo afim','Dados independentes'],[f'<b>{gnn:.2f} px</b>',f'<b>{improve:.1f}%</b>','<b>2 espécimes</b>']],[150,200,140])
p('Foi implementada e treinada uma rede de passagem de mensagens que recebe a geometria inicial, as ligações do grafo e uma fase prescrita do protocolo de compressão. A saída é o deslocamento bidimensional de cada nó. O repositório inclui preparação dos dados, pesos treinados, inferência, resultados numéricos e testes de integridade.')
pic('image_graphs.png')
p('Figura 1. Imagens originais, nós e arestas da geometria inicial. As ligações provêm dos grafos da investigação; não foram substituídas por uma grelha ou por vizinhos artificiais. Fonte: repositório original [2], licença MIT.','SmallP')
h('O resultado que é defensável')
p('A GNN superou os dois modelos de comparação neste pequeno teste de transferência entre espécimes. A melhoria é sobretudo horizontal. O conjunto contém apenas duas realizações independentes e alvos derivados de rastreamento automático de imagem: o resultado constitui uma prova de conceito, ainda sem validação como modelo geral de materiais ou solver mecânico.')

page('1. Dados e rastreabilidade')
p('Foram auditados dois repositórios. Graph-CB contém exportações de cinco ensaios, incluindo cópias idênticas e versões com diferentes amostragens. Graphs_Analysis_Corrugated_Boards contém dois arquivos completos e internamente coerentes: imagens, topologia, tabelas de rastreamento e anotações de compressão.')
rows=[['Espécime','Nós / arestas','Frames usados','Cobertura']]
for s in m['specimens']:rows.append([s['specimen_id'],f"{s['n_nodes']} / {s['n_undirected_edges']}",f"{s['compression_window'][0]}–{s['compression_window'][1]}",f"{100*s['compression_coverage']:.3f}%"])
table(rows,[140,110,120,120])
p('Cada espécime tem 205 imagens de câmara e 205 imagens de visualização do grafo. A janela de compressão reúne 251 estados e 10 444 observações nó-frame válidas. Estes valores não equivalem a 251 ou 10 444 ensaios independentes.')
h('Preparação')
p('O importador fixa o commit de origem e verifica SHA-256 dos ZIPs. Confirma que as coordenadas iniciais dos 84 nós coincidem exatamente com as primeiras coordenadas das tabelas de rastreamento. As posições são convertidas de (linha, coluna) para (x, y), com x para a direita e y para baixo. A geometria e os deslocamentos são divididos pela altura inicial do grafo: 41 px ou 39 px.')
p('As observações em falta são excluídas por uma máscara explícita. Não há interpolação, preenchimento temporal nem criação de alvos. As arestas iniciais são usadas nos dois sentidos, mantendo a mesma topologia ao longo da previsão.')
pic('tracking_coverage.png',475)
p('Figura 2. Número de nós observados em cada fase. No final de Probe1_22, apenas 20 dos 40 nós têm observação; uma previsão desses nós ausentes não pode ser pontuada como correta.','SmallP')
h('Exclusões justificadas')
p('As versões recentes de Probe1_22 têm numeração diferente; não foram unidas aos grafos antigos. Exp3 tem trajetórias que começam tardiamente e não dispõe do arquivo completo correspondente. Exp4/Exp5 não têm geometria inicial e imagens/topologia suficientes nos ficheiros encontrados. As cópias em data_st não foram contadas novamente.')

page('2. Abordagem e arquitetura')
p('A tarefa escolhida usa os alvos mais diretamente sustentados pelos dados: deslocamentos dos nós rastreados. A deteção e a transformação original de imagens em grafos vêm do estudo de origem. O trabalho novo valida essa correspondência e aprende o mapa entre geometria inicial, fase e deformação.')
h('Entradas e saída')
p('Para cada nó: coordenadas iniciais centradas e normalizadas, fase s e s². Para cada aresta: deslocamento relativo inicial em x e y e comprimento. O codificador transforma estas entradas num estado latente de dimensão 32; três blocos trocam mensagens entre os nós ligados. Um descodificador produz duas componentes de deslocamento.')
p('Em cada bloco, uma MLP calcula mensagens a partir do estado dos dois nós e dos atributos da aresta. A média das mensagens recebidas atualiza o nó com uma ligação residual e LayerNorm. Esta operação é equivariante à permutação da numeração dos nós: renumerar o grafo renumera as saídas, sem alterar a previsão física associada.')
table([['Componente','Definição'],['Base afim','9 funções: [1, s, s²] × [1, x centrado, y centrado]. Coeficientes ajustados exclusivamente nos dados de treino.'],['GNN','Base afim + correção aprendida com mensagens nas arestas; 21 634 parâmetros treináveis.'],['MLP sem arestas','Mesma base, mesmas coordenadas/fase e saída; 22 570 parâmetros, sem comunicação entre nós.'],['Otimização','AdamW; learning rate 0,002; weight decay 0,001; limite de gradiente 5; até 600 épocas.'],['Loss','Erro quadrático médio das duas componentes, apenas em observações válidas. Escala residual calculada no treino.']],[125,365])
h('Porquê esta escolha')
p('Os grafos têm 40 e 44 nós e conectividade irregular. Uma GNN partilha as mesmas operações locais e aceita esses tamanhos sem fixar um vetor com a mesma numeração. A base afim oferece uma comparação simples e uma inicialização estável. A MLP de tamanho semelhante ajuda a avaliar se as relações do grafo acrescentam informação.')
h('O significado exato da fase')
p('s = (frame - início anotado)/(fim anotado - início anotado). Os extremos foram identificados no ensaio completo. Trata-se de informação retrospetiva do protocolo, não de deformação imposta medida, relógio sincronizado ou informação disponível automaticamente em tempo real. s = 0 é o início anotado da compressão, e não necessariamente o frame inicial de referência.')
p('A inspiração metodológica vem de modelos de simulação com redes de grafos [3]. Esta implementação é regressão condicionada e não reproduz um solver MeshGraphNets, não integra a dinâmica, nem impõe equilíbrio, contacto ou leis constitutivas.','SmallP')

page('3. Avaliação e resultados')
p('Foram executados dois testes leave-one-specimen-out: treinar em Probe1_22 e testar em Probe2_7; depois inverter. No espécime de treino, a fase 0–0,75 ajusta os parâmetros, (0,75; 0,80) fica excluída e 0,80–1,00 seleciona a época. A época zero também é candidata. Não se reajustam os modelos de benchmark na validação.')
p('A arquitetura e os hiperparâmetros foram fixados antes dos resultados dos espécimes de teste. Cada rede foi treinada com seeds 11, 29 e 47. Os alvos do espécime reservado não participam no treino nem na escolha da época. A parte final do teste combina transferência entre espécimes com extrapolação de fase para além do intervalo usado no ajuste.')
rows=[['Modelo','Teste Probe2_7','Teste Probe1_22','Média entre espécimes']]
for kind,label in [('phase_affine','Afim'),('mlp','MLP'),('gnn','GNN')]:
 vals=[f['baselines'][kind]['compression_window']['rmse_px'] if kind=='phase_affine' else avg(f,kind,'rmse_px') for f in d['folds']]
 rows.append([label,*[f'{v:.3f} px' for v in vals],f'{np.mean(vals):.3f} px'])
table(rows,[100,130,130,130])
pic('benchmark.png',475)
p('Figura 3. RMSE por componente, em píxeis: raiz da média dos erros quadrados de x e y sobre todas as observações válidas. Barras de erro: desvio padrão entre seeds, não intervalo de confiança entre espécimes.','SmallP')
p(f'GNN: {gnn:.3f} ± {a["gnn"]["macro_rmse_px_sd_over_seeds"]:.3f} px; MLP: {a["mlp"]["macro_rmse_px_mean_over_seeds"]:.3f} ± {a["mlp"]["macro_rmse_px_sd_over_seeds"]:.3f} px. A média entre espécimes dá o mesmo peso a cada ensaio. O modelo sem deslocamento tem RMSE {a["zero_displacement"]["macro_rmse_px"]:.3f} px.')
h('Onde melhora')
rows=[['Espécime de teste','GNN: RMSE x','GNN: RMSE y','GNN: R² conjunto']]
for f in d['folds']:rows.append([f['held_out_specimen'],fmt(avg(f,'gnn','rmse_x_px')),fmt(avg(f,'gnn','rmse_y_px')),fmt(avg(f,'gnn','r2_variance_weighted'))])
table(rows,[145,115,115,115])
p('A melhoria resulta sobretudo da componente horizontal. A componente vertical permanece próxima da base afim. O erro global de cerca de 6,35 px corresponde aproximadamente a 16% da altura inicial de 39–41 px; ainda é relevante em termos mecânicos.','SmallP')

page('4. Inspeção das previsões')
p('As figuras seguintes usam a média das previsões dos três modelos de cada fold, sem escolher a seed com menor erro de teste. Esta média de previsões é diferente da média de RMSE da tabela anterior.')
pic('deformation_comparison.png',490)
p('Figura 4. Posições medidas pelo rastreamento (preto) e posições previstas para o espécime reservado (azul). O eixo vertical é ampliado para permitir leitura. A fase 1 de Probe1_22 tem apenas metade dos nós observados; a linha prevista não valida os nós sem medição.','SmallP')
p('A rede acompanha a redução global da altura e parte do movimento lateral. Persistem desvios de posição e detalhe local. Como a topologia inicial é fixa, estas linhas não representam uma simulação de contacto, fusão de nós ou dano.')

page('5. Robustez, limites e próxima etapa')
pic('error_by_phase.png',475)
p('Figura 5. Erro por fase da média das previsões GNN e da base afim. A região cinzenta excede o intervalo de ajuste de parâmetros. Não houve seleção de configurações usando estas curvas de teste.','SmallP')
h('Verificações executadas')
p('Foram aprovados 17 testes. A suite de integridade verifica correspondência com as tabelas originais, máscaras, normalização, arestas, gradientes, equivariância, influência da topologia e recarregamento dos pesos. Um teste perturba os alvos do espécime reservado e confirma que os pesos e a época escolhida permanecem idênticos; outro substitui alvos mascarados por NaN sem afetar o ajuste. A inferência também é testada com um ficheiro que contém apenas a geometria inicial.')
p('Sensibilidade à perda de tracking: usando apenas nós observados em toda a compressão, o RMSE GNN é 6,245 px (42 nós de Probe2_7) e 6,317 px (15 nós de Probe1_22). Estes valores são próximos dos 6,391 e 6,310 px da avaliação principal; ainda não validam os nós perdidos.', 'SmallP')
h('Limites que condicionam a interpretação')
p('<b>Dimensão amostral:</b> dois espécimes do mesmo estudo, com geometria semelhante. <b>Qualidade dos alvos:</b> tracking automático quantizado em píxeis, sem validação independente por DIC. <b>Seleção:</b> os casos disponíveis correspondem a imagens adequadas à análise, não a uma amostra representativa de defeitos e materiais. <b>Informação retrospetiva:</b> a fase usa o fim anotado do ensaio. <b>Física:</b> não há materiais, tensão, forças, contacto nem restrições constitutivas como entradas ou saídas.')
p('As curvas de força Ex1/Ex2 foram identificadas e a associação ao espécime foi verificada. A sincronização original com imagens é inferida por velocidade de compressão e extremos anotados. A escala mm/pixel também usa a deformação final. Por isso, esta versão mantém unidades em píxeis e não inventa uma supervisão de forças ou tensões.')
h('Próxima campanha com maior valor')
p('Reunir mais espécimes independentes, incluindo geometrias, papéis e imperfeições diferentes; guardar imagens, tracking e calibração espacial independente; sincronizar os timestamps da câmara e da máquina. Reservar geometrias e lotes completos para teste. Substituir a fase pela deformação imposta medida e só depois acrescentar uma cabeça de previsão de força. Testar contacto e regularização física após estabelecer baselines fortes.')

page('6. Entrega e reprodução')
p('O projeto está organizado para repetir o percurso desde os arquivos de origem até às previsões. Os dados preparados compactos e os checkpoints de benchmark estão incluídos no repositório privado GNN-CB.')
table([['Localização','Conteúdo'],['gnn_cb/data.py','Importação, hashes, máscaras, normalização e proveniência.'],['gnn_cb/model.py / train.py','GNN, modelo sem arestas, base afim e protocolo completo de treino.'],['gnn_cb/predict.py','Inferência a partir de geometria inicial e fase; exporta NPZ e CSV.'],['data/processed/','Dois datasets preparados, manifesto e imagens de origem selecionadas.'],['results/loso/','Pesos, métricas, histórias de treino e previsões dos dois folds.'],['tests/test_pipeline.py','Testes de integridade e isolamento entre treino e teste.'],['reports/','Este relatório, figuras, auditoria dos dados e revisão científica.']],[170,320])
h('Executar uma previsão já treinada')
p('python -m gnn_cb.predict<br/>  --checkpoint results/loso/<br/>    train_Probe1_22__test_Probe2_7/gnn_seed11.pt<br/>  --graph data/processed/Probe2_7.npz<br/>  --phase 0 0.25 0.5 0.75 1<br/>  --output results/my_prediction.npz','CodeX')
p('Juntar as linhas anteriores num comando; README.md contém a versão pronta a copiar. A saída fornece deslocamentos e posições por nó. A inferência não lê as trajetórias de referência incluídas no dataset.')
h('Repetir preparação, treino e verificações')
p('python scripts/prepare_data.py --download<br/>python -m gnn_cb.train --epochs 600 --seeds 11 29 47<br/>python -m unittest discover -s tests -v<br/>python scripts/plot_data.py<br/>python scripts/plot_results.py<br/>python scripts/build_report.py','CodeX')
p('Ambiente executado: Python e versões de pacotes em reports/environment.json e requirements-tested.txt. CPU; não requer PyTorch Geometric. Os arquivos originais são públicos e fixados ao commit 16eed74ff025e0e0f40f4f5d81c2b71af58b93b5. O modelo conjunto results/final/model.pt foi treinado nos dois espécimes durante 198 épocas, escolhidas pela mediana da validação de origem. Não tem teste independente. Os 12 ajustes do benchmark demoraram 96,2 s de treino acumulado em CPU.','SmallP')
h('Referências e atribuição')
p('[1] Belfekih, T.; Fitas, R.; Schaffrath, H.-J.; Schabel, S. Graph-Based Analysis for the Characterization of Corrugated Board Compression. Materials 17(24), 6083 (2024). DOI: 10.3390/ma17246083.<br/>[2] Ricardo Fitas. Graphs_Analysis_Corrugated_Boards. github.com/ricardofitas/Graphs_Analysis_Corrugated_Boards. Commit fixado acima; licença MIT retida no projeto. Auditoria adicional: Graph-CB, commit 765baa23e364538b05ec293cea2d8ade8b4c4a03.<br/>[3] Pfaff, T. et al. Learning Mesh-Based Simulation with Graph Networks. ICLR 2021. arXiv:2010.03409.','SmallP')

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#d8e0e7'));c.line(48,44,547,44)
 c.setFont('Body',7);c.setFillColor(gray);c.drawString(48,30,'GNN-CB | Abordagem, resultados e limites | 09.10.2026');c.drawRightString(547,30,str(doc.page))
 c.setTitle('GNN-CB | Relatório de abordagem e resultados');c.setAuthor('Prepared for Ricardo Fitas')
REPORT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(REPORT),pagesize=(595.28,841.89),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=58)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(REPORT)
