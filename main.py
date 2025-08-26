from PIL import Image
import numpy as np
import math

# --- Funções de Pré-processamento e Detecção de Bordas ---

def carregar_imagem_cinza(caminho):
    """Carrega uma imagem, converte para escala de cinza e retorna uma matriz de pixels."""
    img = Image.open(caminho).convert('L')
    largura, altura = img.size
    pixels = list(img.getdata())
    matriz = [[pixels[y * largura + x] for x in range(largura)] for y in range(altura)]
    return matriz, largura, altura

### NOVA ADIÇÃO: Função para salvar a matriz de bordas ###
def salvar_imagem_de_matriz(matriz, caminho):
    """Salva uma matriz de pixels como uma imagem em escala de cinza."""
    altura = len(matriz)
    largura = len(matriz[0])
    img = Image.new('L', (largura, altura))
    # Achata a matriz de volta para uma lista de pixels
    pixels = [max(0, min(255, matriz[y][x])) for y in range(altura) for x in range(largura)]
    img.putdata(pixels)
    img.save(caminho)

def suavizar_gaussiana_3x3(matriz, largura, altura):
    """Aplica um filtro de suavização Gaussiana 3x3."""
    suavizada = [[0 for _ in range(largura)] for _ in range(altura)]
    pesos = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            soma = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    soma += matriz[y + dy][x + dx] * pesos[dy + 1][dx + 1]
            suavizada[y][x] = soma // 16
    return suavizada

def operador_sobel(matriz, largura, altura):
    """Aplica o operador de Sobel para detecção de bordas."""
    resultado = [[0 for _ in range(largura)] for _ in range(altura)]
    kernel_gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            soma_gx, soma_gy = 0, 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    pixel_val = matriz[y + dy][x + dx]
                    soma_gx += pixel_val * kernel_gx[dy + 1][dx + 1]
                    soma_gy += pixel_val * kernel_gy[dy + 1][dx + 1]
            magnitude = math.sqrt(soma_gx**2 + soma_gy**2)
            resultado[y][x] = min(255, int(magnitude))
    return resultado

# --- Funções para Limpeza Morfológica ---

def dilatar(matriz, largura, altura):
    """Aplica a operação de dilatação para engrossar as bordas."""
    resultado = [[0 for _ in range(largura)] for _ in range(altura)]
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            vizinhanca_branca = False
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if matriz[y + dy][x + dx] == 255:
                        vizinhanca_branca = True
                        break
                if vizinhanca_branca: break
            if vizinhanca_branca: resultado[y][x] = 255
    return resultado

def erodir(matriz, largura, altura):
    """Aplica a operação de erosão para afinar as bordas e remover ruído."""
    resultado = [[0 for _ in range(largura)] for _ in range(altura)]
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            todos_brancos = True
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if matriz[y + dy][x + dx] == 0:
                        todos_brancos = False
                        break
                if not todos_brancos: break
            if todos_brancos: resultado[y][x] = 255
    return resultado

# --- Função de Preenchimento (Flood Fill) ---

def flood_fill_iterativo(matriz, x, y, cor_alvo, cor_preenchimento):
    """Preenche uma área da matriz com uma nova cor de forma iterativa."""
    altura, largura = matriz.shape
    if x < 0 or x >= largura or y < 0 or y >= altura or matriz[y, x] != cor_alvo:
        return
    pilha = [(x, y)]
    while len(pilha) > 0:
        px, py = pilha.pop()
        if px < 0 or px >= largura or py < 0 or py >= altura or matriz[py, px] != cor_alvo:
            continue
        matriz[py, px] = cor_preenchimento
        pilha.extend([(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)])

# --- EXECUÇÃO PRINCIPAL (ABORDAGEM HÍBRIDA) ---

if __name__ == "__main__":
    # --- Configurações ---
    ARQUIVO_ENTRADA = 'metal_nut/test/scratch/000.png'
    ARQUIVO_SAIDA_BORDAS = 'resultado_bordas.png' ### NOVA ADIÇÃO ###
    ARQUIVO_SAIDA_FINAL = 'resultado_final.png'
    
    # Parâmetros para ISOLAR
    LIMIAR_BORDA = 40
    # Ponto de semente DENTRO
    PONTO_SEMENTE = (150, 250) 
    
    # Parâmetro para ENCONTRAR AS RACHADURAS
    LIMIAR_RACHADURA = 150 # Limiar de escuridão para as rachaduras
    
    # Cor de destaque
    COR_DESTAQUE_RGB = [255, 100, 0] # Laranja

    print("Iniciando abordagem híbrida...")

    # ETAPA A: ISOLAR
    print("Etapa A: Segmentando a imagem...")
    matriz_cinza, largura, altura = carregar_imagem_cinza(ARQUIVO_ENTRADA)
    matriz_suave = suavizar_gaussiana_3x3(matriz_cinza, largura, altura)
    matriz_bordas = operador_sobel(matriz_suave, largura, altura)
    
    ### NOVA ADIÇÃO: Salvando a imagem de bordas ###
    salvar_imagem_de_matriz(matriz_bordas, ARQUIVO_SAIDA_BORDAS)
    print(f"Detecção de bordas concluída. Resultado salvo em '{ARQUIVO_SAIDA_BORDAS}'")

    bordas_np = np.array(matriz_bordas, dtype=np.uint8)
    bordas_np[bordas_np > LIMIAR_BORDA] = 255
    bordas_np[bordas_np <= LIMIAR_BORDA] = 0
    
    bordas_lista = bordas_np.tolist()
    bordas_dilatadas = dilatar(bordas_lista, largura, altura)
    bordas_fechadas = erodir(bordas_dilatadas, largura, altura)
    
    mascara_limpa = np.array(bordas_fechadas, dtype=np.uint8)
    
    # Usamos o Flood Fill para obter a máscara 
    mascara_np = np.copy(mascara_limpa)
    flood_fill_iterativo(mascara_np, PONTO_SEMENTE[0], PONTO_SEMENTE[1], cor_alvo=0, cor_preenchimento=255)
    mascara = (mascara_np == 255) # Máscara booleana
    print("Segmentação concluída.")

    # ETAPA B: ENCONTRAR RACHADURAS (pixels escuros)
    print("Etapa B: Encontrando todos os pixels escuros...")
    matriz_cinza_np = np.array(matriz_cinza, dtype=np.uint8)
    mascara_escura = (matriz_cinza_np < LIMIAR_RACHADURA)
    print(f"Máscara de pixels com valor < {LIMIAR_RACHADURA} criada.")

    # ETAPA C: COMBINAR MÁSCARAS (A MÁGICA ACONTECE AQUI)
    print("Etapa C: Combinando as máscaras para isolar as rachaduras...")
    # A máscara final é a intersecção: pixels que estão na imagem E são escuros
    mascara_final_rachaduras = mascara & mascara_escura

    # ETAPA D: APLICAR COR E SALVAR
    imagem_original_colorida = Image.open(ARQUIVO_ENTRADA).convert('RGB')
    pixels_coloridos = np.array(imagem_original_colorida)
    pixels_coloridos[mascara_final_rachaduras] = COR_DESTAQUE_RGB
    
    imagem_final = Image.fromarray(pixels_coloridos)
    imagem_final.save(ARQUIVO_SAIDA_FINAL)

    print("\nProcesso finalizado com sucesso!")
    print(f"Resultado final salvo em '{ARQUIVO_SAIDA_FINAL}'.")