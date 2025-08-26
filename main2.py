import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def gerar_saidas_sobel_canny(caminho_local, salvar_arquivos=True):
    """
    Carrega uma imagem local de uma peça, aplica os filtros Sobel e Canny
    e exibe e/ou salva as saídas para comparação manual.

    Args:
        caminho_local (str): Caminho para o arquivo de imagem local.
        salvar_arquivos (bool, optional): Se True, salva as imagens resultantes no disco.
    """
    try:
        # Carregar a imagem do caminho local fornecido
        imagem_original = cv2.imread(caminho_local)
        if imagem_original is None:
            raise FileNotFoundError(f"Não foi possível encontrar ou ler a imagem em: {caminho_local}")
            
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return

    # --- 1. Pré-processamento ---
    # Converter para escala de cinza para detecção de bordas
    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    # Suavizar a imagem para reduzir o ruído (muito importante para esta imagem com textura)
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (7, 7), 0)

    # --- 2. Aplicar Filtro Sobel ---
    # Ajustando a profundidade para CV_64F para capturar melhor os gradientes
    sobel_x = cv2.Sobel(imagem_suavizada, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(imagem_suavizada, cv2.CV_64F, 0, 1, ksize=5)
    # Obter a magnitude absoluta dos gradientes
    bordas_sobel = cv2.magnitude(sobel_x, sobel_y)
    bordas_sobel = cv2.convertScaleAbs(bordas_sobel)

    # --- 3. Aplicar Detector Canny ---
    # Os limiares (thresholds) podem precisar de ajuste dependendo da imagem.
    # Para esta imagem, valores um pouco mais altos podem ajudar a focar nas bordas principais.
    bordas_canny = cv2.Canny(imagem_suavizada, threshold1=40, threshold2=120)

    # --- 4. Salvar as Saídas (Opcional) ---
    if salvar_arquivos:
        output_dir = "resultados_peca"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        caminho_sobel = os.path.join(output_dir, "saida_sobel.png")
        caminho_canny = os.path.join(output_dir, "saida_canny.png")
        caminho_original = os.path.join(output_dir, "imagem_original_analisada.png")

        cv2.imwrite(caminho_sobel, bordas_sobel)
        cv2.imwrite(caminho_canny, bordas_canny)
        cv2.imwrite(caminho_original, imagem_original)
        
        print(f"Arquivos de saída salvos no diretório '{os.path.abspath(output_dir)}'")

    # --- 5. Exibir as Saídas para Comparação Manual ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, eixos = plt.subplots(1, 3, figsize=(18, 6))

    imagem_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
    
    eixos[0].imshow(imagem_rgb)
    eixos[0].set_title('Sua Imagem Original')
    eixos[0].axis('off')

    eixos[1].imshow(bordas_sobel, cmap='gray')
    eixos[1].set_title('Saída do Filtro Sobel')
    eixos[1].axis('off')

    eixos[2].imshow(bordas_canny, cmap='gray')
    eixos[2].set_title('Saída do Detector Canny')
    eixos[2].axis('off')

    plt.suptitle('Análise de Bordas na Sua Peça', fontsize=16)
    plt.tight_layout()
    plt.show()

# --- Execução ---
# Certifique-se de que a imagem "minha_peca.jpeg" está na mesma pasta que este script.
caminho_da_sua_imagem = "metal_nut/test/scratch/000.png" 
gerar_saidas_sobel_canny(caminho_local=caminho_da_sua_imagem, salvar_arquivos=True)