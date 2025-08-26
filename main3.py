import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def operador_sobel_manual(matriz, largura, altura):
    print("Iniciando a execucao do Sobel manual (pode levar alguns segundos)...")
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
            
    print("Sobel manual concluido.")
    return resultado

def comparar_sobel_vs_canny(caminho_imagem):
    try:
        imagem_original = cv2.imread(caminho_imagem)
        if imagem_original is None:
            raise FileNotFoundError(f"Imagem nao encontrada em: {caminho_imagem}")
        
        imagem_cinza_np = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
        
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return

    altura, largura = imagem_cinza_np.shape
    matriz_cinza_lista = imagem_cinza_np.tolist()
    
    inicio_sobel = time.time()
    resultado_sobel_lista = operador_sobel_manual(matriz_cinza_lista, largura, altura)
    fim_sobel = time.time()
    tempo_sobel = fim_sobel - inicio_sobel
    
    resultado_sobel_np = np.array(resultado_sobel_lista, dtype=np.uint8)

    print("\nIniciando a execucao do Canny (OpenCV)...")
    imagem_desfocada = cv2.GaussianBlur(imagem_cinza_np, (5, 5), 0)
    
    inicio_canny = time.time()
    resultado_canny = cv2.Canny(imagem_desfocada, 50, 150)
    fim_canny = time.time()
    tempo_canny = fim_canny - inicio_canny
    print("Canny concluido.")
    
    print("\n--- Relatorio de Performance ---")
    print(f"Tempo de execucao do Sobel Manual: {tempo_sobel:.4f} segundos")
    print(f"Tempo de execucao do Canny OpenCV: {tempo_canny:.4f} segundos")
    print("--------------------------------\n")
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagem_cinza_np, cmap='gray')
    plt.title("Imagem Original em Tons de Cinza")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(resultado_sobel_np, cmap='gray')
    plt.title(f"Sobel (Implementacao Manual)\nTempo: {tempo_sobel:.2f}s")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(resultado_canny, cmap='gray')
    plt.title(f"Canny (Biblioteca OpenCV)\nTempo: {tempo_canny:.4f}s")
    plt.axis('off')
    
    plt.suptitle("Comparacao de Desempenho e Qualidade: Sobel Manual vs. Canny OpenCV", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    caminho_da_imagem = "metal_nut/test/scratch/000.png"
    comparar_sobel_vs_canny(caminho_da_imagem)