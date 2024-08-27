import cv2
import os
import glob2
from skimage.transform import resize


def show_images(titles, images, wait=True):
    for (title, image) in zip(titles, images):
        cv2.imshow(title, image)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_people(imagem, hog):
    imagem_suave = cv2.GaussianBlur(imagem, (5, 5), 0)

    img_cinza = cv2.cvtColor(imagem_suave, cv2.COLOR_BGR2GRAY)

    retos, pesos = hog.detectMultiScale(img_cinza, winStride=(4, 4), padding=(8, 8), scale=1.05)

    retangulos_filtrados = cv2.groupRectangles(list(retos), groupThreshold=1, eps=0.1)[0]

    return retos, pesos


def processa_imagem(img_pasta, hog):
    nome_imagem = os.path.basename(img_pasta)
    imagem = cv2.imread(img_pasta)

    if imagem.shape[1] < 400:
        (altura, largura) = imagem.shape[:2]
        relacao = largura / float(largura)
        imagem = cv2.resize(imagem, (400, int(altura * relacao)))

    retangulos, pesos = detect_people(imagem, hog)
    print(f'Pessoas detectadas: {len(retangulos)}')

    if len(retangulos) == 0:
        return

    for i, (x, y, w, h) in enumerate(retangulos):
        print(f'Peso {i}: {pesos[i]}')

        if pesos[i] < 0.13:
            continue
        elif pesos[i] < 0.3 and pesos[i] > 0.13:
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if pesos[i] < 0.7 and pesos[i] > 0.3:
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (50, 122, 255), 2)
        if pesos[i] > 0.7:
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(imagem, 'Alta confianca', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(imagem, 'Confianca moderada', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 122, 255), 2)
    cv2.putText(imagem, 'Baixa confianca', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.imshow('Detecção HOG', imagem)

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, nome_imagem), imagem)
    cv2.waitKey(0)


if __name__ == '__main__':
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    caminho_imagens = glob2.glob('./images/*')
    for img_pasta in caminho_imagens:
        processa_imagem(img_pasta, hog)
