import cv2
import numpy as np
import mss

''' Biblioteka mss służy do przechwytywania ekranu.'''
sct = mss.mss()
full = sct.monitors[1]

''' Ustawienia przechwytywania ekranu - lewa połowa ekranu '''
monitor = {
    "top": 0,
    "left": 0,
    "width": full["width"] // 2,
    "height": full["height"]
}

''' Funkcja szacująca średnią wartość koloru BGR w danym obrazie '''
def mean_bgr(img):
    return np.mean(img.reshape(-1, 3), axis=0)

''' Funkcje sprawdzające, czy dany kolor BGR odpowiada określonemu kolorowi flagi '''
def is_white(bgr):
    b, g, r = bgr
    return abs(r-g) < 25 and abs(r-b) < 25 and abs(g-b) < 25 and (r+g+b)/3 > 140

def is_red(bgr):
    b, g, r = bgr
    return r > g + 30 and r > b + 30

def is_blue(bgr):
    b, g, r = bgr
    return b > g + 30 and b > r + 30

def is_yellow(bgr):
    b, g, r = bgr
    return r > 140 and g > 140 and abs(r-g) < 40 and b < 150


while True:

    ''' Przechwytywanie ekranu '''
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    ''' Przetwarzanie obrazu '''    
    blur = cv2.GaussianBlur(frame, (5,5), 0)

    ''' Maskowanie kolorów flag '''
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    ''' Maski dla kolorów nasyconych i jasnych '''
    mask_color = cv2.inRange(s, 30, 255)
    mask_white = cv2.inRange(v, 180, 255)
    color_mask = cv2.bitwise_or(mask_color, mask_white)

    kernel = np.ones((7,7), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    ''' Znajdowanie konturów na masce '''
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ''' Analiza konturów '''
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        if ratio < 1.0 or ratio > 3.5:
            continue

        roi = blur[y:y+h, x:x+w]

        ''' Podział na górną i dolną połowę '''            
        top = mean_bgr(roi[0:h//2])
        bottom = mean_bgr(roi[h//2:h])

        ''' Rozpoznawanie flag '''
        flag = None

        if is_white(top) and is_red(bottom):
            flag = "Polska"

        elif is_blue(top) and is_yellow(bottom):
            flag = "Ukraina"

        else:
            third = h // 3
            t = mean_bgr(roi[0:third])
            m = mean_bgr(roi[third:2*third])
            b = mean_bgr(roi[2*third:3*third])

            if is_white(t) and is_blue(m) and is_red(b):
                flag = "Rosja"

        if flag:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            label = f"{flag}  x:{x} y:{y}"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            print(f"{flag} -> x={x}, y={y}, w={w}, h={h}")

    ''' Zakończenie pętli po naciśnięciu klawisza ESC '''
    if cv2.waitKey(1) == 27:
        break