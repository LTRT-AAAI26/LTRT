import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.ndimage import convolve
from numpy import unwrap
from matplotlib import cm


def pad_to_square(image, pad_color=0):
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

def crop_to_original(image, original_h, original_w):
    h, w = image.shape[:2]
    top = (h - original_h) // 2
    left = (w - original_w) // 2
    return image[top:top+original_h, left:left+original_w]

def fconv(x, y, c):
    N = len(np.concatenate([x.ravel(), y.ravel()])) - 1
    P = 2**int(np.ceil(np.log2(N)))
    z = fft(x, P) * fft(y, P)
    z = ifft(z)
    return z[:N] if c else ifft(z)

def fractional_frft_axis(temp, a, shft, sN):
    N = temp.shape[0]
    if a > 2:
        a -= 2
        temp = np.flipud(temp)
    elif a > 1.5:
        a -= 1
        temp[shft, :] = fft(temp[shft, :], axis=0) / sN
    elif a < 0.5:
        a += 1
        temp[shft, :] = ifft(temp[shft, :], axis=0) * sN

    alpha = a * np.pi / 2
    s = np.pi / (N + 1) / np.sin(alpha) / 4
    t = np.pi / (N + 1) * np.tan(alpha / 2) / 4
    Cs = np.sqrt(s / np.pi) * np.exp(-1j * (1 - a) * np.pi / 4)

    snc = np.sinc(np.arange(-(2*N-3), 2*N-3+1, 2)/2)
    chrp = np.exp(-1j * t * np.arange(-N+1, N)**2)
    chrp2 = np.exp(1j * s * np.arange(-(2*N-1), 2*N-1+1)**2)

    for ix in range(N):
        f0 = temp[:, ix]
        f1 = fconv(f0, snc, 1)[N-1:2*N-2]
        l0, l1 = chrp[::2], chrp[1::2]
        e1, e0 = chrp2[::2], chrp2[1::2]
        h0 = ifft(fconv(f0*l0, e0, 0) + fconv(f1*l1, e1, 0))[N-1:2*N-1]
        temp[:, ix] = Cs * l0 * h0
    return temp

def frft2d(matrix, angles):
    matrix = matrix.astype(np.complex128)
    temp = np.transpose(matrix)

    for dim, a in enumerate(angles):
        N = temp.shape[0]
        a %= 4
        shft = (np.arange(N) + N // 2) % N
        sN = np.sqrt(N)

        if a == 0:
            pass
        elif a == 2:
            temp = np.flipud(temp)
        elif a == 1:
            temp[shft, :] = fft(temp[shft, :], axis=0) / sN
        elif a == 3:
            temp[shft, :] = ifft(temp[shft, :], axis=0) * sN
        else:
            temp = fractional_frft_axis(temp, a, shft, sN)

        temp = np.transpose(temp)
    return np.transpose(temp)

def trker(input, t):
    m, n, v = input.shape
    trkerResult = np.zeros((m, n, v))

    d = 2
    g1 = math.gamma((1 + d) / 2)
    g2 = math.gamma(1.5)
    g3 = math.gamma((2 + d) / 2)
    constant = g1 * np.pi * t / (g2 * g3)
    factor = -np.pi**2 * t**2

    U, V = np.meshgrid(np.arange(n), np.arange(m))
    dist2 = (U - n//2)**2 + (V - m//2)**2
    dist2 = dist2 / np.max(dist2)
    dist2[dist2 == 0] = np.finfo(float).eps

    maxIter = 30
    pochhammer = lambda x, k: math.gamma(x + k) / math.gamma(x)

    a = 0.5
    b = [1.5, 1 + d/2]
    Z = factor * dist2
    H = np.zeros_like(Z)

    for k in range(maxIter+1):
        term = (pochhammer(a, k) / (pochhammer(b[0], k) * pochhammer(b[1], k))) * (Z**k) / math.factorial(k)
        H += term
        if np.max(np.abs(term)) < 1e-8:
            break

    for c in range(v):
        trkerResult[:,:,c] = 1 - (constant * np.sqrt(dist2)) * H

    return trkerResult

def generate_Truncated_riesz_kernels(shape, p, trkerResult):
    m, n = shape
    X, Y = np.meshgrid(np.arange(n)+1, np.arange(m)+1)
    a1 = 0.5 * np.pi * p[0]
    a2 = 0.5 * np.pi * p[1]
    denom = np.sqrt((X/np.sin(a1))**2 + (Y/np.sin(a2))**2) + 1e-6
    dxker = -1j * (X/np.sin(a1)) / denom * trkerResult[:,:,0]
    dyker = -1j * (Y/np.sin(a2)) / denom * trkerResult[:,:,0]
    return dxker, dyker


def fractional_truncated_riesz_transform(image, p, t=0.0001):
    img_expand = np.expand_dims(image, axis=-1)
    trkerResult = trker(img_expand, t)

    faI = frft2d(image, p)
    dxk, dyk = generate_Truncated_riesz_kernels(image.shape, p, trkerResult)

    Rfa1 = dxk * faI
    Rfa2 = dyk * faI

    R1 = np.abs(frft2d(Rfa1, [-p[0], -p[1]]))
    R2 = np.abs(frft2d(Rfa2, [-p[0], -p[1]]))

    orientation = np.arctan2(R2, R1)
    phase = unwrap(np.arctan2(np.sqrt(R1**2 + R2**2), image))
    amplitude = np.sqrt(image**2 + R1**2 + R2**2)

    return orientation, phase, amplitude


def gaussian_filter_2d(size, sigma):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, y)
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return g / g.sum()

def apply_gaussian_filter(img, size=2, sigma=0.5):
    kernel = gaussian_filter_2d(size, sigma)
    return convolve(img, kernel, mode='nearest')

def binarize_image(img, threshold=0.7863):
    return (img > threshold).astype(np.uint8) * 255


if __name__ == "__main__":
    a = [1, 1]
    img = cv2.imread('Figure.png', cv2.IMREAD_GRAYSCALE)
    original_h, original_w = img.shape
    img_pad = pad_to_square(img)

    img_double = img_pad.astype(np.float64)
    orientation, phase, amplitude = fractional_truncated_riesz_transform (img_double, a)

    G = apply_gaussian_filter(phase, size=2, sigma=0.5)
    img_out = binarize_image(G)
    img_out = crop_to_original(img_out, original_h, original_w)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_out, cmap='gray')
    #plt.title('Fractional Riesz Edge Detection')
    plt.axis('off')
    plt.show()

