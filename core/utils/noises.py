import numpy as np

from PIL import Image


def perlin_noise_2d(shape, res):
    def interpolant(t):
        return t*t*t*(t*(t*6 - 15) + 10)

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]

    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0]    , grid[:, :, 1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0]    , grid[:, :, 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1]-1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1

    for _ in range(octaves):
        noise += amplitude * perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= lacunarity
        amplitude *= persistence
    return (noise - np.min(noise)) / (np.max(noise) - np.min(noise))


def random_fractal_image(width, height):
    _pow = int(np.ceil(np.log(max(width, height)) / np.log(2)))
    octaves = _pow - 4
    size = 2 ** _pow
    r = fractal_noise_2d((size, size), (32, 32), octaves=octaves)
    g = fractal_noise_2d((size, size), (32, 32), octaves=octaves)
    b = fractal_noise_2d((size, size), (32, 32), octaves=octaves)

    tile = np.dstack((r, g, b))[:height, :width, :]
    return Image.fromarray((255.9 * tile).astype('uint8'))


def random_noise_image(width, height):
    return Image.fromarray(
        np.random.randint(0, 255, (width, height, 3), dtype=np.dtype('uint8'))
    )


def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, starts, stops, is_horizontal_list):
    result = np.zeros((height, width, len(starts)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(starts, stops, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result


def random_gradient_image(width, height):
    array = gradient_3d(
        width,
        height,
        (0, 0, np.random.randint(0, 255)),
        (np.random.randint(1, 255), np.random.randint(2, 255), np.random.randint(3, 128)),
        (True, False, False)
    )
    random_image = Image.fromarray(np.uint8(array))
    return random_image
