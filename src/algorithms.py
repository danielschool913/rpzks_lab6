import random
import numpy as np
from shapely.geometry import Point, Polygon


def gauss_area(polygon: Polygon) -> float:
    """
    Обчислює площу полігону за методом Гауса (Формула шнурків / Shoelace formula).

    Площа = 0.5 * |Σ (x_i * y_{i+1} - x_{i+1} * y_i)|

    Args:
        polygon (Polygon): Об'єкт полігону Shapely.

    Returns:
        float: Площа полігону.
    """
    coords = list(polygon.exterior.coords[:-1])  # Без повторення першої точки
    n = len(coords)
    area = 0.0
    for i in range(n):
        x_i, y_i = coords[i]
        x_next, y_next = coords[(i + 1) % n]
        area += (x_i * y_next) - (x_next * y_i)
    return abs(area) / 2.0


def monte_carlo_area(polygon: Polygon, num_points: int = 100_000, seed: int = None) -> float:
    """
    Обчислює площу полігону методом Монте-Карло.

    Алгоритм:
    1. Визначається Bounding Box полігону.
    2. Генерується num_points випадкових точок всередині BB.
    3. Рахується кількість точок, що потрапили всередину полігону (через Shapely .contains()).
    4. Площа = площа_BB * (к-сть_влучань / к-сть_точок).

    Args:
        polygon (Polygon): Об'єкт полігону Shapely.
        num_points (int): Кількість випадкових точок.
        seed (int): Seed для відтворюваності.

    Returns:
        float: Наближена площа полігону.
    """
    if seed is not None:
        random.seed(seed)

    min_x, min_y, max_x, max_y = polygon.bounds
    bbox_area = (max_x - min_x) * (max_y - min_y)

    hits = 0
    for _ in range(num_points):
        rx = random.uniform(min_x, max_x)
        ry = random.uniform(min_y, max_y)
        if polygon.contains(Point(rx, ry)):
            hits += 1

    return bbox_area * (hits / num_points)
