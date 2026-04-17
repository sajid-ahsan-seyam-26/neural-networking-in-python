
import math
import random
import json
import os
import numpy as np
import pygame


WIDTH, HEIGHT = 1200, 800
FPS = 60

BACKGROUND = (18, 22, 28)

GRASS_BASE = (46, 125, 60)
GRASS_DARK = (36, 102, 49)

ASPHALT_DARK = (48, 50, 56)
ASPHALT_MID = (60, 63, 70)
ASPHALT_LIGHT = (76, 79, 87)

CURB_RED = (196, 48, 48)
CURB_WHITE = (235, 235, 235)
LANE_WHITE = (245, 245, 245)

CAR_COLOR = (60, 160, 255)
BEST_CAR_COLOR = (255, 200, 70)
DEAD_CAR_COLOR = (95, 95, 100)
CAR_GLASS = (170, 220, 255)
CAR_SHADOW = (25, 25, 28)
TIRE_COLOR = (28, 28, 30)
HEADLIGHT_COLOR = (255, 245, 190)
BRAKE_LIGHT_COLOR = (255, 90, 90)

SENSOR_COLOR = (120, 255, 180)
CHECKPOINT_COLOR = (80, 255, 120)

TEXT_COLOR = (245, 245, 245)
TEXT_DIM = (180, 185, 195)
HUD_PANEL = (20, 24, 30)
HUD_PANEL_2 = (28, 33, 41)
HUD_BORDER = (70, 78, 92)
HUD_ACCENT = (88, 170, 255)

TRACK_OUTER_RECT = pygame.Rect(80, 80, 900, 600)
TRACK_INNER_RECT = pygame.Rect(280, 200, 500, 360)
CENTER_LOOP_RECT = pygame.Rect(180, 140, 700, 480)

POPULATION_SIZE = 60
ELITE_COUNT = 6
TOURNAMENT_SIZE = 5

MUTATION_RATE = 0.10
MUTATION_STRENGTH = 0.25
CROSSOVER_BLEND = 0.50

MAX_STEPS_PER_GENERATION = 1400
IDLE_LIMIT = 120
SIMULATION_STEPS_PER_FRAME = 3

MAX_SPEED = 6.0
MAX_REVERSE_SPEED = -1.5
ACCELERATION = 0.18
BRAKE_FORCE = 0.10
FRICTION = 0.03
TURN_RATE = 0.055
CAR_RADIUS = 9
MAX_SENSOR_DISTANCE = 220

MODEL_FILE = "best_brain.json"

SENSOR_ANGLES = [-90, -60, -35, -18, 0, 18, 35, 60, 90]

INPUT_SIZE = len(SENSOR_ANGLES) + 4
HIDDEN_1 = 16
HIDDEN_2 = 12
OUTPUT_SIZE = 3  # steer, throttle, brake


def clamp(value, low, high):
    return max(low, min(high, value))


def distance(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)


def point_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = clamp(t, 0.0, 1.0)

    closest_x = ax + abx * t
    closest_y = ay + aby * t
    return math.hypot(px - closest_x, py - closest_y)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def line_intersection_point(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (px, py)


def segment_intersection(p1, p2, q1, q2):
    if not segments_intersect(p1, p2, q1, q2):
        return None
    return line_intersection_point(p1, p2, q1, q2)


def rect_to_segments(x, y, w, h):
    return [
        ((x, y), (x + w, y)),
        ((x + w, y), (x + w, y + h)),
        ((x + w, y + h), (x, y + h)),
        ((x, y + h), (x, y)),
    ]


def angle_wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def checkpoint_center(cp):
    (x1, y1), (x2, y2) = cp
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def rotate_point(px, py, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        px * cos_a - py * sin_a,
        px * sin_a + py * cos_a
    )


def world_points(cx, cy, angle, local_points):
    pts = []
    for px, py in local_points:
        rx, ry = rotate_point(px, py, angle)
        pts.append((cx + rx, cy + ry))
    return pts


def draw_dashed_line(surface, color, start, end, dash_length=24, gap_length=16, width=3):
    x1, y1 = start
    x2, y2 = end
    total_len = math.hypot(x2 - x1, y2 - y1)

    if total_len == 0:
        return

    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len

    dist = 0.0
    while dist < total_len:
        dash_end = min(dist + dash_length, total_len)

        sx = x1 + dx * dist
        sy = y1 + dy * dist
        ex = x1 + dx * dash_end
        ey = y1 + dy * dash_end

        pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)
        dist += dash_length + gap_length


def draw_striped_edge(surface, start, end, thickness=10, segment_length=26):
    x1, y1 = start
    x2, y2 = end
    total_len = math.hypot(x2 - x1, y2 - y1)

    if total_len == 0:
        return

    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len

    count = int(total_len // segment_length) + 1

    for i in range(count):
        a = i * segment_length
        b = min((i + 1) * segment_length, total_len)

        sx = x1 + dx * a
        sy = y1 + dy * a
        ex = x1 + dx * b
        ey = y1 + dy * b

        color = CURB_RED if i % 2 == 0 else CURB_WHITE
        pygame.draw.line(surface, color, (sx, sy), (ex, ey), thickness)


def build_static_track_surface():
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill(BACKGROUND)

    rng = random.Random(7)

    surface.fill(GRASS_BASE)
    for _ in range(2200):
        x = rng.randint(0, 995)
        y = rng.randint(0, HEIGHT - 1)
        radius = rng.randint(4, 12)
        shade = rng.randint(-14, 14)

        color = (
            clamp(GRASS_BASE[0] + shade, 0, 255),
            clamp(GRASS_BASE[1] + shade, 0, 255),
            clamp(GRASS_BASE[2] + shade, 0, 255),
        )
        pygame.draw.circle(surface, color, (x, y), radius)

    pygame.draw.rect(surface, ASPHALT_DARK, TRACK_OUTER_RECT, border_radius=34)
    pygame.draw.rect(surface, ASPHALT_MID, TRACK_OUTER_RECT.inflate(-14, -14), border_radius=28)

    pygame.draw.rect(surface, GRASS_DARK, TRACK_INNER_RECT, border_radius=28)
    pygame.draw.rect(surface, GRASS_BASE, TRACK_INNER_RECT.inflate(-12, -12), border_radius=22)

    ox, oy, ow, oh = TRACK_OUTER_RECT
    ix, iy, iw, ih = TRACK_INNER_RECT

    draw_striped_edge(surface, (ox, oy), (ox + ow, oy), thickness=10)
    draw_striped_edge(surface, (ox + ow, oy), (ox + ow, oy + oh), thickness=10)
    draw_striped_edge(surface, (ox + ow, oy + oh), (ox, oy + oh), thickness=10)
    draw_striped_edge(surface, (ox, oy + oh), (ox, oy), thickness=10)

    draw_striped_edge(surface, (ix, iy), (ix + iw, iy), thickness=10)
    draw_striped_edge(surface, (ix + iw, iy), (ix + iw, iy + ih), thickness=10)
    draw_striped_edge(surface, (ix + iw, iy + ih), (ix, iy + ih), thickness=10)
    draw_striped_edge(surface, (ix, iy + ih), (ix, iy), thickness=10)

    cx, cy, cw, ch = CENTER_LOOP_RECT
    draw_dashed_line(surface, LANE_WHITE, (cx, cy), (cx + cw, cy), 28, 18, 4)
    draw_dashed_line(surface, LANE_WHITE, (cx + cw, cy), (cx + cw, cy + ch), 28, 18, 4)
    draw_dashed_line(surface, LANE_WHITE, (cx + cw, cy + ch), (cx, cy + ch), 28, 18, 4)
    draw_dashed_line(surface, LANE_WHITE, (cx, cy + ch), (cx, cy), 28, 18, 4)

    pygame.draw.rect(surface, ASPHALT_LIGHT, TRACK_OUTER_RECT.inflate(-36, -36), width=2, border_radius=24)
    pygame.draw.rect(surface, ASPHALT_LIGHT, TRACK_INNER_RECT.inflate(18, 18), width=2, border_radius=34)

    return surface


def draw_next_checkpoint(screen, checkpoints, index):
    cp = checkpoints[index]
    pygame.draw.line(screen, CHECKPOINT_COLOR, cp[0], cp[1], 4)

    cx, cy = checkpoint_center(cp)
    pygame.draw.circle(screen, CHECKPOINT_COLOR, (int(cx), int(cy)), 6)


def draw_card(surface, rect):
    pygame.draw.rect(surface, HUD_PANEL, rect, border_radius=16)
    pygame.draw.rect(surface, HUD_BORDER, rect, width=1, border_radius=16)


def draw_meter(surface, x, y, w, h, value, color):
    pygame.draw.rect(surface, (42, 46, 54), (x, y, w, h), border_radius=8)
    pygame.draw.rect(surface, HUD_BORDER, (x, y, w, h), 1, border_radius=8)

    fill_w = int(clamp(value, 0.0, 1.0) * w)
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=8)


def create_track():
    outer = rect_to_segments(80, 80, 900, 600)
    inner = rect_to_segments(280, 200, 500, 360)
    walls = outer + inner

    checkpoints = [
        ((180, 80), (180, 200)),
        ((340, 80), (340, 200)),
        ((500, 80), (500, 200)),
        ((660, 80), (660, 200)),
        ((820, 80), (820, 200)),
        ((780, 260), (980, 260)),
        ((780, 380), (980, 380)),
        ((780, 500), (980, 500)),
        ((820, 560), (820, 680)),
        ((660, 560), (660, 680)),
        ((500, 560), (500, 680)),
        ((340, 560), (340, 680)),
        ((180, 560), (180, 680)),
        ((80, 500), (280, 500)),
        ((80, 380), (280, 380)),
        ((80, 260), (280, 260)),
    ]

    spawn_x = 140
    spawn_y = 140
    spawn_angle = 0.0

    return walls, checkpoints, spawn_x, spawn_y, spawn_angle


WALLS, CHECKPOINTS, SPAWN_X, SPAWN_Y, SPAWN_ANGLE = create_track()


class NeuralNetwork:
    def __init__(self, genome=None):
        self.genome_length = (
            INPUT_SIZE * HIDDEN_1 +
            HIDDEN_1 +
            HIDDEN_1 * HIDDEN_2 +
            HIDDEN_2 +
            HIDDEN_2 * OUTPUT_SIZE +
            OUTPUT_SIZE
        )

        if genome is None:
            genome = np.random.randn(self.genome_length).astype(np.float32) * 0.5

        self.genome = np.array(genome, dtype=np.float32)
        self.unpack()

    def unpack(self):
        idx = 0

        end = idx + INPUT_SIZE * HIDDEN_1
        self.w1 = self.genome[idx:end].reshape(INPUT_SIZE, HIDDEN_1)
        idx = end

        end = idx + HIDDEN_1
        self.b1 = self.genome[idx:end]
        idx = end

        end = idx + HIDDEN_1 * HIDDEN_2
        self.w2 = self.genome[idx:end].reshape(HIDDEN_1, HIDDEN_2)
        idx = end

        end = idx + HIDDEN_2
        self.b2 = self.genome[idx:end]
        idx = end

        end = idx + HIDDEN_2 * OUTPUT_SIZE
        self.w3 = self.genome[idx:end].reshape(HIDDEN_2, OUTPUT_SIZE)
        idx = end

        end = idx + OUTPUT_SIZE
        self.b3 = self.genome[idx:end]

    def forward(self, inputs):
        x = np.array(inputs, dtype=np.float32)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        out = np.tanh(h2 @ self.w3 + self.b3)
        return out

    def copy(self):
        return NeuralNetwork(self.genome.copy())

    def save(self, path):
        data = {"genome": self.genome.tolist()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return NeuralNetwork(np.array(data["genome"], dtype=np.float32))


class Car:
    def __init__(self, brain=None):
        self.brain = brain if brain is not None else NeuralNetwork()
        self.reset()

    def reset(self):
        self.x = float(SPAWN_X)
        self.y = float(SPAWN_Y)
        self.prev_x = self.x
        self.prev_y = self.y
        self.angle = float(SPAWN_ANGLE)
        self.speed = 0.0

        self.alive = True
        self.time_alive = 0
        self.distance_travelled = 0.0
        self.idle_counter = 0

        self.fitness = 0.0
        self.next_checkpoint = 0
        self.checkpoints_passed = 0
        self.laps = 0

        self.sensor_values = [1.0] * len(SENSOR_ANGLES)

        self.best_distance_to_cp = float("inf")
        self.last_progress_bonus = 0.0

    def sense(self, walls):
        values = []

        for offset_deg in SENSOR_ANGLES:
            ray_angle = self.angle + math.radians(offset_deg)
            end_x = self.x + math.cos(ray_angle) * MAX_SENSOR_DISTANCE
            end_y = self.y + math.sin(ray_angle) * MAX_SENSOR_DISTANCE

            closest_dist = MAX_SENSOR_DISTANCE

            for wall in walls:
                hit = segment_intersection(
                    (self.x, self.y),
                    (end_x, end_y),
                    wall[0],
                    wall[1]
                )
                if hit is not None:
                    d = distance(self.x, self.y, hit[0], hit[1])
                    if d < closest_dist:
                        closest_dist = d

            values.append(closest_dist / MAX_SENSOR_DISTANCE)

        self.sensor_values = values
        return values

    def build_inputs(self, walls, checkpoints):
        sensor_data = self.sense(walls)

        cp = checkpoints[self.next_checkpoint]
        cx, cy = checkpoint_center(cp)

        to_cp_angle = math.atan2(cy - self.y, cx - self.x)
        relative_angle = angle_wrap(to_cp_angle - self.angle)

        cp_distance = distance(self.x, self.y, cx, cy)
        cp_distance_norm = clamp(cp_distance / 500.0, 0.0, 1.0)

        speed_norm = (self.speed - MAX_REVERSE_SPEED) / (MAX_SPEED - MAX_REVERSE_SPEED)
        speed_norm = clamp(speed_norm, 0.0, 1.0)

        angle_norm = relative_angle / math.pi
        time_norm = clamp(self.time_alive / MAX_STEPS_PER_GENERATION, 0.0, 1.0)

        return sensor_data + [speed_norm, angle_norm, cp_distance_norm, time_norm]

    def think(self, walls, checkpoints):
        inputs = self.build_inputs(walls, checkpoints)
        outputs = self.brain.forward(inputs)

        steer = float(outputs[0])
        throttle = (float(outputs[1]) + 1.0) / 2.0
        brake = (float(outputs[2]) + 1.0) / 2.0

        return steer, throttle, brake

    def update(self, walls, checkpoints):
        if not self.alive:
            return

        self.prev_x = self.x
        self.prev_y = self.y

        steer, throttle, brake = self.think(walls, checkpoints)

        self.speed += throttle * ACCELERATION
        self.speed -= brake * BRAKE_FORCE

        if self.speed > 0:
            self.speed -= FRICTION
            if self.speed < 0:
                self.speed = 0
        elif self.speed < 0:
            self.speed += FRICTION * 0.5
            if self.speed > 0:
                self.speed = 0

        self.speed = clamp(self.speed, MAX_REVERSE_SPEED, MAX_SPEED)

        turn_amount = TURN_RATE * (0.35 + abs(self.speed) / MAX_SPEED)
        self.angle += steer * turn_amount

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        step_distance = distance(self.prev_x, self.prev_y, self.x, self.y)
        self.distance_travelled += step_distance
        self.time_alive += 1

        if step_distance < 0.15:
            self.idle_counter += 1
        else:
            self.idle_counter = 0

        self.check_collision(walls)
        self.check_checkpoint(checkpoints)
        self.update_progress_reward(checkpoints)
        self.update_fitness()

        if self.time_alive >= MAX_STEPS_PER_GENERATION:
            self.alive = False

        if self.idle_counter >= IDLE_LIMIT:
            self.alive = False

    def check_collision(self, walls):
        for wall in walls:
            ax, ay = wall[0]
            bx, by = wall[1]
            d = point_segment_distance(self.x, self.y, ax, ay, bx, by)
            if d <= CAR_RADIUS:
                self.alive = False
                return

        if not (0 <= self.x <= WIDTH and 0 <= self.y <= HEIGHT):
            self.alive = False

    def check_checkpoint(self, checkpoints):
        cp = checkpoints[self.next_checkpoint]
        hit = segment_intersection(
            (self.prev_x, self.prev_y),
            (self.x, self.y),
            cp[0],
            cp[1]
        )

        if hit is not None:
            self.checkpoints_passed += 1
            self.next_checkpoint += 1
            self.best_distance_to_cp = float("inf")

            if self.next_checkpoint >= len(checkpoints):
                self.next_checkpoint = 0
                self.laps += 1

    def update_progress_reward(self, checkpoints):
        cp = checkpoints[self.next_checkpoint]
        cx, cy = checkpoint_center(cp)
        d = distance(self.x, self.y, cx, cy)

        bonus = 0.0
        if d < self.best_distance_to_cp:
            improvement = self.best_distance_to_cp - d if self.best_distance_to_cp != float("inf") else 0.0
            bonus += max(0.0, improvement) * 0.8
            self.best_distance_to_cp = d

        self.last_progress_bonus = bonus

    def update_fitness(self):
        self.fitness = (
            self.checkpoints_passed * 2500.0 +
            self.laps * 25000.0 +
            self.distance_travelled * 1.2 +
            self.time_alive * 0.3 +
            max(0.0, self.speed) * 3.0 +
            self.last_progress_bonus -
            self.idle_counter * 0.1
        )

    def draw(self, screen, best=False, draw_sensors=True):
        if self.alive:
            body_color = BEST_CAR_COLOR if best else CAR_COLOR
        else:
            body_color = DEAD_CAR_COLOR

        if draw_sensors and self.alive:
            for i, offset_deg in enumerate(SENSOR_ANGLES):
                ray_angle = self.angle + math.radians(offset_deg)
                ray_dist = self.sensor_values[i] * MAX_SENSOR_DISTANCE
                end_x = self.x + math.cos(ray_angle) * ray_dist
                end_y = self.y + math.sin(ray_angle) * ray_dist

                pygame.draw.line(screen, SENSOR_COLOR, (self.x, self.y), (end_x, end_y), 1)
                pygame.draw.circle(screen, SENSOR_COLOR, (int(end_x), int(end_y)), 2)

        body_local = [
            (18, 0),
            (10, 11),
            (-10, 11),
            (-16, 7),
            (-16, -7),
            (-10, -11),
            (10, -11),
        ]

        windshield_local = [
            (8, -7),
            (13, -4),
            (13, 4),
            (8, 7),
        ]

        rear_glass_local = [
            (-9, -6),
            (-3, -5),
            (-3, 5),
            (-9, 6),
        ]

        shadow_points = world_points(self.x + 4, self.y + 4, self.angle, body_local)
        body_points = world_points(self.x, self.y, self.angle, body_local)
        windshield_points = world_points(self.x, self.y, self.angle, windshield_local)
        rear_glass_points = world_points(self.x, self.y, self.angle, rear_glass_local)

        pygame.draw.polygon(screen, CAR_SHADOW, shadow_points)
        pygame.draw.polygon(screen, body_color, body_points)
        pygame.draw.polygon(screen, (245, 245, 245), body_points, 1)

        pygame.draw.polygon(screen, CAR_GLASS, windshield_points)
        pygame.draw.polygon(screen, (120, 150, 170), rear_glass_points)

        wheel_centers = [
            (7, -12),
            (7, 12),
            (-8, -12),
            (-8, 12),
        ]

        for wx, wy in wheel_centers:
            rx, ry = rotate_point(wx, wy, self.angle)
            pygame.draw.circle(screen, TIRE_COLOR, (int(self.x + rx), int(self.y + ry)), 3)

        for hx, hy in [(16, -5), (16, 5)]:
            rx, ry = rotate_point(hx, hy, self.angle)
            pygame.draw.circle(screen, HEADLIGHT_COLOR, (int(self.x + rx), int(self.y + ry)), 2)

        if self.speed < 1.0:
            for bx, by in [(-14, -5), (-14, 5)]:
                rx, ry = rotate_point(bx, by, self.angle)
                pygame.draw.circle(screen, BRAKE_LIGHT_COLOR, (int(self.x + rx), int(self.y + ry)), 2)


class GeneticTrainer:
    def __init__(self):
        self.generation = 1
        self.steps = 0

        self.population = [Car() for _ in range(POPULATION_SIZE)]

        self.best_fitness_ever = -float("inf")
        self.best_brain_ever = None
        self.best_laps_ever = 0

        self.history_best = []
        self.history_avg = []

        self.replay_car = None
        self.replay_mode = False

    def update(self):
        if self.replay_mode:
            if self.replay_car is not None and self.replay_car.alive:
                self.replay_car.update(WALLS, CHECKPOINTS)
            else:
                self.start_replay()
            return

        alive_count = 0
        for car in self.population:
            if car.alive:
                car.update(WALLS, CHECKPOINTS)
            if car.alive:
                alive_count += 1

        self.steps += 1

        if alive_count == 0 or self.steps >= MAX_STEPS_PER_GENERATION:
            self.evolve()

    def evolve(self):
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        best = self.population[0]
        avg_fitness = sum(c.fitness for c in self.population) / len(self.population)

        self.history_best.append(best.fitness)
        self.history_avg.append(avg_fitness)

        if best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = best.fitness
            self.best_brain_ever = best.brain.copy()
            self.best_laps_ever = best.laps

        print(
            f"Gen {self.generation:03d} | "
            f"Best: {best.fitness:.2f} | "
            f"Avg: {avg_fitness:.2f} | "
            f"CP: {best.checkpoints_passed} | "
            f"Laps: {best.laps}"
        )

        next_population = []

        for i in range(ELITE_COUNT):
            elite = self.population[i].brain.copy()
            next_population.append(Car(elite))

        parent_pool = self.population[: max(12, POPULATION_SIZE // 3)]

        while len(next_population) < POPULATION_SIZE:
            parent_a = self.tournament_select(parent_pool)
            parent_b = self.tournament_select(parent_pool)

            child_genome = self.crossover(parent_a.brain.genome, parent_b.brain.genome)
            child_genome = self.mutate(child_genome)

            next_population.append(Car(NeuralNetwork(child_genome)))

        self.population = next_population
        self.steps = 0
        self.generation += 1

    def tournament_select(self, pool):
        selected = random.sample(pool, min(TOURNAMENT_SIZE, len(pool)))
        selected.sort(key=lambda c: c.fitness, reverse=True)
        return selected[0]

    def crossover(self, genome_a, genome_b):
        alpha = np.random.uniform(-CROSSOVER_BLEND, 1.0 + CROSSOVER_BLEND, size=len(genome_a))
        child = genome_a * alpha + genome_b * (1.0 - alpha)

        mask = np.random.rand(len(genome_a)) < 0.5
        child = np.where(mask, child, genome_b)
        return child.astype(np.float32)

    def mutate(self, genome):
        genome = genome.copy()

        mutation_mask = np.random.rand(len(genome)) < MUTATION_RATE
        noise = np.random.randn(len(genome)).astype(np.float32) * MUTATION_STRENGTH
        genome[mutation_mask] += noise[mutation_mask]

        if random.random() < 0.08:
            idx = random.randint(0, len(genome) - 1)
            genome[idx] += np.random.randn() * 0.8

        return genome

    def best_current_car(self):
        return max(self.population, key=lambda c: c.fitness)

    def alive_count(self):
        return sum(1 for c in self.population if c.alive)

    def save_best(self):
        if self.best_brain_ever is not None:
            self.best_brain_ever.save(MODEL_FILE)
            print(f"Saved best model to {MODEL_FILE}")

    def load_best(self):
        if os.path.exists(MODEL_FILE):
            self.best_brain_ever = NeuralNetwork.load(MODEL_FILE)
            print(f"Loaded best model from {MODEL_FILE}")
            return True
        return False

    def start_replay(self):
        if self.best_brain_ever is None:
            loaded = self.load_best()
            if not loaded:
                return

        self.replay_mode = True
        self.replay_car = Car(self.best_brain_ever.copy())

    def stop_replay(self):
        self.replay_mode = False
        self.replay_car = None


def draw_text(screen, font, text, x, y, color=TEXT_COLOR):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def draw_graph(screen, trainer, x, y, w, h):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (35, 40, 48), rect, border_radius=12)
    pygame.draw.rect(screen, HUD_BORDER, rect, 1, border_radius=12)

    if len(trainer.history_best) < 2:
        return

    values_best = trainer.history_best[-40:]
    values_avg = trainer.history_avg[-40:]

    max_val = max(max(values_best), max(values_avg))
    min_val = min(min(values_best), min(values_avg))

    if abs(max_val - min_val) < 1e-6:
        max_val += 1.0

    def to_points(values):
        pts = []
        for i, v in enumerate(values):
            px = x + 10 + (i / max(1, len(values) - 1)) * (w - 20)
            py = y + h - 10 - ((v - min_val) / (max_val - min_val)) * (h - 20)
            pts.append((px, py))
        return pts

    avg_points = to_points(values_avg)
    best_points = to_points(values_best)

    if len(avg_points) >= 2:
        pygame.draw.lines(screen, (120, 180, 255), False, avg_points, 2)
    if len(best_points) >= 2:
        pygame.draw.lines(screen, (255, 220, 90), False, best_points, 2)


def draw_hud(screen, font, small_font, trainer, show_all, draw_sensors, sim_steps):
    panel_rect = pygame.Rect(995, 0, 205, HEIGHT)
    pygame.draw.rect(screen, HUD_PANEL, panel_rect)
    pygame.draw.line(screen, HUD_BORDER, (995, 0), (995, HEIGHT), 2)

    card1 = pygame.Rect(1006, 16, 183, 220)
    card2 = pygame.Rect(1006, 246, 183, 212)
    card3 = pygame.Rect(1006, 468, 183, 316)

    draw_card(screen, card1)
    draw_card(screen, card2)
    draw_card(screen, card3)

    draw_text(screen, font, "RACE HUD", 1035, 28, HUD_ACCENT)

    y = 62
    if trainer.replay_mode:
        car = trainer.replay_car
        draw_text(screen, small_font, "Mode: REPLAY", 1018, y)
        y += 28
        if car:
            draw_text(screen, small_font, f"Alive: {1 if car.alive else 0}", 1018, y)
            y += 24
            draw_text(screen, small_font, f"Speed: {car.speed:.2f}", 1018, y)
            y += 24
            draw_text(screen, small_font, f"Checkpoint: {car.checkpoints_passed}", 1018, y)
            y += 24
            draw_text(screen, small_font, f"Laps: {car.laps}", 1018, y)
            y += 24
            draw_text(screen, small_font, f"Fitness: {car.fitness:.1f}", 1018, y)
            y += 34
            draw_text(screen, small_font, "Throttle", 1018, y)
            y += 22
            draw_meter(screen, 1018, y, 160, 12, clamp(car.speed / MAX_SPEED, 0.0, 1.0), HUD_ACCENT)
    else:
        best = trainer.best_current_car()
        draw_text(screen, small_font, "Mode: TRAINING", 1018, y)
        y += 28
        draw_text(screen, small_font, f"Generation: {trainer.generation}", 1018, y)
        y += 24
        draw_text(screen, small_font, f"Alive: {trainer.alive_count()}/{POPULATION_SIZE}", 1018, y)
        y += 24
        draw_text(screen, small_font, f"Steps: {trainer.steps}/{MAX_STEPS_PER_GENERATION}", 1018, y)
        y += 24
        draw_text(screen, small_font, f"Best now: {best.fitness:.1f}", 1018, y)
        y += 24
        draw_text(screen, small_font, f"Best ever: {trainer.best_fitness_ever:.1f}", 1018, y)
        y += 24
        draw_text(screen, small_font, f"Laps ever: {trainer.best_laps_ever}", 1018, y)
        y += 24
        draw_text(screen, small_font, f"Checkpoint: {best.checkpoints_passed}", 1018, y)
        y += 34
        draw_text(screen, small_font, "Speed", 1018, y)
        y += 22
        draw_meter(screen, 1018, y, 160, 12, clamp(best.speed / MAX_SPEED, 0.0, 1.0), HUD_ACCENT)

    draw_text(screen, font, "VIEW", 1070, 258, HUD_ACCENT)
    draw_text(screen, small_font, f"All cars: {show_all}", 1018, 300)
    draw_text(screen, small_font, f"Sensors: {draw_sensors}", 1018, 330)
    draw_text(screen, small_font, f"Sim speed: {sim_steps}", 1018, 360)
    draw_text(screen, small_font, "Next checkpoint shown", 1018, 390, TEXT_DIM)

    draw_text(screen, font, "CONTROLS", 1032, 480, HUD_ACCENT)
    controls = [
        "SPACE  toggle all/best",
        "D      toggle sensors",
        "UP     faster training",
        "DOWN   slower training",
        "S      save best brain",
        "L      load best brain",
        "R      replay best brain",
        "T      back to training",
    ]
    y = 520
    for line in controls:
        draw_text(screen, small_font, line, 1018, y)
        y += 28

    draw_graph(screen, trainer, 1010, 650, 170, 120)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Advanced Self-Driving Car Simulator - Realistic UI")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 24)
    small_font = pygame.font.SysFont("consolas", 18)

    trainer = GeneticTrainer()
    track_surface = build_static_track_surface()

    show_all = True
    draw_sensors = True
    sim_steps = SIMULATION_STEPS_PER_FRAME

    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_all = not show_all
                elif event.key == pygame.K_d:
                    draw_sensors = not draw_sensors
                elif event.key == pygame.K_UP:
                    sim_steps = min(30, sim_steps + 1)
                elif event.key == pygame.K_DOWN:
                    sim_steps = max(1, sim_steps - 1)
                elif event.key == pygame.K_s:
                    trainer.save_best()
                elif event.key == pygame.K_l:
                    trainer.load_best()
                elif event.key == pygame.K_r:
                    trainer.start_replay()
                elif event.key == pygame.K_t:
                    trainer.stop_replay()

        for _ in range(sim_steps):
            trainer.update()

        screen.blit(track_surface, (0, 0))

        if trainer.replay_mode:
            if trainer.replay_car is not None:
                draw_next_checkpoint(screen, CHECKPOINTS, trainer.replay_car.next_checkpoint)
                trainer.replay_car.draw(screen, best=True, draw_sensors=draw_sensors)
        else:
            best_car = trainer.best_current_car()
            draw_next_checkpoint(screen, CHECKPOINTS, best_car.next_checkpoint)

            if show_all:
                for car in trainer.population:
                    car.draw(screen, best=(car is best_car), draw_sensors=draw_sensors)
            else:
                best_car.draw(screen, best=True, draw_sensors=draw_sensors)

        draw_hud(screen, font, small_font, trainer, show_all, draw_sensors, sim_steps)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
