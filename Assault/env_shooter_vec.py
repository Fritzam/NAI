import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

WIDTH = 600
HEIGHT = 400

PLAYER_Y = HEIGHT - 40
PLAYER_SPEED = 6
BULLET_SPEED = 10
ENEMY_SPEED = 2


class ShooterEnvVec(gym.Env):
    """
    Środowisko Gym reprezentujące prostą grę typu shooter.

    Agent steruje pozycją gracza i może oddawać strzały w kierunku przeciwnika.

    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        """
        Inicjalizacja środowiska.
        
        """
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.window = None
        self.clock = None

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resetuje środowisko do stanu początkowego.

        """
        super().reset(seed=seed)

        self.player_x = WIDTH // 2
        self.bullet_active = False
        self.bullet_x = 0
        self.bullet_y = 0

        self.enemy_x = np.random.randint(50, WIDTH - 50)
        self.enemy_y = 50
        self.enemy_dir = 1

        self.steps = 0
        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        """

        Na podstawie podanej akcji:
        - przesuwa gracza,
        - obsługuje strzał i ruch pocisku,
        - aktualizuje pozycję przeciwnika,
        - sprawdza kolizje,
        - oblicza nagrodę.
        
        """
        reward = -0.001
        self.steps += 1

        """ Ruch gracza """
        if action == 1:
            self.player_x -= PLAYER_SPEED
        elif action == 2:
            self.player_x += PLAYER_SPEED
        self.player_x = np.clip(self.player_x, 20, WIDTH - 20)

        """ Wystrzał pocisku """
        if action == 3 and not self.bullet_active:
            self.bullet_active = True
            self.bullet_x = self.player_x
            self.bullet_y = PLAYER_Y

        """ Odświeżenie ruchu pocisku"""
        if self.bullet_active:
            self.bullet_y -= BULLET_SPEED
            if self.bullet_y < 0:
                self.bullet_active = False

        """ Ruch nieprzyjaciela"""
        self.enemy_x += ENEMY_SPEED * self.enemy_dir
        if self.enemy_x < 20 or self.enemy_x > WIDTH - 20:
            self.enemy_dir *= -1
            self.enemy_y += 20

        """ Kolizja"""
        hit = False
        if self.bullet_active:
            dist = np.hypot(
                self.bullet_x - self.enemy_x,
                self.bullet_y - self.enemy_y
            )
            if dist < 20:
                hit = True

        if hit:
            reward += 1.0
            self.bullet_active = False
            self.enemy_x = np.random.randint(50, WIDTH - 50)
            self.enemy_y = 50

        """ Zakończenie"""
        terminated = False
        truncated = self.steps > 3000

        return self._get_obs(), reward, terminated, truncated, {}


    def _get_obs(self):
        """
        Buduje wektor na podstawie aktualnego stanu gry.

        Zwraca:
            np.ndarray: wektor obserwacji.
        """
        def norm(x, maxv):
            return (x / maxv) * 2.0 - 1.0

        bullet_y = self.bullet_y if self.bullet_active else HEIGHT

        return np.array([
            norm(self.player_x, WIDTH),
            norm(bullet_y, HEIGHT),
            norm(self.enemy_x, WIDTH),
            norm(self.enemy_y, HEIGHT),
        ], dtype=np.float32)

    def render(self):
        """

        Rysuje:
        - statek gracza,
        - przeciwnika,
        - pocisk (jeśli jest aktywny).
        """
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        self.window.fill((10, 10, 20))

        """ gracz """
        pygame.draw.rect(
            self.window, (100, 200, 255),
            (self.player_x - 20, PLAYER_Y, 40, 15)
        )

        """ wróg """
        pygame.draw.rect(
            self.window, (255, 80, 80),
            (self.enemy_x - 15, self.enemy_y - 15, 30, 30)
        )

        """ nabój """
        if self.bullet_active:
            pygame.draw.circle(
                self.window, (255, 255, 0),
                (int(self.bullet_x), int(self.bullet_y)), 4
            )

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        """
        Zamyka zasoby graficzne i kończy działanie Pygame.
        """
        if self.window is not None:
            pygame.quit()