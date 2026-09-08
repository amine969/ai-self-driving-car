import pickle
import numpy as np
import pygame
import math
import os
from network import NeuralNetwork
from utils import scale_image, blit_rotate_center


Red_car = scale_image(pygame.image.load("imgs/red-car.png"), 0.8)

win = pygame.display.set_mode((1000, 800))
pygame.display.set_caption("ai_rasing_gam")

gen = 409
start_time = pygame.time.get_ticks()

cars = 100

class AbstractCar:
    def __init__(self, max_vel, rotatuon_vel, brain=None):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotatuon_vel
        self.angel = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.rect = self.img.get_rect(topleft=(self.x, self.y))
        self.alive = True
        self.fitness = 0
        
        if brain is not None:
            self.brain = brain
        else:
            self.brain = NeuralNetwork()
        
    def rotate(self, left=False, right=False):
        if self.vel > 0:
            current_rotation_vel = self.rotation_vel * (self.vel / self.max_vel)
            if left:
                self.angel += current_rotation_vel  
            elif right:
                self.angel -= current_rotation_vel  

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angel)
    
    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 4, 0)
        
    def mov(self):
        radians = math.radians(self.angel)
        vertical = math.cos(radians) * self.vel
        horizantal = math.sin(radians) * self.vel
        self.y -= vertical
        self.x -= horizantal
        self.rect.topleft = (self.x, self.y)
    
    def get_mask(self):
            return pygame.mask.from_surface(self.img)


    def cast_ray(self, win, obstacle_mask, angle_offset, start_pos, max_distance=200):
        ray_angle = math.radians(self.angel + angle_offset)
        
        for dist in range(0, max_distance, 2):
            ray_x = start_pos[0] - math.sin(ray_angle) * dist
            ray_y = start_pos[1] - math.cos(ray_angle) * dist
            
            try:
                if obstacle_mask.get_at((int(ray_x), int(ray_y))):
                    #pygame.draw.circle(win, (255, 0, 0), (int(ray_x), int(ray_y)), 5)
                    return dist
            except IndexError:
                pass
                
            #pygame.draw.line(win, (0, 255, 0), start_pos, (int(ray_x), int(ray_y)), 1)
            
        return max_distance

class PlayerCar(AbstractCar):
    IMG = Red_car
    START_POS = (30, 600) 

def draw(win, cars, obstacle, obstacle_mask):
    win.fill((0, 0, 0)) 
    
    win.blit(track_surface, (0, 0))
    for car in cars:
        if car.alive:
            car_center = car.rect.center
                
            rad = math.radians(car.angel)
    
            half_length = car.rect.height / 2
            half_width = car.rect.width / 4
    
            front_edge = (car_center[0] - math.sin(rad) * half_length, car_center[1] - math.cos(rad) * half_length)
            front_left_edge = (front_edge[0] - math.cos(rad) * half_width, front_edge[1] + math.sin(rad) * half_width)
            front_right_edge = (front_edge[0] + math.cos(rad) * half_width, front_edge[1] - math.sin(rad) * half_width)
            left_edge = (car_center[0] - math.cos(rad) * half_width, car_center[1] + math.sin(rad) * half_width)
            right_edge = (car_center[0] + math.cos(rad) * half_width, car_center[1] - math.sin(rad) * half_width)

            left_90_dist = car.cast_ray(win, obstacle_mask, angle_offset=90, start_pos=left_edge)
            left_45_dist = car.cast_ray(win, obstacle_mask, angle_offset=45, start_pos=front_left_edge)
            front_dist = car.cast_ray(win, obstacle_mask, angle_offset=0, start_pos=front_edge)
            right_45_dist = car.cast_ray(win, obstacle_mask, angle_offset=-45, start_pos=front_right_edge)
            right_90_dist = car.cast_ray(win, obstacle_mask, angle_offset=-90, start_pos=right_edge)

            car.sensors = [left_90_dist, left_45_dist, front_dist, right_90_dist, right_45_dist]
    
            car.draw(win) 
            
    pygame.display.update()

def evolve(population):
    population.sort(key=lambda car: car.fitness, reverse=True)
    best_car = population[0]
    
    new_population = []
    new_population.append(PlayerCar(4, 4, brain=best_car.brain))
    
    for _ in range(cars - 1):

        child_car = PlayerCar(4, 4)
        
        child_car.brain.dense1.weight = np.copy(best_car.brain.dense1.weight)
        child_car.brain.dense1.biases = np.copy(best_car.brain.dense1.biases)
        
        child_car.brain.dense2.weight = np.copy(best_car.brain.dense2.weight)
        child_car.brain.dense2.biases = np.copy(best_car.brain.dense2.biases)
        
        child_car.brain.dense3.weight = np.copy(best_car.brain.dense3.weight)
        
        child_car.brain.dense3.biases = np.copy(best_car.brain.dense3.biases)
        
        child_car.brain.dense1.weight += np.random.randn(*best_car.brain.dense1.weight.shape) * 0.04
        child_car.brain.dense1.biases += np.random.randn(*best_car.brain.dense1.biases.shape) * 0.04
        
        child_car.brain.dense2.weight += np.random.randn(*best_car.brain.dense2.weight.shape) * 0.04
        child_car.brain.dense2.biases += np.random.randn(*best_car.brain.dense2.biases.shape) * 0.04
        
        child_car.brain.dense3.weight += np.random.randn(*best_car.brain.dense3.weight.shape) * 0.04
        child_car.brain.dense3.biases += np.random.randn(*best_car.brain.dense3.biases.shape) * 0.04


        new_population.append(child_car)
    return new_population
        
def save_best_brain(brain):
    with open("brain.pkl", "wb") as f:
        pickle.dump(brain, f)

run = True
FPS = 60
clock = pygame.time.Clock()
population = [PlayerCar(4, 4) for _ in range(cars)]

obstacle_rect = pygame.Rect(400, 300, 100, 100)

track_surface = pygame.image.load("imgs/track9.png")
track_surface.set_colorkey((0, 0, 0))
obstacle_mask = pygame.mask.from_surface(track_surface)

#obstacle_surface = pygame.Surface((100, 100))
#obstacle_surface.fill((255, 255, 255))

if os.path.exists("brain.pkl"):
    with open("brain.pkl", "rb") as f:
        saved_brain = pickle.load(f)
    
    population = []
    population.append(PlayerCar(4, 4, brain=saved_brain))
    
    for _ in range(cars - 1):
        child_car = PlayerCar(4, 4)
        
        child_car.brain.dense1.weight = np.copy(saved_brain.dense1.weight)
        child_car.brain.dense1.biases = np.copy(saved_brain.dense1.biases)
        
        child_car.brain.dense2.weight = np.copy(saved_brain.dense2.weight)
        child_car.brain.dense2.biases = np.copy(saved_brain.dense2.biases)
        
        child_car.brain.dense3.weight = np.copy(saved_brain.dense3.weight)
        child_car.brain.dense3.biases = np.copy(saved_brain.dense3.biases)
        
        child_car.brain.dense1.weight += np.random.randn(*saved_brain.dense1.weight.shape) * 0.04
        child_car.brain.dense1.biases += np.random.randn(*saved_brain.dense1.biases.shape) * 0.04
        
        child_car.brain.dense2.weight += np.random.randn(*saved_brain.dense2.weight.shape) * 0.04
        child_car.brain.dense2.biases += np.random.randn(*saved_brain.dense2.biases.shape) * 0.04
        
        child_car.brain.dense3.weight += np.random.randn(*saved_brain.dense3.weight.shape) * 0.04
        child_car.brain.dense3.biases += np.random.randn(*saved_brain.dense3.biases.shape) * 0.04
        
        population.append(child_car)
        

else:
    population = [PlayerCar(4, 4) for _ in range(cars)]


while run:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
    
    current_time = pygame.time.get_ticks()
    if current_time - start_time >= 36000: 
        start_time = current_time
        
        for car in population:
            car.alive = False



    for car in population:
        if car.alive:  
            old_x, old_y = car.x, car.y
            if hasattr(car, 'sensors'):
                inputs = np.array([car.sensors])
                outputs = car.brain.forward(inputs)
                
                if outputs[0][0] > 0.5:
                    car.rotate(left=True)
                if outputs[0][1] > 0.5:
                    car.rotate(right=True)
                if outputs[0][2] > 0.5:
                    car.move_forward()
                else:
                    car.reduce_speed()
                car.mov()
                car.fitness += car.vel

            else:
                car.mov()
                car.fitness += car.vel
            
            
            if hasattr(car, 'sensors'):
                if min(car.sensors) < 5:
                    car.vel = 0
                    car.alive = False

    alive_cars = [car for car in population if car.alive]
    if len(alive_cars) == 0:
        population.sort(key=lambda car: car.fitness, reverse=True)
        save_best_brain(population[0].brain) 
        population = evolve(population)
        gen += 1
        print(f"gen nbr: {gen}")

    draw(win, population, obstacle_rect, obstacle_mask)

pygame.quit()