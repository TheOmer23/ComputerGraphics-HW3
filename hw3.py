from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    
    scene = {
    'objects': objects,
    'lights': lights,
    'ambient': ambient
    }

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            hit_obj, obj_distance, hit_point = ray.nearest_intersected_object(scene['objects'])
            color = np.zeros(3)
            
            # This is the main loop where each pixel color is computed.
            color = get_color(scene, ray, hit_obj, hit_point, 0, max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image

def get_color(scene, ray, hit_obj, hit_point, depth, max_depth):
    if depth > max_depth:
        return np.zeros(3)
 
    if hit_obj is None:
        return np.zeros(3)
    color = np.zeros(3) + calc_ambient(scene, hit_obj)

    for light in scene['lights']:
        if calc_sj(scene, hit_point, light):
            color = color + calc_diffuse(hit_obj, hit_point, light) + calc_specular(hit_obj, hit_point, ray, light)
    
    curr_depth = depth + 1
    if curr_depth > max_depth:
        return color
    
    r_ray = ConstructReflectiveRay(hit_obj, hit_point, ray)
    color = color + hit_obj.reflection * get_color(scene, r_ray, hit_obj, hit_point, curr_depth, max_depth)
    return color 

    
def calc_ambient(scene, hit_obj):
    material_ambient = hit_obj.ambient
    global_ambient = scene['ambient']
    return material_ambient * global_ambient

def calc_diffuse(hit_obj, hit_point, light):
    norm = np.zeros(3)
    if isinstance(hit_obj, Sphere):
        norm = normalize(hit_point - hit_obj.center)
    else:
        norm = hit_obj.normal # If the object is not a sphere it is already normalized by helper_class
    

    material_diffuse = np.array(hit_obj.diffuse)
    light_ray = light.get_light_ray(hit_point)
    light_intensity = np.array(light.get_intensity(hit_point))
    

    diffuse_intensity = max(np.dot(light_ray.direction, norm), 0)
    
    return (diffuse_intensity * material_diffuse * light_intensity) # light_ray = -light.direction (L = -L)

def calc_specular(hit_obj, hit_point, ray, light):
    norm = np.zeros(3)
    if isinstance(hit_obj, Sphere):
        norm = normalize(hit_point - hit_obj.center)
    else:
        norm = hit_obj.normal
    v = -ray.direction
    reflected_light = np.array(reflected(-(light.get_light_ray(hit_point).direction), norm))
    v_dot_r_pow_shininess = max(np.dot(v, reflected_light), 0) ** hit_obj.shininess
    return np.array(hit_obj.specular) * v_dot_r_pow_shininess * np.array(light.get_intensity(hit_point) )
    
def calc_sj(scene, hit_point, light):
    hit_to_light_ray = light.get_light_ray(hit_point)
    hit_to_light_dist = light.get_distance_from_light(hit_point)
    t = hit_to_light_ray.nearest_intersected_object(scene['objects'])[1]
    if t < hit_to_light_dist:
        return 0 
    return 1

def ConstructReflectiveRay(hit_obj, hit_point, ray): 
    norm = np.zeros(3)
    if isinstance(hit_obj, Sphere):
        norm = normalize(hit_point - hit_obj.center)
    else:
        norm = hit_obj.normal
    return reflected(ray.direction, norm)







# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
