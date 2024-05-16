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

            color = get_color(scene, ray, 1, max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def get_color(scene, ray: Ray, depth, max_depth):
    if depth > max_depth:
        return np.zeros(3)
    hit_obj, obj_distance, hit_point = ray.nearest_intersected_object(scene['objects'])
    if hit_obj is None:
        return np.zeros(3)
    color = np.zeros(3) + calc_ambient(scene, hit_obj)
    for light in scene['lights']:
        if calc_sj(scene, hit_point, light):
            color = color + calc_diffuse(hit_obj, hit_point, light, ray) + calc_specular(hit_obj, hit_point, ray, light)
    
    curr_depth = depth + 1
    if curr_depth > max_depth:
        return color
    
    r_ray = construct_reflective_ray(hit_obj, hit_point, ray)
    color = color + hit_obj.reflection * get_color(scene, r_ray, curr_depth, max_depth)
    return color 

    
def calc_ambient(scene, hit_obj):
    material_ambient = hit_obj.ambient
    global_ambient = scene['ambient']
    return (material_ambient * global_ambient)


def calc_diffuse(hit_obj, hit_point, light, ray: Ray):
    norm = np.zeros(3)
    if isinstance(hit_obj, Sphere):
        norm = normalize(hit_point - hit_obj.center)
    elif isinstance(hit_obj, Pyramid):
        triangle = hit_obj.intersect(ray)[1]
        norm = triangle.normal
    else:
        norm = hit_obj.normal  

    material_diffuse = np.array(hit_obj.diffuse)
    light_ray = light.get_light_ray(hit_point)
    light_intensity = light.get_intensity(hit_point)
    diffuse_intensity = np.dot(light_ray.direction, norm)
    
    return material_diffuse * diffuse_intensity * light_intensity  


def calc_specular(hit_obj, hit_point, ray: Ray, light):
    norm = np.zeros(3)
    if isinstance(hit_obj, Sphere):
        norm = normalize(hit_point - hit_obj.center)
    elif isinstance(hit_obj, Pyramid):
        triangle = hit_obj.intersect(ray)[1]
        norm = triangle.normal
    else:
        norm = hit_obj.normal

    v = -ray.direction
    reflected_light = normalize(reflected(-light.get_light_ray(hit_point).direction, norm))
    v_dot_r_pow_shininess = np.dot(v, reflected_light) ** hit_obj.shininess
    material_specular =  np.array(hit_obj.specular)
    light_intensity = light.get_intensity(hit_point)
    return material_specular * v_dot_r_pow_shininess * light_intensity


def calc_sj(scene, hit_point, light):
    epsilon = 1e-6
    direction_to_light = normalize(light.get_light_ray(hit_point).direction)
    biased_hit_point = hit_point + epsilon * direction_to_light
    hit_to_light_ray = Ray(biased_hit_point, direction_to_light)
    hit_to_light_dist = light.get_distance_from_light(hit_point)
    nearest_obj, dist, _ = hit_to_light_ray.nearest_intersected_object(scene['objects'])
    if nearest_obj is None or dist >= hit_to_light_dist:
        return 1
    return 0


def construct_reflective_ray(hit_obj, hit_point, ray: Ray):
    norm = np.zeros(3)
    if isinstance(hit_obj, Sphere):
        norm = normalize(hit_point - hit_obj.center)
    elif isinstance(hit_obj, Pyramid):
        triangle = hit_obj.intersect(ray)[1]
        norm = triangle.normal
    else:
        norm = hit_obj.normal
    r_ray = Ray(hit_point, reflected(ray.direction, norm))
    return r_ray


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])

    # Light sources
    lights = [
        DirectionalLight(intensity=0.8, direction=np.array([1, -1, -2])),
        PointLight(intensity=0.6, position=np.array([2, 10, -5]), kc=1, kl=0.1, kq=0.01)
    ]

    # Objects in the scene
    objects = [
        Plane(normal=np.array([0, 1, 0]), point=np.array([0, -1, 0])),  # Horizontal plane at y = -1
        Sphere(center=np.array([-2, 1, -3]), radius=1),  # Sphere on the left
        Sphere(center=np.array([2, 0, -3]), radius=0.5),  # Sphere on the right
        Sphere(center=np.array([0, 2, -3]), radius=0.25)  # Smaller sphere above
    ]

    # Setting materials for the objects
    objects[1].set_material(
        ambient=np.array([1, 0, 0]),  # Red
        diffuse=np.array([1, 0.5, 0.5]),
        specular=np.array([1, 1, 1]),
        shininess=50,
        reflection=0.5
    )
    
    objects[2].set_material(
        ambient=np.array([0, 0, 1]),  # Blue
        diffuse=np.array([0.5, 0.5, 1]),
        specular=np.array([1, 1, 1]),
        shininess=30,
        reflection=0.3
    )

    objects[3].set_material(
        ambient=np.array([0.64, 1, 0.12]),  # Green
        diffuse=np.array([0.5, 0.8, 1]),
        specular=np.array([1, 1, 1]),
        shininess=30,
        reflection=1
    )

    objects[0].set_material(
        ambient=np.array([0.2, 0.2, 0.2]),  # Dark grey plane
        diffuse=np.array([0.6, 0.6, 0.6]),
        specular=np.array([0.5, 0.5, 0.5]),
        shininess=10,
        reflection=0.8
    )

    # Adjusted vertices for the pyramid to be closer to the camera and visible
    v_list = np.array(
    [
        [-0.5, -0.142, -0.998],
        [-0.034, 0.092, -0.145],
        [0.484, 0.031, -0.998],
        [-0.104, 0.851, -0.828],
        [0.23, -0.833, -0.591]
    ])

    pyramid = Pyramid(v_list)
    pyramid.set_material([0.5, -0.4, 0.1], [0.1, 0.1, -0.7], [0.3, 0.8, 0.3], 5, 0.5)
    pyramid.apply_materials_to_triangles()
    objects.append(pyramid)

    return camera, lights, objects

