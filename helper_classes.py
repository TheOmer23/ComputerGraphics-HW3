import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    v = np.array([0,0,0])
    v = vector - 2 * (np.dot(vector, axis) * axis)
    return v

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection_point):
        #we did
        return Ray(intersection_point,-self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        #TODO
        pass

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        #TODO
        pass


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        # TODO

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        #TODO
        pass

    def get_distance_from_light(self, intersection):
        #TODO
        pass

    def get_intensity(self, intersection):
        #TODO
        pass


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        min_t = np.inf
        #we did
        for obj in objects:
            if obj.intersect(self) != None:
                t = obj.intersect(self)[0]
                if t < min_t:
                    min_t = t
                    nearest_object = obj
        p = self.origin + min_t * self.direction
        min_distance = np.linalg.norm(p - self.origin)
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()
        self.plane = Plane(self.normal, self.a)

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        #we did
        vecAB = self.b - self.a
        vecAC = self.c - self.a
        cross_product = np.cross(vecAB, vecAC)
        return normalize(cross_product)

    def intersect(self, ray: Ray):
        #we did
        if self.plane.intersect(ray) is None:
            return None
        t = self.plane.intersect(ray)[0]
        p = ray.origin + t * ray.direction
        vecAB = self.b - self.a
        vecAC = self.c - self.a
        areaABC = np.linalg.norm(np.cross(vecAB , vecAC)) / 2
        vecPA = self.a - p
        vecPB = self.b - p
        vecPC = self.c - p
        alpha = (np.linalg.norm(np.cross(vecPB, vecPC))) / (2 * areaABC)
        beta = (np.linalg.norm(np.cross(vecPA, vecPC))) / (2 * areaABC)
        gamma = 1 - alpha - beta
        if alpha > 1 or alpha < 0:
            return None
        if beta > 1 or beta < 0:
            return None
        if gamma > 1 or gamma < 0:
            return None
        if alpha + beta + gamma != 1:
            return None
        return t, self

class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> D
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                [4,1,0],
                [4,2,1],
                [2,4,0]]
        for i in t_idx:
            l.append(Triangle(self.v_list[i[0]], self.v_list[i[1]], self.v_list[i[2]]))
        # l.append(Triangle(self.v_list[0], self.v_list[1], self.v_list[3]))
        # l.append(Triangle(self.v_list[1], self.v_list[2], self.v_list[3]))
        # l.append(Triangle(self.v_list[0], self.v_list[3], self.v_list[2]))
        # l.append(Triangle(self.v_list[4], self.v_list[1], self.v_list[0]))
        # l.append(Triangle(self.v_list[4], self.v_list[2], self.v_list[1]))
        # l.append(Triangle(self.v_list[2], self.v_list[4], self.v_list[0]))
        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)
        
    def intersect(self, ray: Ray):
        min_t = np.inf
        hit_obj = None
        for triangle in self.triangle_list:
            if triangle.intersect(ray) != None:
                t = triangle.intersect(ray)
                if t < min_t:
                    min_t = t
                    hit_obj = triangle
        if hit_obj is None:
            return None
        return min_t, hit_obj



class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        p0 = ray.origin
        v = ray.direction
        q = self.center
        r = self.radius
        
        a = np.dot(v, v)
        b = 2 * np.dot(v, p0 - q)
        c = np.dot(p0 - q, p0 - q) - r**2

        delta = b**2 - 4 * a * c

        if delta < 0:
            return None

        sqrt_delta = np.sqrt(delta)
        t1 = (-b - sqrt_delta) / (2 * a)
        t2 = (-b + sqrt_delta) / (2 * a)

        if t1 >= 0 and t2 >= 0:
            return min(t1, t2), self
        elif max(t1, t2) >= 0:
            return max(t1, t2), self
        return None

