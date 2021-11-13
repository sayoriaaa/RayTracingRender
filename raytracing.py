# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:28:41 2021

@author: sayori
"""

import numpy as np
from numpy import array as ar
import math
from PIL import Image
import tqdm


ambient=0.05

def normalize(x):
    return x/np.linalg.norm(x)

class Objects:
    
    objects_num=0
    objects_item=[]
    
    def __init__(self):
        Objects.objects_num+=1
        Objects.objects_item.append(self)
        
class Triangular_mesh(Objects):  
    def __init__(self,filename):
        super().__init__()
        self.vertices = []
        self.vn_vertices = []#法线
        self.uv_vertices = []#贴图
        self.uv_indices = []
        self.indices = []#对应哪三个顶点构成一个面，从1开始
        self.is_poly=False

        with open(filename,encoding='utf-8') as f:
            for line in f:
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")]
                    self.vertices.append(ar([x, y, z]))
                    
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")]
                    self.uv_vertices.append(ar([u, v]))
                    
                elif line.startswith("vn "):
                    x, y, z = [float(d) for d in line.strip("vn").strip().split(" ")]
                    self.vn_vertices.append(ar([x, y, z]))

                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    if len(facet)==3:
                        self.indices.append([int(d[0]) for d in facet])
                        self.uv_indices.append([int(d[1]) for d in facet])                   
                    elif len(facet)==4: 
                        self.is_poly=True
                        self.indices.append([int(d[0]) for d in facet[:3]])
                        self.indices.append([int(d[0]) for d in [facet[0],facet[2],facet[3]]])
                        self.uv_indices.append([int(d[1]) for d in facet[:3]])
                        self.uv_indices.append([int(d[1]) for d in [facet[0],facet[2],facet[3]]])
                        
            if self.is_poly:
                z=[]
                for i in self.vn_vertices:
                    z.append(i)
                    z.append(i)
                self.vn_vertices=z#等待换写法
                
    def set_shade(self,color=ar([1.,.5,.6]),reflection=.85,diffuse=1.,specular_c=.6,specular_k=50):
        self.color=color
        self.reflection=reflection
        self.diffuse=diffuse
        self.specular_c=specular_c
        self.specular_k=specular_k
    
        
    def intersect_tri_time(self,tri,direction,camera):
        matrix_A=ar([[tri[0][0]-tri[1][0],tri[0][0]-tri[2][0],direction[0]],[tri[0][1]-tri[1][1],tri[0][1]-tri[2][1],direction[1]],[tri[0][2]-tri[1][2],tri[0][2]-tri[2][2],direction[2]]])#P78
        matrix_T=ar([[tri[0][0]-tri[1][0],tri[0][0]-tri[2][0],tri[0][0]-camera[0]],[tri[0][1]-tri[1][1],tri[0][1]-tri[2][1],tri[0][1]-camera[1]],[tri[0][2]-tri[1][2],tri[0][2]-tri[2][2],tri[0][2]-camera[2]]])
    
        matrix_beta=ar([[tri[0][0]-camera[0],tri[0][0]-tri[2][0],direction[0]],[tri[0][1]-camera[1],tri[0][1]-tri[2][1],direction[1]],[tri[0][2]-camera[2],tri[0][2]-tri[2][2],direction[2]]])
        matrix_gama=ar([[tri[0][0]-tri[1][0],tri[0][0]-camera[0],direction[0]],[tri[0][1]-tri[1][1],tri[0][1]-camera[1],direction[1]],[tri[0][2]-tri[1][2],tri[0][2]-camera[2],direction[2]]])
    
        det_A=np.linalg.det(matrix_A)
        det_T=np.linalg.det(matrix_T)
    
        det_beta=np.linalg.det(matrix_beta)
        det_gama=np.linalg.det(matrix_gama)
        
        if abs(det_A)<1e-6 :
            return np.inf
        else:
            beta=det_beta/det_A
            gama=det_gama/det_A
            if gama<0 or gama>1 or beta<0 or beta>1-gama:
                return np.inf
            return det_T/det_A
    
    def get_color_blinn(self,intersect_point,light,camera,norm,intensity):
        l=light-intersect_point
        l=normalize(l)
        h=(light-intersect_point)+(camera-intersect_point)
        h=normalize(h)
        
        diffuse_light=self.diffuse*intensity*self.color*max(np.dot(l,norm))#intensity,color至少一个为np.array
        specular_light=self.specular_c*intensity*math.pow(max(np.dot(h,norm),0), self.specular_k)
       
        return specular_light+diffuse_light#当trangular_mesh不止一个又要改
    
    def get_color_ambient(self):
        ambient_light=ambient*self.color
        return ambient_light
               
        
    def intersect_time(self,direction,camera):
        t_min=np.inf
        tri_norm_min=None
        
        for count,i in enumerate(self.indices):
            tri=ar([self.vertices[idx-1] for idx in i])
            normal=self.vn_vertices[count]
            t=self.intersect_tri_time(tri,direction,camera) 
            
            if t<t_min and t>0:#找到最小获得插值
                t_min=t
                tri_norm_min=normal
                
        if t_min<np.inf:
            self.min_point=camera+t_min*direction
            self.min_norm=tri_norm_min
            return t_min
        return np.inf

class Plane(Objects):  
    def __init__(self,center=ar([0,-1,0]),norm=ar([0,1,0])):  
        super().__init__()
        self.center=center
        self.normal=norm
        
    def set_shade(self,color=ar([.1,.2,.3]),reflection=.85,diffuse=1.,specular_c=.6,specular_k=50):
        self.color=color
        self.reflection=reflection
        self.diffuse=diffuse
        self.specular_c=specular_c
        self.specular_k=specular_k
        
    def intersect_time(self,direction,camera):
        delta_d=np.dot(direction,self.normal)
        if abs(delta_d)<1e-6:
            return np.inf
        t=np.dot((self.center-camera),self.normal)/delta_d
        if t<np.inf and t>0:
            self.min_point=camera+t*direction
            self.min_norm=self.normal
            return max(t,0)
        return np.inf
    
    def get_color_blinn(self,intersect_point,light,camera,norm,intensity):
        l=light-intersect_point
        l=normalize(l)
        h=(light-intersect_point)+(camera-intersect_point)
        h=normalize(h)
        
        diffuse_light=self.diffuse*intensity*self.color*max(np.dot(l,norm),0)#intensity,color至少一个为np.array
        specular_light=self.specular_c*intensity*math.pow(max(np.dot(h,norm),0), self.specular_k)
       
        return specular_light+diffuse_light#当trangular_mesh不止一个又要改
    
    def get_color_ambient(self):
        ambient_light=ambient*self.color
        return ambient_light
    
class Light:
    light_num=0
    light_item=[]
    def __init__(self,position=ar([2,5,5]),intensity=ar([1.,1.,1.]),mode=1):
        Light.light_num+=1
        Light.light_item.append(self)
        self.position=position
        self.intensity=intensity
        self.mode=mode
        
class Camera:
    def __init__(self,position=ar([2.,2.,5.]),target=ar([0,0,0]),height=625,width=475,focal=25):
        self.position=position
        self.target=target
        self.height=height
        self.width=width
        
        self.equivant_focal=focal*width/35
        
        
    def generate_canvas(self):
        up=ar([0,1,0])
        gaze=normalize(self.target-self.position)
        u=normalize(np.cross(up,-1*gaze))
        v=np.cross(-1*gaze,u)
        canvas_center=self.position+self.equivant_focal*gaze
        self.canvas_startpoint=canvas_center+self.height/2*v-self.width/2*u
        self.u=u
        self.v=v
        
    def get_direction(self,i,j):
        point_at=self.canvas_startpoint+i*self.u-j*self.v
        return normalize(point_at-self.position)
        
class Scene:
    def __init__(self,filename="results/ray_trace_12.png",width=900,height=900):
        self.img=Image.new("RGB",(width,height),(0,0,0))
        self.filename=filename
        
    def show(self):
        self.img.show()
        
    def save(self):
        self.img.save(self.filename)
        
        
def get_color(start,direction,intensity,light):
    is_in_shadow=False
    t_min=np.inf
    object_min=None
    color=ar([0.,0.,0.])
    for i in Objects.objects_item:
        time=i.intersect_time(direction,start)
        if time<t_min:
            t_min=time
            object_min=i
    if t_min==np.inf or max(intensity)<0.2:
        return color
    color+=i.get_color_ambient()
    '''
    做阴影测试
    '''
    time_limit=np.linalg.norm(light-object_min.min_point)
    point_to_light_direction=normalize(light-i.min_point)
    for i in Objects.objects_item:
        if i!=object_min:
            t=i.intersect_time(point_to_light_direction,object_min.min_point+point_to_light_direction*.00001)
            if t<time_limit:
                is_in_shadow=True
                break
    if not is_in_shadow:
        color+=i.get_color_blinn(object_min.min_point,light,start,object_min.min_norm,intensity)
    
    reflect_ray=direction-2*np.dot(direction,object_min.min_norm)*object_min.min_norm
    color+=i.reflection*get_color(object_min.min_point+reflect_ray*.00001, reflect_ray, intensity, light)#87
    
    return np.clip(color,0,1)
            
  
    
                
if __name__=="__main__":
    height=625
    width=475
    a=Triangular_mesh("model/sph.obj") 
    a.set_shade()

    b=Plane() 
    b.set_shade()
    
    l=Light()
        
    scene=Scene()
    camera=Camera(height=height,width=width)
    camera.generate_canvas()
    
    for i in tqdm.tqdm(range(width)):
        for j in range(height):
            direction=camera.get_direction(i, j)
            color=get_color(camera.position, direction, l.intensity, l.position)
            scene.img.putpixel((i,j), (int(color[0]*255),int(color[1]*255),int(color[2]*255)))
            
    scene.save()
    scene.show()
    
            

            
            
    
    

          
        
        
    