import random
import os, shutil

def save_files(route, shape, max_number):
    format = ".png"
    new_route = "shapes/dataset/"
    for file in range(260):
        shape_file = str(random.randint(0,max_number)) + format
        saved_file = os.path.join(route,shape_file)
        print(shape_file)
        if os.path.isfile(saved_file) and not os.path.isfile(os.path.join(new_route, shape + shape_file)) and not os.path.isfile(os.path.join(new_route,shape_file)):
            shutil.copy(saved_file, new_route)
            new_file = shape + shape_file
            os.rename(new_route + shape_file, new_route + new_file)
            ctr += 1
    print(f"Number of {shape}s created: {ctr}")

save_files("shapes/triangle/", "triangle_", 3719)
save_files("shapes/circle/", "circle_", 3719)
save_files("shapes/square/", "square_", 3764)
save_files("shapes/star/", "star_", 3764)
'''
circle files: 3720 ~= 24.85%
square files: 3765 ~= 25.15%
triangle files: 3720
star files: 365
'''
