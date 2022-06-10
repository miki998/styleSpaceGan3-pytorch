from copy import deepcopy

def edit_image_stylegan2(sample_gen,ys,rgb_ys,alpha,c,L,raw=False):
        modif = deepcopy(ys)
        modif[L][0,c] = modif[L][0,c] + alpha
        return sample_gen.generate_image_from_ys(modif, rgb_ys, raw=raw)[0]

def edit_image_stylegan3(sample_gen,ys,alpha,c,L,raw=False):
        modif = deepcopy(ys)
        modif[L][0,c] = modif[L][0,c] + alpha
        return sample_gen.generate_image_from_ys(modif, raw=raw)[0]