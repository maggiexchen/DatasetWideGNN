import numpy

# median absolute deviation scaling
def MAD_norm(dist1, dist2):
    dist2_median = numpy.median(dist2)
    dist2_MAD = numpy.median(abs(dist2-dist2_median))
    norm_dist1 = (dist1 - dist2_median)/dist2_MAD
    norm_dist2 = (dist2 - dist2_median)/dist2_MAD

    return norm_dist1, norm_dist2


# minmax normalisation that scales distribution to be between 0 and 1
def minmax(dist, d_min, d_max):
    norm_dist = (dist - d_min)/(d_max-d_min)
    return norm_dist


# The reverse of minmax that returns the original values
def reverse_minmax(norm_value, d_min, d_max):
    value = norm_value * (d_max-d_min) + d_min
    return value
