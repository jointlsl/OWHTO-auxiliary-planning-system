

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def getPointsFromHeatmap(arr):
    ''' 
        arr: numpy.ndarray, channel x imageshape
        ret: [(x,y..)]* channel
    '''
    points = []
    for img in arr:
        index = img.argmax()
        points.append(unravel_index(index, img.shape))
    return points
##ondef